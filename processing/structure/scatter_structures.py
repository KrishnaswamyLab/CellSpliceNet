"""Graph scattering of the folded structures (backbone + base-pair edges) -> per-event [N, 11, 4] moments (structure step 2)."""
import os
import argparse
import pickle as pk
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from processing.common import add_location_args, work_dir_for  # noqa: E402
from processing.scatter import Scatter  # noqa: E402


def dot2pairs(db):
    """Dot-bracket -> list of (i, j) base pairs (0-indexed), stack parser."""
    stack, pairs = [], []
    for i, c in enumerate(db):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            pairs.append((stack.pop(), i))
    return pairs


def dotbracket_to_edge_index(db_str):
    """RNA graph: backbone edges (i -- i+1) plus base-pair edges, both directions."""
    N = len(db_str)
    i = torch.arange(max(N - 1, 0))
    parts = [torch.stack([i, i + 1])]
    pairs = dot2pairs(db_str)
    if pairs:
        parts.append(torch.from_numpy(np.asarray(pairs, dtype=np.int64).T))
    ei = torch.cat(parts, dim=1)
    return torch.cat([ei, ei[[1, 0]]], dim=1)


def moments_gpu(scatter_coeffs, num_moments=4):
    """Mean/std/skew/kurtosis along the last dim; std uses unbiased (N-1) to match the
    original torch.std default; skew/kurtosis use the biased (N) divisor to match
    scipy.stats.{skew,kurtosis} defaults. Returns CPU tensor."""
    mean = scatter_coeffs.mean(dim=-1)
    moments = torch.zeros(*scatter_coeffs.shape[:-1], num_moments,
                          device=scatter_coeffs.device, dtype=scatter_coeffs.dtype)
    moments[..., 0] = mean
    moments[..., 1] = scatter_coeffs.std(dim=-1)
    if num_moments > 2:
        centered = scatter_coeffs - mean.unsqueeze(-1)
        var_biased = centered.pow(2).mean(dim=-1, keepdim=True)
        std_biased = var_biased.sqrt().clamp(min=1e-12)
        z = centered / std_biased
        moments[..., 2] = z.pow(3).mean(dim=-1)
        if num_moments > 3:
            moments[..., 3] = z.pow(4).mean(dim=-1) - 3.0
    return moments.cpu()


def run_batch(scattering, eye_N, edge_indices_cpu, N, device):
    """Block-diagonal-batch B graphs of N nodes each; one Scatter forward; per-event moments on CPU."""
    B = len(edge_indices_cpu)
    offsets = torch.arange(B, dtype=torch.long) * N  # CPU; small
    edge_index = torch.cat([ei + off for ei, off in zip(edge_indices_cpu, offsets)], dim=1).to(device, non_blocking=True)
    x = eye_N.repeat(B, 1)
    data = Data(x=x, edge_index=edge_index)
    scatter_coeffs = scattering(data)  # (B*N, 11, N)
    sc = scatter_coeffs.view(B, N, scatter_coeffs.shape[1], scatter_coeffs.shape[2])
    return moments_gpu(sc)  # (B, N, 11, 4) on CPU


def main():
    parser = argparse.ArgumentParser(description='Compute graph scattering moments from folded structures (batched).')
    add_location_args(parser)
    parser.add_argument('--batch-size', type=int, default=32, help='Events per Scatter forward pass')
    parser.add_argument('--dryrun', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    verbose = args.verbose or args.dryrun

    def vlog(msg):
        if verbose:
            print(f'[verbose] {msg}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vlog(f'Using device: {device}, batch_size={args.batch_size}')

    base_dir = work_dir_for(args)
    input_filename = os.path.join(base_dir, 'processed_sequences.csv')
    output_file = os.path.join(base_dir, 'structure_scattering_moments.pkl')
    dict_file = os.path.join(base_dir, 'structure_dict.pkl')

    vlog(f'Reading {input_filename}')
    df = pd.read_csv(input_filename)
    records = df.to_dict('records')
    n_events = len(records)
    vlog(f'Loaded {n_events} rows')

    scatters = {}

    def scatter_for(N):
        """Scatter module and identity features for graphs of N nodes (worm windows vary in length)."""
        if N not in scatters:
            sc = Scatter(in_channels=N, max_graph_size=N).to(device).eval()
            scatters[N] = (sc, torch.eye(N, dtype=torch.float32, device=device))
        return scatters[N]

    n_done = 0
    if args.resume and os.path.exists(output_file) and not args.dryrun:
        truncate_at = 0
        with open(output_file, 'rb') as f:
            while True:
                try:
                    pk.load(f)
                    truncate_at = f.tell()
                    n_done += 1
                except EOFError:
                    break
                except Exception:
                    break
        if truncate_at > 0:
            with open(output_file, 'r+b') as f:
                f.truncate(truncate_at)
        vlog(f'Resume: {n_done} events already pickled; truncated to last clean boundary')

    structure_dictionary = {
        'structure_seq': [r['sequence'] for r in records],
        'secondary_structure': [r['secondary_structure'] for r in records],
        'structure_scattering_coeffs': [],
        'gene_id': [r['gene_id'] for r in records],
        'event_id': [r['event_id'] for r in records],
        'exon_coordinates': [(r['exon_start'], r['exon_end']) for r in records],
        'intron_coordinates': [(r['intron_start'], r['intron_end']) for r in records],
    }

    if args.dryrun:
        rec = records[0]; N = len(rec['secondary_structure'])
        with torch.no_grad():
            moments = run_batch(*scatter_for(N), [dotbracket_to_edge_index(rec['secondary_structure'])], N, device)
        print(f'[dryrun] OK: moments shape {tuple(moments.shape)} (B, N, 11, 4)')
        return

    file_mode = 'ab' if args.resume and n_done > 0 else 'wb'
    with open(output_file, file_mode) as f, torch.no_grad():
        pbar = tqdm(total=n_events - n_done)
        i = n_done
        while i < n_events:
            # batch consecutive events that share a graph size
            N = len(records[i]['secondary_structure'])
            j = i
            while j < n_events and j - i < args.batch_size and len(records[j]['secondary_structure']) == N:
                j += 1
            eis = [dotbracket_to_edge_index(records[k]['secondary_structure']) for k in range(i, j)]
            moments_batched = run_batch(*scatter_for(N), eis, N, device)  # (B, N, 11, 4) on CPU
            for k in range(j - i):
                pk.dump(moments_batched[k].clone(), f)
            pbar.update(j - i)
            i = j
        pbar.close()

    vlog(f'Writing dictionary to {dict_file}')
    with open(dict_file, 'wb') as f:
        pk.dump(structure_dictionary, f)
    vlog('Done.')


if __name__ == "__main__":
    main()
