"""Fold the 500 bp splice-junction windows with RNAfold (ViennaRNA) -> processed_sequences.csv (structure step 1)."""
import os
import argparse
import RNA
import torch
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from processing.common import add_location_args, work_dir_for  # noqa: E402


MAX_SEQUENCE_LEN = 1000
ALPHABET = 'NAGTC'  # 0=N/PAD, 1=A, 2=G, 3=T, 4=C  (matches pth_to_fasta.py)


def read_event_coords(path):
    ext = os.path.splitext(path)[1].lower()
    sep = '\t' if ext in ['.tsv', '.txt'] else ','
    return pd.read_csv(path, sep=sep)


def load_pth_tokens(path):
    payload = torch.load(path, map_location='cpu')
    if not isinstance(payload, dict):
        raise ValueError(
            f'Expected dict payload (id -> token tensor) in {path}, got {type(payload).__name__}'
        )
    tokens_map = {}
    for k, v in payload.items():
        vt = torch.as_tensor(v)
        if vt.ndim == 1:
            tokens = vt.long()
        elif vt.ndim == 2 and vt.shape[0] == 1:
            tokens = vt.squeeze(0).long()
        elif vt.ndim == 2:
            tokens = vt[0].long()
        else:
            raise ValueError(f'Unexpected tensor shape {tuple(vt.shape)} for key {k!r}')
        tokens_map[str(k)] = tokens
    return tokens_map


def decode_tokens(tokens):
    return ''.join(ALPHABET[t] if 0 <= t < len(ALPHABET) else 'N' for t in tokens.tolist())


def build_task(row, tokens_map):
    if row['exon_start'] < row['intron_start'] or row['exon_end'] > row['intron_end']:
        return None

    event_id = str(row['event_id'])
    tokens = tokens_map.get(event_id)
    if tokens is None:
        return None

    tensor_len = int(tokens.shape[0])
    if MAX_SEQUENCE_LEN < tensor_len:
        excess = tensor_len - MAX_SEQUENCE_LEN
        start = excess // 2
        end = start + MAX_SEQUENCE_LEN
        window = tokens[start:end]
    else:
        window = tokens

    sequence = decode_tokens(window)
    if not sequence:
        return None
    return row, sequence


def build_task_gene_roi(row, tokens_map):
    """C. elegans windows: the intron-exon-intron span when it is shorter than MAX_SEQUENCE_LEN,
    otherwise MAX_SEQUENCE_LEN bases centred on the exon (clipped to the gene), cut from the gene
    sequence (encoded_sequences.pth, keyed by gene_id)."""
    if row['exon_start'] < row['intron_start'] or row['exon_end'] > row['intron_end']:
        return None
    tokens = tokens_map.get(str(row['gene_id']))
    if tokens is None:
        return None
    L = MAX_SEQUENCE_LEN
    if row['intron_end'] - row['intron_start'] < L:
        start, end = row['intron_start'], row['intron_end']
    else:
        center = (row['exon_start'] + row['exon_end']) // 2
        start, end = center - L // 2, center + L // 2
        if start < 0:
            start, end = 0, L
        end = min(end, row['gene_length'])
    sequence = decode_tokens(tokens[int(start):int(end)])
    return (row, sequence) if sequence else None


def fold_one(task):
    row, sequence = task
    fc = RNA.fold_compound(sequence)
    ss, mfe = fc.mfe()
    return f"{row['event_id']},{row['gene_id']},{row['exon_start']},{row['exon_end']},{row['intron_start']},{row['intron_end']},{mfe},{sequence},{ss}\n"


def main():
    global MAX_SEQUENCE_LEN

    parser = argparse.ArgumentParser(description='Fold sequence windows from an encoded .pth and write processed_sequences.csv')
    add_location_args(parser)
    parser.add_argument('--encoded-pth', default=None, help='encoded SJ windows (default: <data_root>/<organism>/processed/enc_sj_data.pth)')
    parser.add_argument('--event-coords-file', default=None, help='event coordinates CSV/TSV (default: <data_root>/<organism>/processed/events_coordinates.*)')
    parser.add_argument('--limit', type=int, default=0, help='fold only the first N events (test runs)')
    parser.add_argument('--window', choices=['sj', 'gene_roi'], default=None,
                        help='sj: centre of the splice-junction window in enc_sj_data.pth (human default); '
                             'gene_roi: intron-exon-intron region of the gene sequence (C. elegans default)')
    parser.add_argument('--max-sequence-len', type=int, default=MAX_SEQUENCE_LEN, help='Max fold-window length; if smaller than the per-event tensor, the tensor is center-trimmed. Otherwise the whole tensor is folded.')
    parser.add_argument('--dryrun', action='store_true', help='Fold a single example and print the result without writing the output file')
    parser.add_argument('--verbose', action='store_true', help='Print step-by-step progress messages (auto-enabled when --dryrun is set)')
    parser.add_argument('--resume', action='store_true', help='Skip events already present in the existing output CSV (truncates any partial trailing line)')
    args = parser.parse_args()

    verbose = args.verbose or args.dryrun

    def vlog(msg):
        if verbose:
            print(f'[verbose] {msg}')

    MAX_SEQUENCE_LEN = args.max_sequence_len
    proc = Path(args.data_root) / args.organism / 'processed'
    args.window = args.window or ('gene_roi' if args.organism == 'c_elegans' else 'sj')
    default_pth = 'encoded_sequences.pth' if args.window == 'gene_roi' else 'enc_sj_data.pth'
    args.encoded_pth = args.encoded_pth or str(proc / default_pth)
    make_task = build_task_gene_roi if args.window == 'gene_roi' else build_task
    if args.event_coords_file is None:
        args.event_coords_file = str(next(p for p in (proc / 'events_coordinates.csv', proc / 'events_coordinates.tsv') if p.exists()))
    output_file = os.path.join(work_dir_for(args), 'processed_sequences.csv')
    vlog(f'MAX_SEQUENCE_LEN={MAX_SEQUENCE_LEN}')
    vlog(f'output_file={output_file}')

    vlog(f'Loading event coordinates from {args.event_coords_file}')
    event_coords_df = read_event_coords(args.event_coords_file)
    event_coords_df['event_id'] = event_coords_df['event_id'].astype(str)
    event_coords_df['gene_id'] = event_coords_df['gene_id'].astype(str)
    vlog(f'Loaded {len(event_coords_df)} event rows; columns={list(event_coords_df.columns)}')

    vlog(f'Loading encoded sequences from {args.encoded_pth}')
    tokens_map = load_pth_tokens(args.encoded_pth)
    vlog(f'Loaded encoded tokens for {len(tokens_map)} entries')

    if verbose:
        sample_pth = list(tokens_map.keys())[:5]
        sample_event = event_coords_df['event_id'].head(5).tolist()
        sample_shapes = [tuple(tokens_map[k].shape) for k in sample_pth]
        vlog(f'Sample .pth keys: {sample_pth}')
        vlog(f'Sample .pth tensor shapes: {sample_shapes}')
        vlog(f'Sample event-table event_ids: {sample_event}')
        event_set = set(event_coords_df['event_id'].unique())
        pth_set = set(tokens_map.keys())
        overlap = event_set & pth_set
        vlog(f'ID overlap: {len(overlap)} of {len(event_set)} unique event_ids found in .pth ({len(pth_set)} pth keys)')
        if not overlap:
            vlog('WARNING: zero overlap — event_id formats do not match between events CSV and .pth')

    resume_ids = set()
    resume_active = args.resume and not args.dryrun and os.path.exists(output_file)
    if resume_active:
        with open(output_file, 'r') as rf:
            lines = rf.readlines()
        header = lines[0] if lines else ''
        complete_data_lines = [ln for ln in lines[1:] if ln.endswith('\n')]
        resume_ids = {ln.split(',', 1)[0] for ln in complete_data_lines if ln.split(',', 1)[0]}
        with open(output_file, 'w') as wf:
            wf.write(header)
            wf.writelines(complete_data_lines)
        vlog(f'Resume: {len(resume_ids)} events already in {output_file} (file truncated to complete lines)')

    vlog('Building fold tasks')
    tasks = []
    missing_events = 0
    skipped_data_errors = 0
    skipped_resume = 0
    for row in event_coords_df.to_dict('records'):
        if row['event_id'] in resume_ids:
            skipped_resume += 1
            continue
        task = make_task(row, tokens_map)
        if task is None:
            if (row['gene_id'] if args.window == 'gene_roi' else row['event_id']) not in tokens_map:
                missing_events += 1
            else:
                skipped_data_errors += 1
            continue
        tasks.append(task)
        if args.dryrun or (args.limit and len(tasks) >= args.limit):
            break
    vlog(f'Built {len(tasks)} tasks ({skipped_resume} skipped: already in output, {missing_events} skipped: event not in .pth, {skipped_data_errors} skipped: data error or empty window)')

    if not tasks:
        raise ValueError('No tasks produced — check gene_id format alignment between events CSV and .pth.')

    if args.dryrun:
        row, sequence = tasks[0]
        tensor_len = int(tokens_map.get(row['event_id'], tokens_map.get(str(row['gene_id']))).shape[0])
        vlog(f"first example: event_id={row['event_id']}, gene_id={row['gene_id']}, "
             f"exon=[{row['exon_start']},{row['exon_end']}], intron=[{row['intron_start']},{row['intron_end']}], "
             f"tensor_length={tensor_len}, fold_window_len={len(sequence)}")
        print('[dryrun] Folding the first task.')
        result = fold_one(tasks[0])
        print('[dryrun] event_id,gene_id,exon_start,exon_end,intron_start,intron_end,minimum_free_energy,sequence,secondary_structure')
        print(f'[dryrun] {result.rstrip()}')
        return

    if not resume_active:
        if os.path.exists(output_file):
            vlog(f'Removing existing output file {output_file}')
            os.remove(output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    n_workers = cpu_count()
    vlog(f'Starting Pool with {n_workers} workers, chunksize=10')

    file_mode = 'a' if resume_active else 'w'
    with open(output_file, file_mode) as f:
        if not resume_active:
            f.write("event_id,gene_id,exon_start,exon_end,intron_start,intron_end,minimum_free_energy,sequence,secondary_structure\n")
        with Pool(processes=n_workers) as pool:
            with tqdm(total=len(tasks)) as pbar:
                for result in pool.imap(fold_one, tasks, chunksize=10):
                    if result is not None:
                        f.write(result)
                    pbar.update()
    vlog('Done.')


if __name__ == "__main__":
    main()
