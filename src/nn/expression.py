import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from data.dataloader import neuron_type_fn


# Module-level caches. prep_expression is called once per forward pass per
# sample, and the original GraphExpressionModality reloaded the per-neuron
# scatter tensor from disk every time — same shape of bug as the structure
# pickle, just smaller per-file. Cache by absolute path so we tolerate
# multiple ablation roots in the same process.
_SCATTER_CACHE = {}
_MEANVEC_CACHE = {}
_GRAPH_CACHE = {}


def _load_scatter(scatter_dir: Path, neuron: str, device):
    """Worm legacy: scatter_coeffs_<neuron>.pt. New ablation dirs: <neuron>.pt."""
    key = (str(scatter_dir), neuron)
    if key not in _SCATTER_CACHE:
        candidates = [
            scatter_dir / f"scatter_coeffs_{neuron}.pt",
            scatter_dir / f"{neuron}.pt",
        ]
        for p in candidates:
            if p.exists():
                _SCATTER_CACHE[key] = torch.load(p, map_location="cpu", weights_only=False)
                break
        else:
            raise FileNotFoundError(f"No scatter file for neuron {neuron} in {scatter_dir} (tried {[str(c) for c in candidates]})")
    return _SCATTER_CACHE[key].to(device, non_blocking=True)


def _load_mean_vec(mean_dir: Path, neuron: str, device):
    key = (str(mean_dir), neuron)
    if key not in _MEANVEC_CACHE:
        p = mean_dir / f"{neuron}_mean.pt"
        if not p.exists():
            raise FileNotFoundError(f"Missing mean vec {p}")
        _MEANVEC_CACHE[key] = torch.load(p, map_location="cpu", weights_only=False)
    return _MEANVEC_CACHE[key].to(device, non_blocking=True)


def _load_graph(graph_dir: Path, neuron: str):
    """Load PyG Data from <neuron>_graph.pt (x, edge_index, edge_attr).
    Returns the CPU Data; caller handles device/batching."""
    key = (str(graph_dir), neuron)
    if key not in _GRAPH_CACHE:
        p = graph_dir / f"{neuron}_graph.pt"
        if not p.exists():
            raise FileNotFoundError(f"Missing graph file {p}")
        _GRAPH_CACHE[key] = torch.load(p, map_location="cpu", weights_only=False)
    return _GRAPH_CACHE[key]


class ExpAttention(nn.Module):
    def __init__(self, n_splice_factors, n_neurons):
        super().__init__()

        self.neuron_dictionary = neuron_type_fn()
        self.alphas = nn.Parameter(torch.empty([n_neurons, n_splice_factors]))
        nn.init.kaiming_uniform_(self.alphas, a=math.sqrt(5))

    def forward(self, x, neuron_list):
        x = rearrange(x, 'b n c s -> b n (c s)')
        index = [self.neuron_dictionary[neuron] for neuron in neuron_list]
        alphas_att = self.alphas[index, :].softmax(1)
        attn_output = x * alphas_att.unsqueeze(-1)
        return attn_output.sum(1), alphas_att


class _ExpressionGlobAttnMixin:
    """Shared post-transformer attention-pool over expression tokens.

    All expression encoder variants need this — the Zorro transformer outputs
    per-gene embeddings and the pool turns them into a single z_exp vector.
    """

    def _build_glob_attn(self):
        self.exp_glob_attn_module = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            Rearrange("b l c -> b c l"),
            nn.BatchNorm1d(64),
            nn.GELU(),
            Rearrange("b c l -> b l c"),
            nn.Linear(64, 1, bias=False),
            Rearrange("b l c -> b (c l)"),
            nn.Softmax(dim=1),
        )

    def exp_glob_attn_op(self, output_exp_embed, mask=None, output_glob_attn=False):
        b_size, exp_len, hidden_dim = output_exp_embed.shape
        for indx, layer in enumerate(self.exp_glob_attn_module):
            if indx == 0:
                h = layer(output_exp_embed)
            else:
                h = layer(h)

        att_h = h.reshape(b_size, 1, exp_len)
        z_rep = torch.bmm(att_h, output_exp_embed)
        z_rep = z_rep.reshape(b_size, hidden_dim)

        if output_glob_attn:
            return z_rep, att_h.reshape(b_size, exp_len)
        else:
            return z_rep


class GraphExpressionModality(nn.Module, _ExpressionGlobAttnMixin):
    """Scattering-coefficient encoder (paper anchor). Each gene is a token of
    11 scattering coeffs concatenated across the 242-dim coeff matrix axis,
    projected to hidden_dim."""

    def __init__(
        self,
        exp_dim,
        coeff_dim,
        hidden_dim,
        gene_embed_bool,
        bin_exp,
        ntype_feature_bool,
        expression_data_root,
        save_output_hook,
        scatter_coeffs_dir=None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.exp_dim = exp_dim
        self.bin_exp = bin_exp
        self.ntype_feature_bool = ntype_feature_bool
        self.gene_embed_bool = gene_embed_bool
        self.coeff_dim = coeff_dim
        self.expression_data_root = expression_data_root
        self.save_output_hook = save_output_hook

        if scatter_coeffs_dir:
            self.scatter_dir = Path(scatter_coeffs_dir)
        else:
            p = Path(expression_data_root)
            self.scatter_dir = p.parent / "scatter_coeffs_gtex" if p.suffix == ".tsv" else p

        self.expression_dim = int(exp_dim) * 11
        self.to_expression_embedding = nn.Sequential(
            nn.Linear(self.expression_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        self._build_glob_attn()

    def prep_expression(self, metadata):
        device = next(self.to_expression_embedding.parameters()).device
        expression_data = []
        for metadata_i in metadata:
            nueron_i = metadata_i["neuron"]
            scatter_of_neuron_i = _load_scatter(self.scatter_dir, nueron_i, device)
            scatter_of_neuron_i = rearrange(scatter_of_neuron_i, "a b c -> a c b").unsqueeze(0)
            expression_data.append(scatter_of_neuron_i)

        expression_data = torch.cat(expression_data, dim=0).to(device)
        expression_data = rearrange(expression_data, "a b c1 c2 -> a b (c1 c2)")
        embd_expression = self.to_expression_embedding(expression_data)

        return embd_expression, expression_data.mean(-1).unsqueeze(-1)


class MeanVecExpressionModality(nn.Module, _ExpressionGlobAttnMixin):
    """MLP baseline (reviewer's 'simpler encoder; MLP on expression vector').

    Input: <neuron>_mean.pt, a [G] tensor of per-gene mean expression.
    Each gene is a token; its scalar value is projected to hidden_dim via a
    small per-gene MLP. Output shape matches GraphExpressionModality:
    tokens [B, G, hidden_dim], context [B, G, 1]."""

    def __init__(
        self,
        exp_dim,
        coeff_dim,
        hidden_dim,
        gene_embed_bool,
        bin_exp,
        ntype_feature_bool,
        expression_data_root,
        save_output_hook,
        scatter_coeffs_dir=None,
        mean_vec_dir=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.exp_dim = exp_dim
        self.bin_exp = bin_exp
        self.ntype_feature_bool = ntype_feature_bool
        self.gene_embed_bool = gene_embed_bool
        self.coeff_dim = coeff_dim
        self.save_output_hook = save_output_hook

        if mean_vec_dir is None:
            raise ValueError("MeanVecExpressionModality requires mean_vec_dir")
        self.mean_dir = Path(mean_vec_dir)

        # Per-gene learnable embedding so the model can still distinguish genes
        # even though the scalar feature alone is rank-1.
        self.gene_embed = nn.Embedding(int(exp_dim), hidden_dim)
        self.to_expression_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self._build_glob_attn()

    def prep_expression(self, metadata):
        device = next(self.to_expression_embedding.parameters()).device
        vecs = []
        for metadata_i in metadata:
            neuron = metadata_i["neuron"]
            v = _load_mean_vec(self.mean_dir, neuron, device)  # [G]
            vecs.append(v.unsqueeze(0))
        vecs = torch.cat(vecs, dim=0).to(device)              # [B, G]
        g = vecs.shape[1]
        expression_data = vecs.unsqueeze(-1)                  # [B, G, 1]
        proj = self.to_expression_embedding(expression_data)  # [B, G, H]
        gene_ids = torch.arange(g, device=device).unsqueeze(0).expand(vecs.shape[0], -1)
        embd_expression = proj + self.gene_embed(gene_ids)
        return embd_expression, expression_data


class GraphNNExpressionModality(nn.Module, _ExpressionGlobAttnMixin):
    """GNN baseline (reviewer's 'GNN baseline').

    Uses the precomputed PyG Data files under <ablation>/<metric>/graphs/:
      - x: [G, G] per-gene-per-cell denoised expression
      - edge_index: [2, E] MI/correlation graph edges
      - edge_attr: [E] edge weights

    Runs 2-layer GCNConv with edge weights and emits per-gene embeddings
    as tokens of shape [B, G, hidden_dim]."""

    def __init__(
        self,
        exp_dim,
        coeff_dim,
        hidden_dim,
        gene_embed_bool,
        bin_exp,
        ntype_feature_bool,
        expression_data_root,
        save_output_hook,
        scatter_coeffs_dir=None,
        mean_vec_dir=None,
        graph_metric_dir=None,
    ):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.hidden_dim = hidden_dim
        self.exp_dim = int(exp_dim)
        self.bin_exp = bin_exp
        self.gene_embed_bool = gene_embed_bool
        self.save_output_hook = save_output_hook

        if graph_metric_dir is None:
            raise ValueError("GraphNNExpressionModality requires graph_metric_dir")
        self.graph_dir = Path(graph_metric_dir)

        # node feature dim = G (per-cell expression per gene)
        self.node_feat_dim = self.exp_dim
        self.gcn1 = GCNConv(self.node_feat_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
        self.norm = nn.LayerNorm(hidden_dim)

        self._build_glob_attn()

    def prep_expression(self, metadata):
        from torch_geometric.data import Batch
        device = next(self.norm.parameters()).device
        datas = [_load_graph(self.graph_dir, m["neuron"]) for m in metadata]
        batch = Batch.from_data_list(datas).to(device)
        h = F.gelu(self.gcn1(batch.x, batch.edge_index, edge_weight=batch.edge_attr))
        h = self.gcn2(h, batch.edge_index, edge_weight=batch.edge_attr)
        h = self.norm(h)
        # un-batch: PyG concatenates nodes along dim 0; each graph has G nodes,
        # so reshape back to [B, G, hidden_dim].
        b = len(datas)
        g = datas[0].x.shape[0]
        tokens = h.view(b, g, self.hidden_dim)
        # post-transformer context vector — use raw mean over per-cell features
        ctx = batch.x.view(b, g, self.node_feat_dim).mean(dim=-1, keepdim=True)
        return tokens, ctx


def build_expression_modality(encoder, **kwargs):
    """Factory used by ztransformer.BioZorro to pick an expression encoder."""
    encoder = (encoder or "scatter").lower()
    if encoder == "scatter":
        return GraphExpressionModality(
            **{k: v for k, v in kwargs.items() if k not in ("mean_vec_dir", "graph_metric_dir")}
        )
    if encoder == "mlp":
        kwargs.pop("graph_metric_dir", None)
        return MeanVecExpressionModality(**kwargs)
    if encoder == "gnn":
        return GraphNNExpressionModality(**kwargs)
    raise ValueError(f"unknown expression_encoder: {encoder!r} (expected scatter|mlp|gnn)")
