"""Vanilla ViT baselines for CellSpliceNet (ablation models, additive — does NOT
touch the model-under-test code).

Two variants, both predicting PSI:
  * ``vitcsn_seq``  — ViT over ONLY the whole-RNA-sequence tokens.
  * ``vitcsn_all``  — ViT over ALL modality tokens concatenated
                      (sequence + ROI + structure + expression).

Why this is a faithful ablation
-------------------------------
CSN's encoder (``model/ztransformer.py``: ``BioZorro`` + ``ZorroTransformer``) is
already a packed-token transformer over per-modality tokens. The only things that
make it *not* a vanilla ViT are (1) the Zorro attention mask (each modality
attends only to itself; learned fusion tokens attend to everything) and (2)
per-modality attention pooling + 42 learned fusion tokens instead of one CLS.

These baselines REUSE the exact same modality tokenizers (``EntireSeqModality``,
``LocalSeqModality``, structure/expression modalities) so the inputs are
byte-identical to CSN, then differ only by: full all-to-all attention, a single
prepended CLS token, and CLS -> PSI head. That isolates exactly the contribution
of CSN's structured Zorro fusion.

The transformer block (Attention / FeedForward / Transformer) is vendored
verbatim from lucidrains' vit-pytorch (vanilla ViT):
  https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

import math
import torch
from torch import nn
from einops import repeat

from nn.sequence import EntireSeqModality
from nn.transformer_modules import OutputHook


# --------------------------------------------------------------------------- #
# Vendored from lucidrains/vit-pytorch (vit.py) — vanilla ViT transformer.
# --------------------------------------------------------------------------- #
class _FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        b, n, _ = x.shape
        q, k, v = map(
            lambda t: t.reshape(b, n, self.heads, -1).transpose(1, 2), qkv
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class _Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        _Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        _FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# --------------------------------------------------------------------------- #
# Shared base: CLS token, pos-embedding, ViT transformer, PSI head + losses.
# Loss / target transforms are copied verbatim from model.cellsplicenet so the
# model-agnostic training loop (model_perd / quick_validate) works unchanged.
# --------------------------------------------------------------------------- #
class _ViTBase(nn.Module):
    def __init__(self, hparams, max_tokens, dropout=0.0):
        super().__init__()
        self.device = hparams.device
        self.hidden_dim = hparams.hidden_dim
        self.adj_factor = 1e-2
        self.psi_lossfn = hparams.psi_lossfn

        dim = self.hidden_dim
        # +1 for the CLS token; pos-embedding is sliced to the actual token
        # count at forward time (expression token count varies by encoder).
        self.max_tokens = max_tokens + 1
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_tokens, dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = _Transformer(
            dim=dim,
            depth=hparams.layers,
            heads=hparams.nhead,
            dim_head=64,
            mlp_dim=dim * 4,
            dropout=dropout,
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.LogSigmoid(),
        )

    def build_tokens(self, data, metadata):
        """Subclasses return a [B, N, hidden_dim] token tensor."""
        raise NotImplementedError

    def forward(self, data, metadata, embed_list_return=False):
        tokens = self.build_tokens(data, metadata)  # [B, N, D]
        b, n, _ = tokens.shape

        cls = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat([cls, tokens], dim=1)  # [B, N+1, D]
        seq_len = x.shape[1]
        assert seq_len <= self.max_tokens, (
            f"token count {seq_len} exceeds pos-embedding capacity "
            f"{self.max_tokens}; raise max_tokens in the model ctor."
        )
        x = x + self.pos_embedding[:, :seq_len]
        x = self.dropout(x)

        x = self.transformer(x)
        cls_out = x[:, 0]  # [B, D]
        psi_pred = self.mlp_head(cls_out)  # [B, 1], log-sigmoid space

        preds = {"psi": psi_pred}
        return preds, None

    # --- copied verbatim from model.cellsplicenet --------------------------- #
    def loss(self, predictions, targets):
        psi_targets = self.transform_targets(targets)
        psi_preds = predictions["psi"]
        if self.psi_lossfn == "mse":
            psi_loss = nn.MSELoss(reduction="mean")(psi_preds, psi_targets)
        elif self.psi_lossfn == "l1":
            psi_loss = nn.L1Loss(reduction="mean")(psi_preds, psi_targets)
        return psi_loss

    def transform_targets(self, targets):
        psi_targets = targets["psi"].to(self.device)
        psi_targets = torch.log(psi_targets + self.adj_factor) - self.adj_factor
        return psi_targets

    def untransform_targets(self, predictions):
        psi_preds = torch.exp(predictions["psi"] + self.adj_factor) - self.adj_factor
        return psi_preds


# --------------------------------------------------------------------------- #
# Variant A: sequence-only ViT. Builds ONLY the whole-RNA-sequence tokenizer
# (no structure/expression data loaded), feeds its tokens to the ViT.
# --------------------------------------------------------------------------- #
class vitcsn_seq(_ViTBase):
    def __init__(self, hparams):
        # Replicate BioZorro's whole-sequence tokenizer init (window math only;
        # loads no data). Mirrors BioZorro.initialize_params.
        window_size = hparams.window_size
        pad_indx = hparams.dataparams.pad_indx
        vocab_size = hparams.dataparams.vocab_size
        annot_vocab_size = 4
        max_seq_len = hparams.dataparams.max_prime_seq_len
        new_seq_length = math.ceil(max_seq_len / window_size)
        eff_window_size = torch.chunk(torch.ones(max_seq_len), new_seq_length)[0].shape[0]

        # generous cap; sliced dynamically at forward
        super().__init__(hparams, max_tokens=new_seq_length + 8)

        self.sequence_model = EntireSeqModality(
            window_size, new_seq_length, eff_window_size,
            vocab_size, annot_vocab_size, hparams.hidden_dim, pad_indx,
        )

    def build_tokens(self, data, metadata):
        rna_seq_tensor, rna_annot_tensor = data
        rna_tokens, _ = self.sequence_model.prep_sequence(rna_seq_tensor, rna_annot_tensor)
        return rna_tokens


# --------------------------------------------------------------------------- #
# Variant B: all-modality-concat ViT. Composes a BioZorro to reuse the EXACT
# modality tokenizers (including expression/structure data wiring), then drops
# its Zorro transformer + fusion tokens (we use full attention + a CLS instead).
# --------------------------------------------------------------------------- #
class vitcsn_all(_ViTBase):
    def __init__(self, hparams):
        from model.ztransformer import BioZorro

        # 2048 token capacity comfortably covers seq(391)+roi(200)+struct(500)+
        # expression(varies by encoder, ~243) + CLS, for any expression encoder.
        super().__init__(hparams, max_tokens=2048)

        encoder = BioZorro(hparams)
        # Drop the parts we deliberately replace, so param counts are honest and
        # no unused 12-layer transformer is carried around.
        del encoder.ztransformer
        del encoder.fusion_tokens
        self.encoder = encoder

    def build_tokens(self, data, metadata):
        rna_seq_tensor, rna_annot_tensor = data
        enc = self.encoder

        rna_tokens, _ = enc.sequence_model.prep_sequence(rna_seq_tensor, rna_annot_tensor)
        roi_tokens, _ = enc.local_roi_model.prep_local_sequence(
            metadata, rna_seq_tensor, rna_annot_tensor, enc.local_seq_length
        )
        structure_tokens, *_ = enc.local_structure_model.prep_scattering_coeff(
            rna_seq_tensor, metadata
        )
        expression_tokens, _ = enc.expression_model.prep_expression(metadata)

        # flatten any non-(B,N,D) modality token tensors to (B, N, D)
        def _flat(t):
            return t.reshape(t.shape[0], -1, t.shape[-1])

        tokens = torch.cat(
            [_flat(rna_tokens), _flat(roi_tokens), _flat(structure_tokens), _flat(expression_tokens)],
            dim=1,
        )
        return tokens
