"""CellSpliceNet SeqOnly baseline (ablation model, additive -- does NOT touch the
model-under-test code).

``csn_seqonly`` is the full CellSpliceNet with the RNA-structure and
cell-type-expression modalities removed, keeping the whole-sequence and
region-of-interest (local sequence) modalities. This is the baseline Reviewer #2
asked to be presented as the primary one: it isolates the contribution of the two
non-sequence modalities while holding everything else fixed.

Why the zeroing happens at TOKEN level, before the Zorro transformer
--------------------------------------------------------------------
In ``ztransformer.ZorroTransformer`` each modality attends only to itself; only the
42 learned fusion tokens attend across modalities. Zeroing the *pooled* structure
and expression representations after the transformer would therefore NOT remove
their information -- it would still reach the readout via the fusion tokens. So the
tokens are zeroed at source, the same convention used by
``src-ablation/model/ztransformer_abilation.py``.

Why wrapper modules rather than patched functions
-------------------------------------------------
Checkpoints are saved with ``torch.save(model)``, i.e. the whole module is pickled.
Monkeypatching the prep methods with closures makes the model unpicklable and the
run dies at its first checkpoint save. The two modality modules are therefore
wrapped in top-level ``nn.Module`` subclasses, which pickle normally and keep the
inner modules registered so the parameter count stays identical to the full model.

Everything else -- tokenizers, Zorro mask, fusion tokens, attention pooling,
psi_comb_nn (hidden_dim x 5), psi_pred_nn, loss and target transforms -- is
unchanged, so the comparison isolates exactly the two removed modalities.

Env
---
SEQONLY_SYNTH_SHAPES=1  (default in worm_seqonly_train.sh)
    Do not call the real structure/expression loaders at all; synthesise
    correctly-shaped zero tensors. Needed for the grouped 12Jan2026 splits, whose
    events are not in our structure pickle and whose native expression is
    243-factor. Check coverage with preprocessing/ablations/preflight_seqonly.py.
SEQONLY_BASE=cellsplicenet|deltacellsplicenet
    Which full model to ablate. Default deltacellsplicenet, the anchor used by
    preprocessing/ablations/worm_train.py, so the comparison is matched.
"""

import os
import torch
from torch import nn

from model.model import cellsplicenet, deltacellsplicenet


class _ZeroStructureWrapper(nn.Module):
    """Returns zeroed structure tokens. Delegates everything else to the inner module."""

    def __init__(self, inner, hidden_dim, structure_length, synth):
        super().__init__()
        self.inner = inner
        self.hidden_dim = int(hidden_dim)
        self.structure_length = int(structure_length)
        self.synth = bool(synth)

    def prep_scattering_coeff(self, sequence_tensor, metadata, *args, **kwargs):
        if self.synth:
            b = sequence_tensor.shape[0]
            z = torch.zeros(b, self.structure_length, self.hidden_dim,
                            device=sequence_tensor.device, dtype=torch.float32)
            # matches (structure_tokens, structure_annotation, structure_seq,
            #          secondary_structure, structure_seq_letters)
            return z, None, None, None, None
        out = self.inner.prep_scattering_coeff(sequence_tensor, metadata, *args, **kwargs)
        return (out[0] * 0,) + tuple(out[1:])

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["inner"], name)


class _ZeroExpressionWrapper(nn.Module):
    """Returns zeroed expression tokens and a zeroed expression tensor.

    The tensor is zeroed too because ``deltacellsplicenet`` can add an expression
    residual to the PSI head; leaving it non-zero would leak the modality back in.
    """

    def __init__(self, inner, hidden_dim, num_splice_factors, synth):
        super().__init__()
        self.inner = inner
        self.hidden_dim = int(hidden_dim)
        self.num_splice_factors = int(num_splice_factors)
        self.synth = bool(synth)

    def _batch_size(self, metadata):
        """Batch size from metadata.

        The dataloader hands the modality preppers a SEQUENCE of per-sample dicts
        (the real prep_expression does ``for metadata_i in metadata``), so len() of
        the sequence is the batch size. The dict-of-columns form is handled too in
        case a future collate changes.
        """
        if isinstance(metadata, dict):
            for v in metadata.values():
                try:
                    return len(v)
                except TypeError:
                    continue
            return 1
        try:
            return len(metadata)
        except TypeError:
            return 1

    def prep_expression(self, metadata, *args, **kwargs):
        if self.synth:
            b = self._batch_size(metadata)
            dev = next((p.device for p in self.inner.parameters()), torch.device("cpu"))
            tok = torch.zeros(b, self.num_splice_factors, self.hidden_dim,
                              device=dev, dtype=torch.float32)
            ten = torch.zeros(b, self.num_splice_factors, 1, device=dev, dtype=torch.float32)
            return tok, ten
        out = self.inner.prep_expression(metadata, *args, **kwargs)
        return out[0] * 0, out[1] * 0

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["inner"], name)


class _NoAnnotSeqWrapper(nn.Module):
    """Removes exon/intron/flank LABELS from the sequence annotation channel.

    IMPORTANT: it collapses every non-pad annotation class to a single constant
    rather than zeroing the channel. ``EntireSeqModality.annotation_prep`` derives
    the padding mask from the annotation itself
    (``glob_attn_mask = annot_ids.float().mean(-1) != 0``), so zeroing would mark
    every position as padding and break the model instead of merely removing the
    labels. Collapsing to class 1 leaves the mask bit-identical while destroying all
    exon/intron information.
    """

    def __init__(self, inner, keep_class=1):
        super().__init__()
        self.inner = inner
        self.keep_class = int(keep_class)

    def _collapse(self, annot):
        if annot is None:
            return annot
        return torch.where(annot > 0, torch.full_like(annot, self.keep_class), annot)

    def prep_sequence(self, sequence_tensor, annot_type_tensor, *a, **k):
        return self.inner.prep_sequence(sequence_tensor, self._collapse(annot_type_tensor), *a, **k)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["inner"], name)


class _NoAnnotRoiWrapper(nn.Module):
    """Same collapse for the region-of-interest (local sequence) annotation channel."""

    def __init__(self, inner, keep_class=1):
        super().__init__()
        self.inner = inner
        self.keep_class = int(keep_class)

    def _collapse(self, annot):
        if annot is None:
            return annot
        return torch.where(annot > 0, torch.full_like(annot, self.keep_class), annot)

    def prep_local_sequence(self, metadata, rna_seq_tensor, rna_annot_tensor, *a, **k):
        return self.inner.prep_local_sequence(metadata, rna_seq_tensor, self._collapse(rna_annot_tensor), *a, **k)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["inner"], name)


class _ZeroRoiWrapper(nn.Module):
    """Zero the region-of-interest tokens, pre-transformer.

    Mirrors src-ablation's only_ablation == 'sequence' (roi_tokens * 0), i.e. the
    whole-sequence channel is the only sequence input left.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def prep_local_sequence(self, *args, **kwargs):
        roi_tokens, roi_sequence = self.inner.prep_local_sequence(*args, **kwargs)
        return roi_tokens * 0, roi_sequence

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["inner"], name)


def install_annotation_removal(model, keep_class=1):
    """Strip exon/intron annotation labels from an already-constructed model.

    Applies to BOTH the whole-sequence and ROI channels. Usable on the full model as
    well as on csn_seqonly, so an ablation and its reference can be matched.
    """
    enc = model.psi_transformer_encoder
    enc.sequence_model = _NoAnnotSeqWrapper(enc.sequence_model, keep_class)
    enc.local_roi_model = _NoAnnotRoiWrapper(enc.local_roi_model, keep_class)
    model.annotation_removed = True
    return model


def _make_seqonly(base_cls):
    class _SeqOnly(base_cls):
        def __init__(self, hparams):
            super().__init__(hparams)
            synth = os.environ.get("SEQONLY_SYNTH_SHAPES", "0") == "1"
            enc = self.psi_transformer_encoder
            enc.local_structure_model = _ZeroStructureWrapper(
                enc.local_structure_model, hparams.hidden_dim,
                hparams.structure_length, synth)
            enc.expression_model = _ZeroExpressionWrapper(
                enc.expression_model, hparams.hidden_dim,
                hparams.sfgenes + 1, synth)
            self.seqonly_synth = synth
            self.seqonly_base = base_cls.__name__
            # Arman: all ablations also removed the exon/intron sequence annotations.
            if os.environ.get("SEQONLY_DROP_ANNOT", "0") == "1":
                install_annotation_removal(self)
            # src-ablation's 'sequence' ablation also zeroed the ROI channel.
            if os.environ.get("SEQONLY_DROP_ROI", "0") == "1":
                enc.local_roi_model = _ZeroRoiWrapper(enc.local_roi_model)
                self.roi_removed = True

    _SeqOnly.__name__ = f"_SeqOnly_{base_cls.__name__}"
    return _SeqOnly


_BASES = {"cellsplicenet": cellsplicenet, "deltacellsplicenet": deltacellsplicenet}
_Base = _BASES[os.environ.get("SEQONLY_BASE", "deltacellsplicenet")]


class csn_seqonly(_make_seqonly(_Base)):
    """CellSpliceNet minus the structure and expression modalities."""
    pass


class csn_full_nocues(deltacellsplicenet):
    """Full CellSpliceNet (all four modalities) with the exon-location cues removed:
    annotations collapsed and ROI tokens zeroed. The matched reference for
    csn_seqonly with SEQONLY_DROP_ANNOT=1 SEQONLY_DROP_ROI=1."""

    def __init__(self, hparams):
        super().__init__(hparams)
        install_annotation_removal(self)
        enc = self.psi_transformer_encoder
        enc.local_roi_model = _ZeroRoiWrapper(enc.local_roi_model)
        self.roi_removed = True


class _NoAnnotStructureWrapper(nn.Module):
    """Collapse the exon/intron annotation columns fed to the structure modality.

    nn/structure.py attaches its own 6-class annotation one-hot to every structure
    token inside prep_scattering_coeff (self.annotation_prep), so a pass-through
    wrapper cannot reach it; this re-runs that method body with the annotation
    collapsed (non-pad -> keep_class; pad stays 0 so the attention mask is intact).
    """

    def __init__(self, inner, keep_class=1, collapse=True):
        super().__init__()
        self.inner = inner
        self.keep_class = int(keep_class)
        self.collapse = bool(collapse)  # False reproduces the inner method exactly (for testing)

    def prep_scattering_coeff(self, sequence_tensor, metadata):
        from einops import repeat
        from nn.structure import load_structure
        m = self.inner
        (local_structure_tensor, structure_annotation, structure_seq,
         secondary_structure, structure_seq_letters) = load_structure(
            sequence_tensor, metadata, m.structure_data_root, m.structure_length,
            getattr(m, 'coeff_scaling', None))
        collapsed = structure_annotation
        if self.collapse:
            collapsed = torch.where(structure_annotation > 0,
                                    torch.full_like(structure_annotation, self.keep_class),
                                    structure_annotation)
        annot_ids, _ = m.annotation_prep(collapsed)
        comb = torch.cat([local_structure_tensor.unsqueeze(2), annot_ids], dim=-1)
        comb = torch.flatten(comb, 2, 3)
        embed = m.to_scattering_embedding(comb)
        b, n = embed.shape[0], embed.shape[1]
        pos = repeat(torch.arange(n, device=embed.device), "n -> b n", b=b)
        embed = embed + m.structure_pos_embedding(pos)
        return embed, structure_annotation, structure_seq, secondary_structure, structure_seq_letters

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["inner"], name)


class csn_full_nocues_strict(csn_full_nocues):
    """csn_full_nocues plus the structure modality's own annotation columns collapsed,
    so no input marks the exon position."""

    def __init__(self, hparams):
        super().__init__(hparams)
        enc = self.psi_transformer_encoder
        enc.local_structure_model = _NoAnnotStructureWrapper(enc.local_structure_model)
        self.structure_annotation_removed = True


class csn_full_nocues_nostruct(csn_full_nocues):
    """csn_full_nocues with the structure modality zeroed (expression kept)."""

    def __init__(self, hparams):
        super().__init__(hparams)
        enc = self.psi_transformer_encoder
        enc.local_structure_model = _ZeroStructureWrapper(
            enc.local_structure_model, hparams.hidden_dim, hparams.structure_length, False)
        self.structure_removed = True


class csn_nostruct(deltacellsplicenet):
    """Full CellSpliceNet minus the structure modality (sequence, ROI, annotations and
    expression kept). Single-modality ablation for the human experiments."""

    def __init__(self, hparams):
        super().__init__(hparams)
        enc = self.psi_transformer_encoder
        enc.local_structure_model = _ZeroStructureWrapper(
            enc.local_structure_model, hparams.hidden_dim, hparams.structure_length, False)
        self.structure_removed = True


class csn_noexpr(deltacellsplicenet):
    """Full CellSpliceNet minus the expression modality (sequence, ROI, annotations and
    structure kept). Without expression the model has no tissue/neuron information."""

    def __init__(self, hparams):
        super().__init__(hparams)
        enc = self.psi_transformer_encoder
        enc.expression_model = _ZeroExpressionWrapper(
            enc.expression_model, hparams.hidden_dim, hparams.sfgenes + 1, False)
        self.expression_removed = True


class csn_full_nocues_noexpr(csn_full_nocues_strict):
    """csn_full_nocues_strict with the expression modality zeroed (structure kept, its
    annotation columns collapsed). No-cues counterpart of csn_noexpr."""

    def __init__(self, hparams):
        super().__init__(hparams)
        enc = self.psi_transformer_encoder
        enc.expression_model = _ZeroExpressionWrapper(
            enc.expression_model, hparams.hidden_dim, hparams.sfgenes + 1, False)
        self.expression_removed = True


class csn_seqonly_nocues(csn_seqonly):
    """SeqOnly with annotations collapsed and ROI tokens zeroed, independent of the
    SEQONLY_DROP_* env vars: the SeqOnly definition used in the rebuttal (Minor 3),
    matched to csn_full_nocues_strict."""

    def __init__(self, hparams):
        super().__init__(hparams)
        if not getattr(self, "annotation_removed", False):
            install_annotation_removal(self)
        if not getattr(self, "roi_removed", False):
            enc = self.psi_transformer_encoder
            enc.local_roi_model = _ZeroRoiWrapper(enc.local_roi_model)
            self.roi_removed = True
