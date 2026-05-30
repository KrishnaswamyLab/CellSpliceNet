# Training CellSpliceNet

This repo ships two trainers under `src/`:

| Script | Loop | Use it for |
|--------|------|------------|
| **`train_full.py`** | step- & time-budgeted | **recommended for all new work**, required for large data (e.g. GTEx, ~2.18M rows) |
| `train.py` | epoch-based | legacy; small worm dataset / reproducing the original paper runs |

`train_full.py` reads all data paths from each dataset's `data_config.ini` (resolved in `src/utils/paths.py`). Pick a dataset with `--config_fname` or let `--data_tag` select the default config.

---

## 1. Where to put the data

Two dataset bundles live under `<repo>/dataset/`:

| Dataset | Config | `--data_tag` default |
|---------|--------|----------------------|
| Human (GTEx) | `dataset/human/data_config.ini` | `gtex` |
| Worm (*C. elegans*) | `dataset/c_elegans/data_config.ini` | anything else (e.g. `replicate`) |

**`[files]`** — all dataset paths (dataloader + model modalities):
- `enc_seq_file`, `enc_sj_file`, `spliceregion_inds`
- `train_data_file`, `valid_data_file`, `test_data_file`
- `events_coordinates` — optional; required for non-`neuron_replicate` dataset types (human/GTEx)
- `structure_data_root` — structure scattering pickle
- `scatter_coeffs_dir` — directory of per-cell-type `.pt` scatter coefficients
- `mean_vec_dir`, `graph_metric_dir` — optional; for MLP/GNN expression ablations

Paths use `$CONFIG_DIR` (directory containing the INI) or `$ROOT` (repo root). Edit the INI to change paths; there are no duplicate CLI path flags.

---

## 2. Quickstart — human (GTEx, 49 tissues)

```bash
module load miniconda && conda activate mioflow   # or your env
cd <repo>

python src/train_full.py \
    --data_tag   gtex \
    --sfgenes    493 \
    --batch_size 4 --n_steps 200000
```

`--sfgenes 493` must match the number of splice-factor genes in the scatter `.pt` files (GTEx). `--data_tag gtex` selects `dataset/human/data_config.ini` and sets `dataset_type=01Feb2025_gtex`.

## 3. Quickstart — worm (*C. elegans*)

```bash
python src/train_full.py \
    --data_tag   replicate \
    --sfgenes    243 \
    --batch_size 16 --n_steps 200000
```

Notes on `--data_tag` (it becomes `dataset_type=01Feb2025_<tag>`, which gates dataloader behavior):
- contains `neuron_replicate` → coordinates are read inline from each row; `events_coordinates` in the INI is **not** needed.
- contains `singlereplicant` → trains on the ΔPSI target column.
- anything else (e.g. `replicate`, `gtex`) → uses the `events_coordinates` table from `data_config.ini` when required.

The legacy epoch-based path is still available for worm: `python src/train.py --replicate_status replicate`.

---

## 4. Expression encoders / ablations

`--expression_encoder` selects how expression is embedded (default `scatter`, the paper anchor):

| value | needs in `[files]` | meaning |
|-------|-------------------|---------|
| `scatter` | `scatter_coeffs_dir` | geometric-scattering coeffs (default) |
| `mlp` | `mean_vec_dir` (`<cell>_mean.pt`) | MLP on per-gene mean expression |
| `gnn` | `graph_metric_dir` (`<cell>_graph.pt`) | GCN over an MI/correlation graph |

---

## 5. Run-budget & logging knobs

All are flags (visible in `--help`) and also read the matching env var as a fallback, so existing sbatch wrappers that `export` them keep working.

| flag | env | default | meaning |
|------|-----|---------|---------|
| `--n_steps` | — | 200000 | max optimizer steps |
| `--time_budget_s` | `TIME_BUDGET_S` | ~5h50m | wallclock budget; stops cleanly before SLURM walltime |
| `--eval_every_iteration` | — | 500 | validate every N steps |
| `--valid_max_batches` | `VALID_MAX_BATCHES` | 200 | cap validation iters per eval |
| `--save_step_ckpt_every` | `SAVE_STEP_CKPT_EVERY` | 10 | save a step checkpoint every Nth validation (0 disables) |
| `--log_every_iters` / `--log_every_secs` | `LOG_EVERY_ITERS` / `LOG_EVERY_SECS` | 50 / 30 | progress-log cadence |

Training stops at whichever of `--n_steps` or `--time_budget_s` comes first; the best, periodic-step, and a final checkpoint are always saved.

### SLURM

`src/train_full.sh` is a portable sbatch wrapper — edit the `#SBATCH` directives for your cluster, set the data-path env vars at the top (or pass flags), then:

```bash
sbatch src/train_full.sh
```

---

## 6. Outputs

Written to `<repo>/outputs/<model>/<run-key><dataset_type>/`:
- `model/best_validation_model.pth` — best validation loss so far
- `model/final_model.pth` — saved on exit (step cap, time budget, or end of data)
- `model/step_<N>_model.pth` — periodic (throttled by `--save_step_ckpt_every`)
- `model/loss_dict.json` — train/valid loss per validation step
- `scatter_valid/psi/` — per-validation prediction scatter plots
