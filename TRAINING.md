# Training CellSpliceNet

This repo ships two trainers under `src/`:

| Script | Loop | Use it for |
|--------|------|------------|
| **`train_full.py`** | step- & time-budgeted | **recommended for all new work**, required for large data (e.g. GTEx, ~2.18M rows) |
| `train.py` | epoch-based | legacy; small worm dataset / reproducing the original paper runs |

`train_full.py` is path-agnostic — every data path is a CLI flag (see `python src/train_full.py --help`), so a fresh clone runs against data staged anywhere. Nothing is hardcoded to one machine.

---

## 1. Where to put the data

The model needs, per dataset, a `data_config.ini` plus the files it points to. Default paths in `args.py` resolve under `<repo>/dataset/`, but you can point anywhere with flags.

**Files referenced (under `[processed_files]` in `data_config.ini`):**
- `enc_seq_file` — encoded primary sequences (`.pth`)
- `enc_sj_file` — encoded splice-junction windows (`.pth`)
- `spliceregion_inds` — splice-region indices (`.csv`)
- `events_coordinates` — exon/intron/gene coordinates per event (`.csv`/`.tsv`). **Optional** — only needed for non-`neuron_replicate` datasets. Can also be supplied with `--events_coordinates <path>`.

**Passed directly as flags (not via the config):**
- `--expression_data_root` — the train table (`train_data.csv` / `full_data.tsv`)
- `--structure_data_root` — structure scattering pickle (`structure_scattering_dict*.pkl`)
- `--scatter_coeffs_dir` — directory of per-cell-type expression embeddings:
  - worm: `scatter_coeffs_<neuron>.pt`
  - GTEx: `scatter_coeffs_<tissue>.pt` (or `<name>.pt`)

You don't need to move data into the repo; pass absolute paths. If you'd rather use the zero-flag defaults, stage the data under `<repo>/dataset/` matching the layout in `args.py`.

---

## 2. Quickstart — GTEx (49 tissues)

```bash
module load miniconda && conda activate mioflow   # or your env
cd <repo>

python src/train_full.py \
    --data_tag             gtex \
    --sfgenes              493 \
    --config_fname         /path/to/gtex/data_config.ini \
    --expression_data_root /path/to/gtex/train_data.csv \
    --structure_data_root  /path/to/gtex/structure_scattering_dict.pkl \
    --scatter_coeffs_dir   /path/to/gtex/scatter_coeffs \
    --batch_size 4 --n_steps 200000
```

`--sfgenes 493` must match the number of splice-factor genes in the scatter `.pt` files (GTEx). `--data_tag gtex` sets `dataset_type=01Feb2025_gtex`.

## 3. Quickstart — worm (*C. elegans*)

Worm is the original target; use `--sfgenes 243` (the paper baseline) and point at the worm config:

```bash
python src/train_full.py \
    --data_tag             replicate \
    --sfgenes              243 \
    --config_fname         /path/to/worm/data_config.ini \
    --expression_data_root /path/to/worm/train_data.csv \
    --structure_data_root  /path/to/worm/structure_scattering_dict.pkl \
    --scatter_coeffs_dir   /path/to/worm/scatter_coeffs \
    --events_coordinates   /path/to/worm/events_coordinates.tsv \
    --batch_size 16 --n_steps 200000
```

Notes on `--data_tag` (it becomes `dataset_type=01Feb2025_<tag>`, which gates dataloader behavior):
- contains `neuron_replicate` → coordinates are read inline from each row; `--events_coordinates` is **not** needed.
- contains `singlereplicant` → trains on the ΔPSI target column.
- anything else (e.g. `replicate`) → uses the `events_coordinates` table (config key or `--events_coordinates`).

The legacy epoch-based path is still available for worm: `python src/train.py --replicate_status replicate`.

---

## 4. Expression encoders / ablations

`--expression_encoder` selects how expression is embedded (default `scatter`, the paper anchor):

| value | needs | meaning |
|-------|-------|---------|
| `scatter` | `--scatter_coeffs_dir` | geometric-scattering coeffs (default) |
| `mlp` | `--mean_vec_dir` (`<cell>_mean.pt`) | MLP on per-gene mean expression |
| `gnn` | `--graph_metric_dir` (`<cell>_graph.pt`) | GCN over an MI/correlation graph |

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
