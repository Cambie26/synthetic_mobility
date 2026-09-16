# synthetic_mobility

# Synthetic Mobility Data: A User-Level Membership Inference Assessment

Code for my MSc thesis evaluating the **privacy–utility trade-off of deep generative models** (VAE, WGAN-GP) trained on origin–destination (OD) trips derived from the GeoLife GPS dataset (Beijing). A user-level **membership inference attack (MIA)** measures privacy leakage, and a bootstrap resampling generator is used as a positive control.

**Thesis:** *Synthetic Mobility Data: A User-Level Membership Inference Assessment*, Campbell Gray. Master in Intelligent Interactive Systems, Universitat Pompeu Fabra, July 2026. Supervisor: Manuel Portela.
The full thesis is available at [`thesis/UPF_Thesis.pdf`](thesis/UPF_Thesis.pdf).

---

## Research questions

| | Question |
|---|---|
| **RQ1** | Can generative models accurately reproduce origin–destination mobility patterns from real trip data? |
| **RQ2** | To what extent do these models leak information about individual users through membership inference attacks? |
| **RQ3** | Are users with higher activity at greater risk of privacy leakage than less active users? |

## Key findings

- **Utility (RQ1).** The generative models reproduced **temporal structure** well (hour-of-day and day-of-week JS distance 0.12–0.15). **Spatial structure** was reproduced poorly (origin and destination JS distance > 0.8). On downstream destination prediction, the synthetic data recovered only about a quarter of real-data performance (TSTR ratio 0.25–0.27, against 0.99 for the bootstrap).
- **Privacy (RQ2).** Neither generative model showed measurable membership leakage (AUC ≈ 0.50), and near-copy rates were below 1%. The bootstrap control leaked clearly (AUC 0.75), which confirms the attack works.
- **Activity level (RQ3).** Membership inference could not separate activity groups: as few as 28 users per group gave confidence intervals too wide to support a conclusion. The **near-copy rate did rise monotonically with user activity across all models**, so higher-activity users carry greater record-reproduction risk.
- **Overall.** The observed privacy is largely a by-product of **under-fitting**, not evidence of a favourable trade-off.

| | VAE | WGAN-GP | Bootstrap |
|---|---|---|---|
| OD-flow JS distance ↓ | 0.672 ± 0.027 | 0.637 ± 0.035 | 0.308 ± 0.077 |
| Hour-of-day JS distance ↓ | 0.154 ± 0.021 | 0.149 ± 0.023 | 0.058 ± 0.024 |
| TSTR ratio ↑ | 0.271 ± 0.045 | 0.251 ± 0.038 | 0.987 ± 0.042 |
| MIA AUC (95% CI) | 0.497 [0.433, 0.567] | 0.503 [0.432, 0.575] | 0.750 [0.673, 0.823] |
| Near-copy rate | 0.009 ± 0.001 | 0.010 ± 0.002 | 1.000 ± 0.000 |

Utility metrics are reported as mean ± std over 5 seeds. MIA metrics are pooled across seeds with 95% user-level cluster-bootstrap CIs. Full results are in thesis Tables 7–9.

---

## Method at a glance

**Data.**
- **Trip extraction.** GeoLife trajectories were segmented into OD trips via stay-point detection: 100 m for 20 min around a rolling-average centroid, a minimum of 5 GPS points, and a 30-minute gap backstop.
- **Spatial handling.** Endpoints were snapped to **H3 resolution-6** cell centroids. Same-cell trips and trips outside the Beijing municipal boundary (GADM 4.1) were removed.
- **Result.** **7,739 trips from 154 users**.

**Representation.** Each trip becomes a 10-dimensional continuous vector:
- 4 centroid coordinates, standardised
- log duration and log haversine distance, z-scored
- sine/cosine encodings of start hour and day of week

User ID is never a model input.

**Generators.**
| Model | Summary |
|---|---|
| Bootstrap | Resamples real training trips with replacement. It is the utility upper bound and the privacy worst case, and serves as the MIA positive control. |
| VAE | MLP encoder/decoder (10→64→32→8 latent) with KL annealing to β = 0.1 over the first 40% of 80 epochs. |
| WGAN-GP | MLP generator (64-d noise) and a layer-normalised critic, with λ_gp = 10, 3 critic steps per generator step, and 100 epochs. |

**Utility metrics (RQ1).**
- JS distance on origin, destination, OD-flow, hour-of-day and day-of-week distributions
- OD-flow coverage
- Wasserstein distance, KS statistic and spread ratio on coordinates
- correlation difference
- TSTR/TRTR ratio for destination-cell prediction (random forest, macro-F1)

**Privacy evaluation (RQ2 and RQ3).**
- **Near-copy rate.** The share of synthetic trips within ε = 0.012 of a real training trip in standardised OD space.
- **Membership inference attack.** A user-level, shadow-model attack (after Shokri et al. 2017 and Stadler et al. 2022):
  - Users with fewer than 10 trips are excluded from the MIA, leaving 85 eligible users.
  - The eligible users are split per seed into Eval Out (20%), Eval In (40%) and Shadow (40%), stratified by activity group.
  - **10 shadow generators** each train on a random half of the shadow users.
  - A random-forest attack classifier learns from **13 per-user features**: nearest-neighbour OD-distance summaries, close-match rates, OD overlap, aggregate similarity and trip count.
  - Reported metrics are balanced accuracy, AUC, maximum (Yeom) advantage and TPR at 1%, 5% and 10% FPR.
- **Activity groups.** Low (10–20 trips), Medium (21–63) and High (≥ 64), with equal user counts.

**Experimental design.** Five seeds (111, 222, 333, 444, 555) anchor every stochastic step, including splits, initialisation and sampling.

---

## Repository structure

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── 00_preprocess_geolife.ipynb         # GeoLife trajectories → OD trips (Appendix A)
├── 01_generate_and_attack.ipynb        # generators, shadow models, MIA (per seed)
├── 02_evaluate.ipynb                   # utility + distance-based privacy (per seed)
├── 03_figures_tables.ipynb             # aggregation, CIs, all thesis figures & tables
├── 04_exploration_hyperparameters.ipynb  # EDA, hyperparameter exploration, training curves (Appendix B)
├── src/
│   ├── feature_pipeline.py             # ODFeaturePipeline: 10-d encoding + inverse transform
│   ├── user_split.py                   # loading, filtering, user-level splits, activity groups
│   ├── log_experiments.py              # JSON experiment logging
│   ├── models/
│   │   ├── bootstrap.py
│   │   ├── vae.py
│   │   ├── wgan_gp.py
│   │   └── wrappers.py                 # GeneratorWrapper: sampling, save/load
│   ├── attacks/
│   │   ├── shadow.py                   # shadow splits + shadow generator training
│   │   ├── features.py                 # per-user MIA feature extraction
│   │   ├── feature_table.py
│   │   └── attack_model.py             # attack classifier training + evaluation
│   └── evaluation/
│       └── v4_metrics.py               # utility, near-copy, MIA and per-group metrics
├── outputs/
│   ├── figures/                        # thesis figures (PDF)
│   ├── tables/                         # aggregate results (CSV + LaTeX)
│   ├── eval_log_v4.json                # per-seed utility + distance-privacy metrics
│   └── mia_log_v4.json                 # per-seed MIA metrics (overall and by activity group)
└── thesis/
    └── UPF_Thesis.pdf
```

The notebooks run from the repository root and use paths relative to it (`src/`, `artifacts_v4/`, `outputs/`). Module and artefact names are kept exactly as they were when the thesis results were produced.

---

## Data

**No real or synthetic trip data is included in this repository.** See [Data and ethics](#data-and-ethics) for why.

1. **GeoLife GPS Trajectories (v1.3)** from Microsoft Research Asia is available from the [Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=52367). Please follow its licence terms and cite the papers listed under [References](#references).
2. **Preprocessing.** `00_preprocess_geolife.ipynb` converts the raw GeoLife trajectories into OD trips. The procedure is described in **Appendix A** of the thesis: stay-point segmentation (Algorithm 1), H3 res-6 discretisation, trip boundary timing, and filtering to the Beijing boundary.

   It writes `od_trips_20260410_input.csv` to the repository root, which all later notebooks read. The file has one row per trip and these columns:
   - user ID
   - start and end timestamps
   - start and end H3 cells
   - start and end centroid coordinates (`start_centroid_lat`, `start_centroid_lon`, `end_centroid_lat`, `end_centroid_lon`)
3. **Beijing boundary (figures only).** Download the GADM 4.1 China level-1 GeoJSON (`gadm41_CHN_1.json.zip`) from [gadm.org](https://gadm.org/download_country.html) and place it at the repository root.

---

## Setup

The experiments were run on **Google Colab** (Python 3.12, `h3` 4.x).

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
pip install -r requirements.txt
```

Main dependencies are `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`, `h3` (v4 API), `matplotlib`, `seaborn`, `geopandas`, `shapely` and `joblib`.

On Colab, mount Google Drive and `%cd` into the repository root before running. The notebooks contain the Drive paths used during the original runs.

---

## Reproducing the results

Run the notebooks in order from the repository root. The saved notebook outputs are the outputs from the thesis runs.

| Step | Notebook | Reads | Writes |
|---|---|---|---|
| 0 | `00_preprocess_geolife.ipynb` | Raw GeoLife trajectories + GADM boundary | `od_trips_20260410_input.csv` (7,739 trips, 154 users) |
| 1 | `01_generate_and_attack.ipynb` | OD trips CSV | For each model and seed: `artifacts_v4/splits/`, synthetic trips, attack features, attack model and per-user MIA scores. It also writes `outputs/mia_log_v4.json`. |
| 2 | `02_evaluate.ipynb` | OD trips CSV + `artifacts_v4/` | Per-seed utility and distance-privacy metrics, including by activity group. It also writes `outputs/eval_log_v4.json`. |
| 3 | `03_figures_tables.ipynb` | `artifacts_v4/` + eval log (no recomputation) | `outputs/tables/*.csv|.tex` and `outputs/figures/*.pdf` |
| — | `04_exploration_hyperparameters.ipynb` | OD trips CSV | EDA, β sweep and training curves. It is not required for the main results. |

Step 1 is the expensive one: 3 generators plus 10 shadow generators per model, across 5 seeds.

**Where the thesis figures and tables come from** (all produced by `03_figures_tables.ipynb`):

| Thesis | Output file |
|---|---|
| Fig. 1: temporal rhythm of active trips | `active_trips_rhythm` |
| Fig. 3: ECDF of trips per user | `ecdf_trips_per_user` |
| Fig. 4: JS distance family | `rq1_fidelity_js_family` |
| Fig. 5: spatial OD distribution (seed 111) | `spatial_od_2x2_seed111` |
| Fig. 6: trips by hour of day | `hourly_trips_seed111` |
| Fig. 7: TSTR ratio | `rq1_tstr_ratio` |
| Fig. 8: MIA ROC curves | `rq2_roc` |
| Fig. 9: pooled MIA AUC | `rq2_mia_auc` |
| Fig. 10: MIA AUC by activity group | `rq3_mia_auc_by_group` |
| Fig. 11: near-copy rate by activity group | `rq3_near_copy_by_group` |
| Fig. 12: privacy–utility trade-off | `tradeoff_discussion` |
| Fig. 15: spatial OD, all seeds | `spatial_od_grid_allseeds` |
| Tables 7, 8 and 9 | `rq1_utility`, `rq2_privacy`, `rq3_by_activity` |

---

## Data and ethics

This project is about privacy leakage from mobility data, so the repository deliberately excludes the following:

- **Real OD trips.** They are user-level, and GeoLife is licensed separately.
- **All synthetic datasets.** The bootstrap output is resampled real trips, and user-level leakage from the deep models cannot be ruled out in general.
- **User splits, per-user attack features, MIA scores and trained attack models.** All of these are keyed to individual users.

Only aggregate results (tables and figures) are published.

---

## Limitations

The main limitations, discussed in thesis §5.3, are:

- **Sparse data.** The dataset is small and most users contribute few trips.
- **Uncertain RQ3 estimates.** MIA estimates are low-powered, especially per activity group. TPR at 1% FPR falls below single-user resolution with about 85 evaluation users.
- **Spatially weighted attack.** The attack features are predominantly spatial.
- **No intermediate models.** The models tested sit at the two extremes of the utility range, so the trade-off curve between them is not traced.

---

## Citation

```bibtex
@mastersthesis{gray2026synthetic,
  author = {Gray, Campbell},
  title  = {Synthetic Mobility Data: A User-Level Membership Inference Assessment},
  school = {Universitat Pompeu Fabra},
  type   = {Master's thesis, Master in Intelligent Interactive Systems},
  year   = {2026},
  month  = jul
}
```

## Licence

Code is released under the MIT Licence; The GeoLife dataset is **not** covered by this licence.

## Acknowledgements

This research was partially funded by the project "Governing Urban Interoperable Data: Empowering those the data is about – GUIDE" (Grant No. PID2023-148115OB-I00) funded by Fondo Europeo de Desarrollo Regional (FEDER), Ministerio de Ciencia, Innovación y Universidades (MCIU) and Agencia Estatal de Investigación (AEI) (10.13039/501100011033).

A huge thanks to my supervisor, Manuel Portela.
