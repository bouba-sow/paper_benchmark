# Paper outline (rewrite plan, step 0, 2026-10-25)

## Definitions (used everywhere)

- **BrainSimilarity**: fitted RSA (RDM regression). The score is the held-out delta log-likelihood per pair of the response RDM
  given the model RDM, over len+freq confounds. It is divided by the split-half (Timkey) ceiling.
  - Model RDM: the whole-network embedding RDM without layer 0.
  - Response RDM:
    - Podcast, MEG-MASC, Broderick, Pereira: the whole-brain RDM.
    - N400 (Nieuwland, F2007, H2019), Fedorenko, Tuckute: the median over units.
- **BehavSimilarity**: the same fitted-RSA score, using the |Δ surprisal| RDM against the behavioral response RDM, with the
  same normalization.
- **Composite (Pareto axes)**: the median across datasets/measures.
- **Matched-scale model**: `score ~ memory * data + log10 params + log10 ppl`.
  - Fitted separately per domain.
  - LMM with a dataset-unit random intercept; for the two-dataset case, OLS with family-clustered SE.
  - Reference = Standard LLM.
- **Appendix direct track**: here the scores are called **BrainScore / BehavScore**.
  - Ridge on embeddings (Pearson r / correlation ceiling).
  - LME on surprisal (held-out dLL / dLL ceiling, layer chosen on TRAIN dLL).

## Main text

| Section | Content | Figure (source) | Table (source) |
|---|---|---|---|
| 1 Intro | brain vs behavioral similarity; trade-off; constraints; three results | (optional) figure1_overview | - |
| 2 Related work | keep, trim | - | - |
| 3 Setup | datasets, models, the two similarity scores, ceilings, matched-scale model | - | - |
| 4.1 NS × Podcast | front + matched scale | `reports/ns_podcast/pareto_dll_wholenet_no0.pdf` | `rsa_nspodcast_table` ← `reports/ns_podcast/coefficients.csv` |
| 4.2 Full benchmark | front + matched scale | `reports/rsa_summary/wholebrain_pareto/dll/pareto_dll_wholenet_no0.pdf` | `rsa_main_table` ← `reports/matched_scale_wholebrain_dll/coefficients.csv` |
| 4.3 DLT memory load | easy-easy / any-hard fronts + matched scale (with difference) | `reports/rsa_summary/dlt_pareto_ge2_wholebrain/pareto_{easy-easy,any-hard}.pdf` | `rsa_dlt_table` ← `reports/matched_scale_dlt_ge2_wholebrain{,_anyhard}/coefficients.csv` |
| 5 Discussion | reframed | - | - |

DLT conclusions (from Linguae_Presentation/main.tex):
- The hypothesis was that the constraint advantage would be largest at high memory load.
- Instead, no constraint effect on BehavSimilarity is significant on any-hard pairs; the behavioral benefit is only found on easy-easy pairs.
- The brain penalty remains on both.

## Appendix

| Section | Content | Figure (source) | Table (source) |
|---|---|---|---|
| A Same predictor on both axes | is the trade-off a matter of predictor? | | |
| A.1 embedding × embedding | front + matched scale | **GAP**: no front figure (ratios exist: `.../dll/dll_ratios_embedding_both.csv`) | **GAP**: table from `reports/matched_scale_wholebrain_embedding_both/coefficients.csv` |
| A.2 surprisal × surprisal | front + matched scale | `reports/rsa_summary/wholebrain_pareto/dll/pareto_dll_surprisal.pdf` | **GAP**: table from `reports/matched_scale_wholebrain_surprisal/coefficients.csv` |
| B Direct prediction | BrainScore × BehavScore | | |
| B.1 ridge embedding | best + last layer fronts + matched scale | `reports/paper/embedding_{best,last}layer_pareto.pdf` | **GAP**: table from `outputs_jeanzay/matched_scale_report/regression_coefficients_embedding{,_bestlayer}*.csv` (check the multilayer file names) |
| B.2 LME surprisal | best + last layer fronts + matched scale | `reports/paper/surprisal_{best,last}layer_pareto.pdf` | **GAP**: same, `regression_coefficients_surprisal{,_bestlayer}.csv` |
| C Datasets, models | | - | `dataset_table`, `model_table` |
| D Noise ceilings | RSA dLL, ridge correlation, LME dLL | - | `rdm_ceiling_table`, `corr_ceiling_table`, `dll_ceiling_table` (the last two need refreshing via `build_supporting_tables.py`, which also writes other tables, so check its targets first) |
| E Implementation | RDM regression, ridge config, LME formulas | - | `ridge_config_table`, `lme_formulas` |

## Removed

- Scaling plots, untrained Pythia, DLT hard-hard (user, 2026-10-25). DLT keeps easy-easy (reference) and any-hard.

- Robustness section: `robustness_table`, `robust_forest_*`, `robust_per_unit_*`. This is the enlarged matched-scale analysis.
- Design-based comparisons: `design_table`, `design_dose_*`, `design_mechanism_*`.
- Unfitted RSA: `rsa_nofit_table`, `rsa_nofit_wholebrain_pareto`.
- From `appendix_direct.tex`: composite / unadjusted medians (`results_medians_table`), the old `results_main_table`, size and perplexity.
- Unused stale tables/figures: `score_ranges_table`, `rsa_summary_table`, `rsa_ceiling_table`, `rdm_dll_ceiling_table`, `figure2-4_*`,
  `figure_dll_pareto_*`, `figure_rsa_pareto_*`, `rdm_pareto_*_evaluated`, `rsa_pareto_*_current`, `appendix_pareto_*`.
  These move to `paperwriting/_unused/` rather than being deleted.
- `build_paper.sh` steps 03_ms_wholebrain_rsa and 17_ms_robust.

## Build changes (DONE 2026-10-25: build_paper.sh steps 18_copy_figures + 19_rsa_tables; paper names = rsa_{ns_podcast,wholebrain,embboth,surpboth}_pareto, rsa_dlt_pareto_{easy-easy,any-hard}, direct_{embedding,surprisal}_{last,best}layer_pareto; tables rsa_{nspodcast,main,dlt,embboth,surpboth}_table, direct_{embedding,surprisal}_table)

1. Add a final `build_paper.sh` step that copies each figure above into `paperwriting/figures/` under its paper name. Nothing
   does this today: the paper's figures date from 10-20 and predate every rebuild since.
2. Extend `paperwriting/build_rsa_tables.py` with the appendix tables (A.1, A.2, B.1, B.2), all in the same
   two-by-two format. Drop the nofit table.
3. Add a front figure for embedding × embedding (`rsa_pareto_wholebrain.py` can draw it from `dll_ratios_embedding_both.csv`).

## Execution order

0 outline (this) → build changes 1–3 → 4.1 → 4.2 → 4.3 → setup/intro/abstract/discussion → A → B → C–E → removals → compile.
