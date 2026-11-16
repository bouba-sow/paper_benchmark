#!/usr/bin/env python3
"""Matched-scale tables of the RSA-regression paper (sections/rsa_*_table.tex), generated from the report coefficient CSVs.

  summary_table.tex        every main-text fit in one table (all datasets, DLT bins, frequency bins, language vs MD);
  summary_z_table.tex      all-datasets rows with robust-z outcomes (MD: reports/matched_scale_md/coefficients{_rank,}.csv;
                           all datasets z: coefficients_lineage_z.csv from scripts/analysis/matched_scale_lineage_z.py)
  rsa_two_table.tex        two datasets, ceiling + robust z (appendix)
  freq_measure_table.tex   frequency bins per reading measure (<FREQ_DIR>/per_measure.csv, scripts/analysis/freq_bins_per_measure.py)
  md_dataset_table.tex     language vs MD per dataset, robust z (reports/matched_scale_md/coefficients.csv, A_<dataset>)
  rsa_main_table.tex       two datasets: reports/ns_podcast/coefficients.csv            (OLS, family-clustered SE)
                           all datasets: reports/matched_scale_wholebrain_dll/coefficients_lineage.csv
                                         (LMM, dataset + lineage intercepts, lme4 / Satterthwaite)
  rsa_dlt_table.tex        reports/<DLT_DIR>/coefficients.csv (OLS x bin: easy-easy, any-hard; both bins / all-pairs reference)
  rsa_freq_table.tex       reports/<FREQ_DIR>/coefficients.csv (frequent / any-rare, Zipf <=3 vs >=4; GECO, OneStop, NS, ZuCo + Podcast)
  rsa_sameq_table.tex      reports/matched_scale_wholebrain_{embedding_both,surprisal}/coefficients_lineage.csv (same quantity on both axes)
  rsa_ckpt_table.tex       paper_results/scratch/ckpt_rsa/matched_scale/coefficients.csv (scripts/analysis/matched_scale_checkpoints.R:
                           original models vs + Pythia / BabyLM-GPT2 training checkpoints, spec S1, ceiling-normalized)
  rsa_ckpt_two_table.tex   paper_results/scratch/ckpt_rsa/ns_podcast/coefficients.csv (scripts/analysis/ns_podcast_checkpoints.py:
                           the same on Natural Stories + Podcast alone)
  direct_embedding_table.tex  outputs/benchmark_summary_multilayer/_matched_scale/regression_coefficients_embedding{,_bestlayer}_multilayer.csv
  direct_surprisal_table.tex  outputs_jeanzay/matched_scale_report/regression_coefficients_surprisal{,_bestlayer}.csv
                           (separate behavioral / neural LMMs, ceiling-normalized, last and best layer)

    .venv/bin/python paperwriting/build_rsa_tables.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# 2026-11-12: both bin analyses divide both bins by the all-pairs reference; frequency restricted to datasets with >= 100 rare words
# (scripts/paper/refit_bins_allpairs.sh). Old: ..._ns_wholebrain_anyhard_binceil, matched_scale_dlt_freq_vh_wholebrain_anyhard_genceil.
DLT_DIR = "matched_scale_dlt_v2_timkey_ns_wholebrain_anyhard_genceil"
FREQ_DIR = "matched_scale_dlt_freq_vh_big4_wholebrain_anyhard_genceil"

ROOT = Path(__file__).resolve().parents[1]
SEC = ROOT / "paperwriting/sections"
REP = ROOT / "reports"
TERMS = [("memory", r"Memory constraint $\beta_M$"), ("data", r"Data constraint $\beta_D$"),
         ("memory:data", r"Interaction $\beta_{MD}$"),
         ("log_params", r"$\log_{10}$ parameters $\beta_S$"), ("log_ppl", r"$\log_{10}$ perplexity $\beta_P$")]


def stars(p: float) -> str:
    return "^{***}" if p < .001 else "^{**}" if p < .01 else "^{*}" if p < .05 else ""


def cell(r) -> str:
    return f"${r.coef:+.3f}{stars(r.p)}$ ({r.se:.3f})"


def get(c: pd.DataFrame, **kw):
    for k, v in kw.items():
        c = c[c[k] == v]
    assert len(c) == 1, kw
    return c.iloc[0]


def short_cell(r) -> str:
    return f"${r.coef:+.3f}{stars(r.p)}$".replace("0.", ".", 1) + f" ({r.se:.3f})".replace("0.", ".", 1)


def main_table(ns: pd.DataFrame, m: pd.DataFrame) -> str:
    """Rows = terms; columns = (two datasets, all datasets) x (behavioral, neural)."""
    lines = []
    for t, name in TERMS:
        cells = [short_cell(get(c, model=mod_name, modality=mod, term=t))
                 for c, mod_name in ((ns, "3_ceiling"), (m, "3_lineage")) for mod in ("behavioral", "neural")]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    return rf"""\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{5pt}}
\caption{{Matched-scale regression of behavioral and brain similarity on the constraint factors, parameter count and
perplexity (Equation~\ref{{eq:group}}). Two datasets: Natural Stories (118 models) and Podcast (116 models), OLS with
family-clustered standard errors. All datasets: 16 behavioral measures and eight neural datasets, random intercepts for the
dataset unit and the model lineage (REML, Satterthwaite $t$-tests). Coefficient (SE); reference: Standard LLM. $^{{*}}p<.05$, $^{{**}}p<.01$, $^{{***}}p<.001$.}}
\label{{tab:rsa-main}}
\begin{{tabular}}{{lcccc}}
\toprule
 & \multicolumn{{2}}{{c}}{{Two datasets}} & \multicolumn{{2}}{{c}}{{All datasets}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}
 & Behavior & Brain & Behavior & Brain \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def same_quantity_table(e: pd.DataFrame, u: pd.DataFrame) -> str:
    """Only the axis whose model RDM changes: embedding RDM -> behavior (x 10^3), surprisal RDM -> brain."""
    lines = []
    for t, name in TERMS:
        a, b = get(e, model="3_lineage", modality="behavioral", term=t), get(u, model="3_lineage", modality="neural", term=t)
        lines.append(f"{name} & ${1e3 * a.coef:+.2f}{stars(a.p)}$ ({1e3 * a.se:.2f}) & {short_cell(b)} \\\\")
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{Matched-scale regression with the same model quantity on both axes. Left: behavioral similarity from the
whole-network embedding RDM (coefficients $\times 10^3$). Right: brain similarity from the $|\Delta$surprisal$|$ RDM. The other
axis is unchanged (Table~\ref{{tab:summary}}, all datasets). Random intercepts for the dataset unit and the model lineage
(REML, Satterthwaite $t$-tests). Coefficient (SE); reference: Standard LLM. $^{{*}}p<.05$,
$^{{**}}p<.01$, $^{{***}}p<.001$.}}
\label{{tab:rsa-sameq}}
\begin{{tabular}}{{lcc}}
\toprule
 & Behavior, embedding RDM & Brain, surprisal RDM \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


CKPT_COLS = (("behavioral", "A"), ("behavioral", "B"), ("neural", "A"), ("neural", "B"))


CKPT_CAPTION = {
    "pooled": (r"""Matched-scale regression with the training checkpoints of Pythia (ten sizes, steps 128--128{,}000) and of the BabyLM
GPT-2 baseline (29 checkpoints, 1M--1B words) added to the original models. Behavior: surprisal RDM, 16 reading measures; brain:
whole-network embedding RDM, the eight neural datasets. Perplexity is computed the same way for checkpoints and original models
(Appendix~\ref{app:checkpoints}); the \emph{Original} columns equal the all-datasets rows of Table~\ref{tab:summary}. Random intercepts for the dataset unit, the model lineage and the
model within its lineage (a training run's checkpoints share one model intercept); REML, Satterthwaite $t$-tests.""", "tab:rsa-ckpt"),
    "two": (r"""Two-dataset matched-scale regression (Natural Stories self-paced reading, Podcast ECoG language network) with the same
training checkpoints added to the original models. One row per model or checkpoint. Random intercepts for the model lineage and, with the
checkpoints, for the model within its lineage; REML, Satterthwaite $t$-tests. The \emph{Original} columns therefore differ from the
fits of Table~\ref{tab:rsa-two}, which are OLS with family-clustered standard errors.""", "tab:rsa-ckpt-two"),
}


def ckpt_table(c: pd.DataFrame, kind: str = "pooled") -> str:
    """Matched scale with and without the training checkpoints (spec S1: lineage + lineage:model intercepts, + dataset unit
    when pooled). Columns = (behavior, brain) x (original models, + checkpoints); all model sizes, ceiling-normalized outcome."""
    c = c[c.spec == "S1"]
    if "outcome" in c:
        c = c[c.outcome == "score_norm"]
    if "max_params" in c:
        c = c[~(c.max_params < float("inf"))]
    caption, label = CKPT_CAPTION[kind]
    lines = []
    for t, name in TERMS:
        cells = [short_cell(get(c, modality=mod, sample=smp, term=t)) for mod, smp in CKPT_COLS]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    n = [get(c, modality=mod, sample=smp, term="memory") for mod, smp in CKPT_COLS]
    lines.append(r"\midrule")
    # n_models counts training runs (distinct model_id = the model intercept): a run's checkpoints share the model_id of its
    # released model, so only the BabyLM GPT-2 baseline adds a run; the checkpoints themselves are counted separately
    n_ck = [int(round(r.n_ckpt_rows / (r.n_units if "n_units" in r and pd.notna(r.n_units) else 1))) for r in n]
    lines.append("Checkpoints added & " + " & ".join(f"{k}" for k in n_ck) + r" \\")
    lines.append("Rows (of which checkpoints) & " + " & ".join(f"{int(r.n_rows):,} ({int(r.n_ckpt_rows):,})" for r in n) + r" \\")
    return rf"""\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{5pt}}
\caption{{{caption}
Coefficient (SE); reference: Standard LLM. $^{{*}}p<.05$, $^{{**}}p<.01$, $^{{***}}p<.001$.}}
\label{{{label}}}
\begin{{tabular}}{{lcccc}}
\toprule
 & \multicolumn{{2}}{{c}}{{Behavior}} & \multicolumn{{2}}{{c}}{{Brain}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}
 & Original & + checkpoints & Original & + checkpoints \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def direct_table(files: dict[str, Path], caption: str, label: str, notes: str) -> str:
    """Rows = terms; columns = (behavioral, neural) x (last, best layer); the separate per-domain LMMs."""
    c = {k: pd.read_csv(f) for k, f in files.items()}
    lines = []
    for t, name in TERMS:
        cells = []
        for mod in ("behavioral", "neural"):
            for lay in ("last", "best"):
                d = c[lay]
                d = d[d.analysis.eq(f"factorial_lmm_separate_{mod}")]
                r = d[d.term.eq(t)].iloc[0]
                cells.append(f"${r.coef:+.3f}{stars(r.p_value)}$ ({r.std_err:.3f})")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    return rf"""\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{4pt}}
\caption{{{caption}}}
\label{{{label}}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lcccc}}
\toprule
 & \multicolumn{{2}}{{c}}{{BehavScore}} & \multicolumn{{2}}{{c}}{{BrainScore}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}
 & Last layer & Best layer & Last layer & Best layer \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}}}
\par\smallskip{{\footnotesize {notes} $^{{*}}p<.05$, $^{{**}}p<.01$, $^{{***}}p<.001$.}}
\end{{table}}
"""


def bin_table(c: pd.DataFrame, heads, caption: str, label: str, cols) -> str:
    """Matched-scale regression on word-pair bins (easy-easy / any-hard rows of matched_scale_dlt.py, ceiling outcome)."""
    lines = []
    for mod, head in heads:
        lines.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\itshape {head}}} \\")
        for t, name in TERMS:
            cells = [cell(get(c, model="3_lineage", modality=mod, term=t, bin=b)) for b in ("easy-easy", "any-hard", "difference")]
            lines.append(rf"\quad {name} & " + " & ".join(cells) + r" \\")
        if mod == "behavioral":
            lines.append(r"\addlinespace")
    return rf"""\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{4pt}}
\caption{{{caption}}}
\label{{{label}}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lccc}}
\toprule
 & {cols[0]} & {cols[1]} & {cols[1]} $-$ {cols[0].lower()} \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}}}
\par\smallskip{{\footnotesize Coefficient (SE). $^{{*}}p<.05$, $^{{**}}p<.01$, $^{{***}}p<.001$.}}
\end{{table}}
"""


def dlt_table(c: pd.DataFrame) -> str:
    return bin_table(
        c, (("behavioral", "Behavioral similarity (Natural Stories)"), ("neural", "Brain similarity (Podcast)")),
        r"""Matched-scale regression on syntactically easy and hard word pairs. Easy--easy: both words have DLT cost 0;
any-hard: at least one word has cost $\geq 2$. Outcome: similarity score divided by the all-pairs split-half ceiling in both bins.
The last column is the bin interaction (any-hard $-$ easy--easy). One fit over both bins per modality, with lineage and
lineage-by-bin random intercepts (REML, Satterthwaite $t$-tests).""", "tab:rsa-dlt", ("Easy--easy", "Any-hard"))


def freq_table(c: pd.DataFrame) -> str:
    return bin_table(
        c, (("behavioral", "Behavioral similarity (9 reading measures)"), ("neural", "Brain similarity (Podcast)")),
        r"""Matched-scale regression on frequent and rare word pairs. Frequent--frequent: both words of Zipf frequency $\geq 4$;
any-rare: at least one word of Zipf frequency $\leq 3$ \citep{vanheuven2014subtlex}. Outcome: similarity score divided by the
measure's all-pairs split-half ceiling in both bins. The last column is the bin interaction (any-rare $-$
frequent--frequent). One fit over both bins per modality, with lineage and lineage-by-bin random intercepts, plus a
measure-by-bin random intercept for the 9 reading measures (REML, Satterthwaite $t$-tests).""", "tab:rsa-freq", ("Frequent--frequent", "Any-rare"))


SUMMARY_TERMS = TERMS + [("memory+data+memory:data", "")]


def md_lineage() -> pd.DataFrame:
    """MD rank fits with the lineage random intercept (scripts/analysis/lineage_refits.R md); the stricter lineage + model
    specification gives the same estimates (model variance estimated at zero)."""
    md = pd.read_csv(REP / "matched_scale_md/coefficients_rank_lineage.csv")
    return md[md.re == "lineage"]


RARE_DIR = "rare_word_surprisal"
RARE_ROWS = (("gap", r"Rare $-$ frequent surprisal"), ("s_rare", "Surprisal, rare words"), ("s_freq", "Surprisal, frequent words"),
             ("freq_sens", r"Frequency sensitivity ($-$slope on Zipf)"), ("lex_r2", r"$R^2$ of frequency and length"))


def rare_surprisal_table() -> str:
    """Appendix: do the constraints change the surprisal of rare words? (scripts/analysis/rare_word_surprisal.py + lineage_refits.R rare)"""
    c = pd.read_csv(REP / RARE_DIR / "coefficients_lineage.csv")
    L = []
    for re, head in (("lineage", "Lineage random intercept"), ("lineage+model", "Lineage and model random intercepts")):
        L.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{head}}}}} \\")
        for y, lab in RARE_ROWS:
            L.append(rf"\quad {lab} & " + " & ".join(tiny_cell(r.coef, r.se, r.p) for t, _ in SUMMARY_TERMS
                                                      for r in [get(c, fit=y, re=re, term=t)]) + r" \\")
        L.append(r"\addlinespace")
    L.append(r"\multicolumn{7}{@{}l}{\textit{Any-rare behavioral score, lineage random intercept}} \\")
    for y, lab in (("score_rare", "Without surprisal covariates"), ("score_rare|mediators", "With rare $-$ frequent gap and $R^2$")):
        L.append(rf"\quad {lab} & " + " & ".join(tiny_cell(r.coef, r.se, r.p) for t, _ in SUMMARY_TERMS
                                                  for r in [get(c, fit=y, re="lineage", term=t)]) + r" \\")
    g = get(c, fit="score_rare|mediators", re="lineage", term="gap_z")
    x = get(c, fit="score_rare|mediators", re="lineage", term="lex_r2_z")
    assert g.p < .001, g.p  # the caption says p < .001
    return rf"""\begin{{table}}[h!]
\centering
\caption{{Do the constraints change how models treat rare words? Surprisal (nats, last layer) of the words of each of the 9 reading
measures, summarized per model and measure and regressed on the model classes at matched size and perplexity (Equation~\ref{{eq:group}}
with a measure random intercept). Rare: Zipf $\leq 3$; frequent: Zipf $\geq 4$. Frequency sensitivity and $R^2$: OLS of surprisal on
Zipf frequency and length over all words of the measure. Bottom: the any-rare behavioral score (as in Table~\ref{{tab:summary}}, fitted
on that bin alone) without and with the rare $-$ frequent gap and the $R^2$ as covariates ($z$ within measure; gap
${g.coef:+.3f}$, $p<.001$; $R^2$ ${x.coef:+.3f}$, $p={x.p:.2f}$). Coefficient (SE); reference: Standard LLM. $^{{*}}p<.05$,
$^{{**}}p<.01$, $^{{***}}p<.001$.}}
\label{{tab:rare-surprisal}}
\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\toprule
 & Memory $\beta_M$ & Data $\beta_D$ & $M\times D$ $\beta_{{MD}}$ & $\log_{{10}}$ params $\beta_S$ & $\log_{{10}}$ PPL $\beta_P$ & C-BabyLM vs.\ LLM \\
\midrule
{chr(10).join(L)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


MD_DIFF = {"memory": "memory:md", "data": "data:md", "memory:data": "memory:data:md", "log_params": "log_params:md",
           "log_ppl": "log_ppl:md", "memory+data+memory:data": "(memory+data+memory:data):md"}


def tiny_cell(coef: float, se: float, p: float) -> str:
    """'+.028** (.010)' with leading zeros dropped (values of the summary table are all below 10 in absolute value)."""
    c = f"{coef:+.3f}".replace("0.", ".", 1)
    s = f"{se:.3f}".replace("0.", ".", 1)
    return f"${c}{stars(p)}$ ({s})"


def summary_table(kind: str = "ceiling") -> str:
    """One table for every matched-scale result of the main text (rows = analysis x condition, columns = terms).

    kind="ceiling": the paper's outcomes (ceiling-normalized scores; MD = percentile rank of the raw score, MD ceilings unusable).
    kind="z":       the all-datasets fits only, outcome replaced by the per-unit robust z (2026-11-12: z dropped for bins / MD).
    """
    z = kind == "z"
    allc = pd.read_csv(REP / ("matched_scale_wholebrain_dll/coefficients_lineage_z.csv" if z
                              else "matched_scale_wholebrain_dll/coefficients_lineage.csv"))
    allm = "4_lineage_z" if z else "3_lineage"
    # 2026-11-12: bins and networks refitted with the main fit's lineage random intercept (scripts/analysis/lineage_refits.R)
    dlt = pd.read_csv(REP / DLT_DIR / "coefficients_lineage.csv")
    frq = pd.read_csv(REP / FREQ_DIR / "coefficients_lineage.csv")
    binm = "3_lineage"
    md = md_lineage()
    mdA, mdB = md[md.fit.str.startswith("A_per_network")], md[md.fit == "B_pooled"]

    def row(label: str, rows: list) -> str:
        return rf"\quad {label} & " + " & ".join(tiny_cell(r.coef, r.se, r.p) for r in rows) + r" \\"

    def head(text: str) -> str:
        return rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{text}}}}} \\"

    L = [head(r"All datasets \quad (16 reading measures; 8 neural datasets)")]
    for mod, lab in (("behavioral", "Behavior"), ("neural", "Brain")):
        L.append(row(lab, [get(allc, model=allm, modality=mod, term=t) for t, _ in SUMMARY_TERMS]))
    for c, title, bins in (() if z else (
            (dlt, r"Syntactic memory load \quad (Natural Stories; Podcast)",
             (("easy-easy", "easy--easy"), ("any-hard", "any-hard"), ("difference", r"$\Delta$ hard $-$ easy"))),
            (frq, r"Word frequency \quad (9 reading measures; Podcast)",
             (("easy-easy", "frequent--frequent"), ("any-hard", "any-rare"), ("difference", r"$\Delta$ rare $-$ frequent"))))):
        L.append(r"\addlinespace")
        L.append(head(title))
        for mod, lab in (("behavioral", "Behavior"), ("neural", "Brain")):
            for b, blab in bins:
                L.append(row(f"{lab}, {blab}", [get(c, model=binm, modality=mod, term=t, bin=b) for t, _ in SUMMARY_TERMS]))
    if not z:
        L.append(r"\addlinespace")
        L.append(head(r"Language vs.\ multiple-demand network \quad (Pereira, Podcast, Tuckute)"))
        for net, lab in (("lang", "Brain, language"), ("md", "Brain, MD")):
            L.append(row(lab, [get(mdA, fit=f"A_per_network_{net}", term=t) for t, _ in SUMMARY_TERMS]))
        L.append(row(r"Brain, $\Delta$ MD $-$ language", [get(mdB, term=MD_DIFF[t]) for t, _ in SUMMARY_TERMS]))
    if z:
        cap = r"""Robustness check of the all-datasets rows of Table~\ref{tab:summary}: the same fits with each outcome replaced by
its robust $z$ score, $(x-\mathrm{median})/(\mathrm{IQR}/1.349)$ over models, computed within each dataset unit, so that no noise
reference is involved and each unit has the same spread of model scores."""
        label = "tab:summary-z"
    else:
        cap = r"""Matched-scale regressions (Equation~\ref{eq:group}) behind all claims of Section~\ref{sec:results}. Each row is
one fit (or, for $\Delta$ rows, the interaction of every fixed effect with the bin or network in a pooled fit). Outcomes: score
divided by the split-half reference (all datasets; word-pair bins, both bins by the all-pairs reference); percentile rank of the raw score among the 118 models within each dataset and network (language vs.\ MD,
whose references are too small to divide by). Fits: all datasets, dataset-unit and lineage random intercepts (REML,
Satterthwaite $t$-tests); word-pair bins, one fit over both bins with measure-by-bin, lineage and lineage-by-bin random
intercepts; networks, dataset fixed effects and a lineage random intercept (all REML, Satterthwaite $t$-tests). All datasets with a
robust-$z$ outcome: Table~\ref{tab:summary-z}."""
        label = "tab:summary"
    return rf"""\begin{{table}}[t]
\centering
\caption{{{cap} Coefficient (SE); reference: Standard LLM of the same size and perplexity. $^{{*}}p<.05$, $^{{**}}p<.01$, $^{{***}}p<.001$.}}
\label{{{label}}}
\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\toprule
 & Memory $\beta_M$ & Data $\beta_D$ & $M\times D$ $\beta_{{MD}}$ & $\log_{{10}}$ params $\beta_S$ & $\log_{{10}}$ PPL $\beta_P$ & C-BabyLM vs.\ LLM \\
\midrule
{chr(10).join(L)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


def md_dataset_table() -> str:
    """Appendix: language vs MD network per dataset (percentile rank of the raw dLL, OLS with model-clustered SE)."""
    md = md_lineage()
    dss = (("pereira", "Pereira"), ("podcast", "Podcast"), ("tuckute", "Tuckute"))
    lines = []
    for t, name in SUMMARY_TERMS:
        name = name or r"C-BabyLM vs.\ LLM"
        cells = [tiny_cell(r.coef, r.se, r.p) for ds, _ in dss for net in ("lang", "md")
                 for r in [get(md, fit=f"A_{ds}_{net}", term=t)]]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    return rf"""\begin{{table}}[t]
\centering
\caption{{Language vs.\ multiple-demand network per dataset: matched-scale regression (Equation~\ref{{eq:group}} without random
effects) of the percentile rank of the raw $\Delta\ell$ among the 118 models (as in Table~\ref{{tab:summary}}), one row per model, lineage random intercept (REML, Satterthwaite $t$-tests). Coefficient (SE);
reference: Standard LLM. $^{{*}}p<.05$, $^{{**}}p<.01$, $^{{***}}p<.001$.}}
\label{{tab:md-datasets}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\toprule
 & \multicolumn{{2}}{{c}}{{Pereira (fMRI)}} & \multicolumn{{2}}{{c}}{{Podcast (ECoG)}} & \multicolumn{{2}}{{c}}{{Tuckute (fMRI)}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}\cmidrule(lr){{6-7}}
 & Language & MD & Language & MD & Language & MD \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


def two_table(ns: pd.DataFrame) -> str:
    """Appendix: the two-dataset (Natural Stories x Podcast) fit, ceiling-normalized (robust z dropped 2026-11-12)."""
    lines = []
    for t, name in TERMS:
        cells = [short_cell(get(ns, model=mn, modality=mod, term=t))
                 for mn in ("3_ceiling",) for mod in ("behavioral", "neural")]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    return rf"""\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{4pt}}
\caption{{Two-dataset matched-scale regression: Natural Stories self-paced reading (118 models) and Podcast ECoG, language
network (116 models); one row per model, OLS with family-clustered standard errors. Outcome: score divided by the split-half
reference. Coefficient (SE); reference: Standard LLM. $^{{*}}p<.05$, $^{{**}}p<.01$, $^{{***}}p<.001$.}}
\label{{tab:rsa-two}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lcc}}
\toprule
 & Behavior & Brain \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


FREQ_UNITS = [("geco", "GECO first pass", 1010), ("geco_gp", "GECO go-past", 1010), ("onestop_fp", "OneStop first pass", 1043),
              ("onestop_gp", "OneStop go-past", 1044), ("onestop_sp", "OneStop scan path", 1043),
              ("naturalstories_spr", "Natural Stories SPR", 275), ("zuco_reading", "ZuCo normal reading", 328),
              ("zuco_reading_sr", "ZuCo sentiment", 360), ("zuco_reading_tsr", "ZuCo relation", 345)]


def freq_measure_table() -> str:
    """Per-measure frequency-bin fits (scripts/analysis/freq_bins_per_measure.py -> <FREQ_DIR>/per_measure.csv)."""
    d = pd.read_csv(REP / FREQ_DIR / "per_measure.csv").set_index(["unit", "bin"])
    lines = []
    for u, name, nrare in FREQ_UNITS:
        f, r = d.loc[(u, "easy-easy")], d.loc[(u, "any-hard")]
        cb = lambda x: f"${x.cb_ref:+.3f}{stars(x.cb_ref_p)}$ ({x.cb_ref_se:.3f})"
        lines.append(rf"{name} & {nrare:,} & {f['median']:.2f} / {r['median']:.2f} & {f.robust_sd:.3f} / {r.robust_sd:.3f} & "
                     rf"{cb(f)} & {cb(r)} \\")
    return r"""\begin{table}[h]
\centering
\caption{Frequency bins per reading measure. For each measure and bin, an OLS fit of Equation~\ref{eq:group} over the 118 models
(HC3 standard errors), outcome = score divided by the all-pairs reference; the C-BabyLM column is the contrast
$\beta_M+\beta_D+\beta_{MD}$ (Constrained-BabyLM vs.\ Standard LLM at matched size and perplexity). Median and spread (robust SD,
IQR$/1.349$) of the model scores are given as frequent--frequent / any-rare. Within a measure, the robust-$z$ coefficient is the
coefficient divided by the spread, with the same $p$. $^{*}p<.05$, $^{**}p<.01$, $^{***}p<.001$.}
\label{tab:freq-measures}
\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lrcccc@{}}
\toprule
Measure & Rare words & Median & Spread & C-BabyLM, frequent--frequent & C-BabyLM, any-rare \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""


def main():
    ns = pd.read_csv(REP / "ns_podcast/coefficients.csv")
    m = pd.read_csv(REP / "matched_scale_wholebrain_dll/coefficients_lineage.csv")
    (SEC / "rsa_main_table.tex").write_text(main_table(ns, m))
    (SEC / "summary_table.tex").write_text(summary_table("ceiling"))
    (SEC / "summary_z_table.tex").write_text(summary_table("z"))
    (SEC / "rsa_two_table.tex").write_text(two_table(ns))
    (SEC / "md_dataset_table.tex").write_text(md_dataset_table())
    (SEC / "freq_measure_table.tex").write_text(freq_measure_table())
    # rare-word surprisal test removed from the paper (2026-11-15, user)
    # (SEC / "rare_surprisal_table.tex").write_text(rare_surprisal_table())
    d = pd.read_csv(REP / DLT_DIR / "coefficients_lineage.csv")
    (SEC / "rsa_dlt_table.tex").write_text(dlt_table(d))
    f = pd.read_csv(REP / FREQ_DIR / "coefficients_lineage.csv")
    (SEC / "rsa_freq_table.tex").write_text(freq_table(f))
    e = pd.read_csv(REP / "matched_scale_wholebrain_embedding_both/coefficients_lineage.csv")
    u = pd.read_csv(REP / "matched_scale_wholebrain_surprisal/coefficients_lineage.csv")
    (SEC / "rsa_sameq_table.tex").write_text(same_quantity_table(e, u))
    CKS = ROOT / "paper_results/scratch/ckpt_rsa"
    (SEC / "rsa_ckpt_table.tex").write_text(ckpt_table(pd.read_csv(CKS / "matched_scale/coefficients.csv"), "pooled"))
    (SEC / "rsa_ckpt_two_table.tex").write_text(ckpt_table(pd.read_csv(CKS / "ns_podcast/coefficients.csv"), "two"))
    ML = ROOT / "outputs/benchmark_summary_multilayer/_matched_scale"
    MS = ROOT / "outputs_jeanzay/matched_scale_report"
    (SEC / "direct_embedding_table.tex").write_text(direct_table(
        {"last": ML / "regression_coefficients_embedding_multilayer.csv",
         "best": ML / "regression_coefficients_embedding_bestlayer_multilayer.csv"},
        r"Matched-scale regression on the direct embedding scores (ridge encoding / decoding, Pearson $r$ divided by the "
        r"correlation ceiling), fitted separately per domain with a dataset random intercept (REML, Wald tests); 118 models, 1,870 behavioral and "
        r"1,290 neural rows.",
        "tab:direct-embedding", "Coefficient (SE); reference = Standard LLM."))
    (SEC / "direct_surprisal_table.tex").write_text(direct_table(
        {"last": MS / "regression_coefficients_surprisal.csv", "best": MS / "regression_coefficients_surprisal_bestlayer.csv"},
        r"Matched-scale regression on the direct surprisal scores (pooled LME, held-out $\Delta$log-likelihood divided by the "
        r"$\Delta$log-likelihood ceiling; best layer chosen on the training $\Delta$log-likelihood), fitted separately per domain "
        r"with a dataset random intercept (REML, Wald tests); 118 models, 1,870 behavioral and 818 neural rows (Broderick, Federmeier "
        r"2007, Hubbard 2019 and Pereira are left out because their $\Delta$log-likelihood references are unusable).",
        "tab:direct-surprisal", "Coefficient (SE); reference = Standard LLM."))
    print("wrote summary{,_z}_table.tex, rsa_{main,two,dlt,freq,sameq,ckpt,ckpt_two}_table.tex, direct_{embedding,surprisal}_table.tex")


if __name__ == "__main__":
    main()
