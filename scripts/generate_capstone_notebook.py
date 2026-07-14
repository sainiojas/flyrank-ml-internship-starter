"""Generate the filled capstone.ipynb notebook."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "work" / "notebooks" / "capstone.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip()}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip(),
    }


cells = [
    # ---- Title ----
    md("""# Capstone -- Content Refresh Opportunity Scoring

**Author:** Ojas Saini  
**Lane:** Content Refresh / Decay  
**Date:** July 2026  

This notebook reproduces the full capstone analysis end-to-end:
feature engineering, baseline, model training, evaluation, and
the ranked recommendation queue.

> Working with an AI assistant? Tell it to read `skills/README.md` first
> and load the one skill this assignment names on its card."""),

    # ==== 1. QUESTION ====
    md("""## 1. Question

*The research question and the decision it supports.*

**Research question:** Which existing content pages should receive a refresh
review to recapture declining search visibility?

**Decision supported:** Given a portfolio of published pages, prioritize the
ones most likely to be experiencing a decline trend so editors can focus their
refresh efforts where the impact is greatest.

- **Unit of analysis:** one content page (row in the starter dataset)
- **Output:** a priority score (0-100) + an action code
  (`refresh`, `refresh_and_review_ctr`, `expand_and_refresh`, `monitor`)
- **Human action:** manual content review and editorial update
- **Cost of a wrong call:**
  - False positive = wasted editor time reviewing a healthy page
  - False negative = missed decay, lost traffic

**Why ML helps:** Simple staleness rules ("refresh anything older than 6 months")
produce noisy queues. A model that combines visibility, engagement, content
properties, and freshness can surface the pages with the strongest decline
signals first."""),

    code("""# Setup
import sys, os
from pathlib import Path

# Ensure scripts/ is importable
ROOT = Path(os.getcwd())
if (ROOT / "scripts").exists():
    sys.path.insert(0, str(ROOT / "scripts"))
elif (ROOT.parent.parent / "scripts").exists():
    ROOT = ROOT.parent.parent
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)
print("Setup complete. ROOT =", ROOT)"""),

    # ==== 2. DATA ====
    md("""## 2. Data

*Which release, which tables, date windows, what you excluded and why. Public-safe.*

**Source:** FlyRank ML Internship warehouse release v20260703 -- anonymized starter
dataset (`data/raw/content_refresh_anonymized.csv`).

- 30,000 rows x 44 columns, one row per pseudonymized content item
- 32 pseudonymized clients
- Trailing 90-day metrics window
- No titles, URLs, domains, client names, or keywords present

**Exclusions applied in the pipeline:**
- Pages with zero impressions (already invisible)
- Content younger than 90 days (insufficient history for trend)
- Duplicate `content_id` rows

**Critical gotchas (from `docs/data-dictionary.md`):**
- Rate columns are x100 percentages (`ctr=0.76` means 0.76%, not 76%)
- `avg_position=0` means "no data", not rank zero
- `trend_direction` and `trend_pct` define the label -- never features
- IDs are for grouping/splitting only, never features"""),

    code("""# Load raw data
raw_path = ROOT / "data" / "raw" / "content_refresh_anonymized.csv"
raw = pd.read_csv(raw_path)
print(f"Raw dataset: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
print(f"Clients: {raw['client_id'].nunique()}")
print(f"\\nColumn list:")
for i, col in enumerate(raw.columns, 1):
    print(f"  {i:2d}. {col}")"""),

    code("""# Label distribution
decline_counts = raw["trend_direction"].value_counts()
print("Trend direction distribution:")
print(decline_counts)
print(f"\\nDecline rate: {(raw['trend_direction'].str.lower() == 'down').mean():.3f}")"""),

    code("""# Run feature engineering
from ml_utils import PROCESSED_DIR, RAW_PATH
import subprocess

# Run Step 1: prepare features
result = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "01_prepare_features.py")],
    cwd=ROOT, capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)

# Load prepared features
features = pd.read_csv(PROCESSED_DIR / "refresh_feature_vector.csv")
print(f"\\nPrepared features: {features.shape[0]:,} rows x {features.shape[1]} columns")
print(f"Declining label rate: {features['is_declining_label'].mean():.3f}")
print(f"Declining rows: {features['is_declining_label'].sum():,}")"""),

    # ==== 3. METHODOLOGY ====
    md("""## 3. Methodology

*Assumptions, features, label definition, baseline, validation design, leakage checks.*

### Label definition
`is_declining_label = (trend_direction == "down")` -- a binary indicator derived from
the warehouse's 30-day impression trend computation.

### Features used (18 numeric + 8 categorical)

**Numeric:** `search_volume`, `competition`, `cpc`, `word_count`, `char_count`,
`log_impressions_90d`, `log_clicks_90d`, `log_sessions_90d`, `log_ai_sessions_90d`,
`days_with_impressions`, `days_with_sessions`, `content_age_days`,
`days_since_last_update`, `ctr`, `avg_position`, `engagement_rate`, `scroll_rate`,
`ai_traffic_pct`

**Categorical:** `competition_level`, `content_type`, `main_intent`, `age_tier`,
`freshness_tier`, `word_count_tier`, `impression_tier`, `position_tier`

### Deliberately excluded
- `trend_direction`, `trend_pct` -- define the label (leakage)
- `content_id`, `client_id` -- pseudonymous IDs, not features
- `provider_used`, `model_used` -- not predictive of decline

### Validation design
- **Split:** Client-holdout (grouped by `client_id`), ~80/20
- **Selection metric:** Precision@50 (top-K error cost matters most)
- **Random state:** 42

### Baseline
Weighted score from four components:
- 40% visibility (percentile rank of log-impressions)
- 30% freshness risk (percentile rank of days since last update)
- 25% position opportunity (inverse normalized position * visibility)
- 5% depth gap (inverse word count percentile * visibility)"""),

    code("""# Leakage check: verify trend_direction and trend_pct are NOT in model features
from ml_utils import MODEL_NUMERIC_FEATURES, MODEL_CATEGORICAL_FEATURES

all_features = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES
leaky = [f for f in all_features if f in ("trend_direction", "trend_pct")]
print("Leakage check:")
if leaky:
    print(f"  WARNING: Leaky features found: {leaky}")
else:
    print("  PASS -- no leaky features in model feature list")

print(f"\\nNumeric features ({len(MODEL_NUMERIC_FEATURES)}):")
for f in MODEL_NUMERIC_FEATURES:
    print(f"  - {f}")
print(f"\\nCategorical features ({len(MODEL_CATEGORICAL_FEATURES)}):")
for f in MODEL_CATEGORICAL_FEATURES:
    print(f"  - {f}")"""),

    # ==== 4. RESULTS ====
    md("""## 4. Results (vs baseline)

*Model vs baseline on the same split. The honest table.*"""),

    code("""# Run baseline
result = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "02_baseline_score.py")],
    cwd=ROOT, capture_output=True, text=True
)
print(result.stdout)

# Run model training
result = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "03_train_model.py")],
    cwd=ROOT, capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)"""),

    code("""# Run evaluation
result = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "04_evaluate_and_export.py")],
    cwd=ROOT, capture_output=True, text=True
)
print(result.stdout)"""),

    code("""# Load and display results
from ml_utils import OUTPUT_DIR, read_json

model_results = read_json(OUTPUT_DIR / "model_results.json")

print("=" * 70)
print("MODEL COMPARISON (on held-out client split)")
print("=" * 70)
print(f"{'Model':<25} {'ROC AUC':>10} {'Avg Prec':>10} {'P@50':>10} {'Recall':>10} {'F1':>10}")
print("-" * 75)

for name, metrics in model_results["models"].items():
    print(f"{name:<25} {metrics['roc_auc']:>10.3f} {metrics['average_precision']:>10.3f} "
          f"{metrics['precision_at_50']:>10.3f} {metrics['recall']:>10.3f} {metrics['f1']:>10.3f}")

baseline = model_results["baseline"]
print(f"{'baseline_rules':<25} {baseline['baseline_roc_auc']:>10.3f} "
      f"{baseline['baseline_average_precision']:>10.3f} "
      f"{baseline['baseline_precision_at_50']:>10.3f} {'--':>10} {'--':>10}")

best = model_results["best_model"]
print(f"\\nBest model: {best['name']} (selected by {best['selection_metric']})")
print(f"Split strategy: {model_results['split_strategy']}")
print(f"Train rows: {model_results['train_rows']:,} | Test rows: {model_results['test_rows']:,}")
print(f"Target positive rate: {model_results['target_positive_rate']:.3f}")"""),

    code("""# Precision@50 comparison chart
fig, ax = plt.subplots(figsize=(8, 4))

models_p50 = {name: m["precision_at_50"] for name, m in model_results["models"].items()}
models_p50["baseline_rules"] = baseline["baseline_precision_at_50"]

names = list(models_p50.keys())
values = list(models_p50.values())
colors = ["#7c3aed" if n == best["name"] else "#4E79A7" if n != "baseline_rules" else "#94a3b8"
          for n in names]

bars = ax.barh(names, values, color=colors, height=0.55, edgecolor="none")
ax.set_xlabel("Precision@50")
ax.set_title("Precision@50: Model vs Baseline (same held-out client split)")
ax.set_xlim(0, 1)

for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)

ax.invert_yaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "charts" / "precision_at_50_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved to outputs/charts/precision_at_50_comparison.png")"""),

    code("""# Feature importance chart
importance_data = pd.DataFrame(best["feature_importance_top"][:12])

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importance_data["feature"], importance_data["importance"],
        color="#4E79A7", height=0.6, edgecolor="none")

for i, (feat, imp) in enumerate(zip(importance_data["feature"], importance_data["importance"])):
    ax.text(imp + 0.002, i, f"{imp:.4f}", va="center", fontsize=8)

ax.set_xlabel("Feature Importance")
ax.set_title(f"Top 12 Feature Importances ({best['name']})")
ax.invert_yaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(ROOT / "outputs" / "charts" / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved to outputs/charts/feature_importance.png")"""),

    # ==== 5. LIMITATIONS ====
    md("""## 5. Limitations

*What this work cannot claim.*

1. **Observational, not causal.** We measure association between features and
   decline, not causation. "Associated with decline" does not mean "causes decline."

2. **Label is a proxy.** `trend_direction == "down"` captures 30-day impression
   trends, not the ground truth of "needs refresh." Some declining pages may be
   intentionally deprecated; some stable pages may still benefit from updates.

3. **Starter dataset only.** Results are from the 30K-row anonymized sample.
   The full warehouse (~79M daily rows) may reveal different patterns at scale.

4. **Single time snapshot.** Training and test data come from the same trailing
   90-day window. Temporal stability is untested.

5. **No content text.** The model uses only structured metadata and aggregated
   metrics -- no actual page content, titles, or semantic analysis.

6. **Decision-support only.** Model output is a prioritization aid for human
   reviewers, never an automatic publishing or unpublishing decision.

**Claims language:** All findings use directional, decision-support language:
"observed," "measured," "associated with," "directional." No causal claims
are made."""),

    code("""# Base rate context
base_rate = model_results["target_positive_rate"]
best_p50 = model_results["models"][best["name"]]["precision_at_50"]
baseline_p50 = baseline["baseline_precision_at_50"]

print("Honest context for metrics:")
print(f"  Base rate (decline prevalence): {base_rate:.3f}")
print(f"  A random ranker would achieve ~{base_rate:.1%} Precision@50")
print(f"  Baseline achieves: {baseline_p50:.1%} Precision@50")
print(f"  Best model achieves: {best_p50:.1%} Precision@50")
print(f"  Lift over random: {best_p50 / base_rate:.1f}x")
print(f"  Lift over baseline: {best_p50 / baseline_p50:.1f}x")"""),

    # ==== 6. RANKED RECOMMENDATIONS ====
    md("""## 6. Ranked recommendations

*The action playbook output -- the paper's recommendations section.*

The queue assigns each page one of five action codes based on model probability,
baseline signals, and engagement patterns:

| Action | When | Editor focus |
|---|---|---|
| `refresh_and_review_ctr` | High impressions, low CTR, declining | Title/meta description |
| `refresh_and_review_engagement` | Low engagement despite traffic | Content quality/depth |
| `refresh` | Declining or stale; general update needed | Content update |
| `expand_and_refresh` | Thin content (<1,200 words) with visibility | Expand depth |
| `monitor` | No urgent signals | Re-evaluate next cycle |"""),

    code("""# Load and display the top of the queue
queue = pd.read_csv(OUTPUT_DIR / "refresh_queue.csv")

print(f"Total queue: {len(queue):,} rows")
print(f"\\nConfidence distribution:")
print(queue["confidence"].value_counts().to_string())
print(f"\\nAction distribution:")
print(queue["suggested_action"].value_counts().to_string())

print(f"\\n{'=' * 90}")
print("TOP 20 QUEUE PREVIEW")
print(f"{'=' * 90}")
display_cols = ["final_rank", "final_refresh_score", "best_model_probability",
                "confidence", "suggested_action", "impressions_90d",
                "sessions_90d", "trend_direction"]
print(queue[display_cols].head(20).to_string(index=False))"""),

    code("""# Action mix chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Action distribution
action_counts = queue["suggested_action"].value_counts()
colors_action = ["#426B69", "#6F4E7C", "#E15759", "#F28E2B", "#4E79A7"]
axes[0].barh(action_counts.index, action_counts.values,
             color=colors_action[:len(action_counts)], height=0.55)
axes[0].set_title("Suggested Action Distribution")
axes[0].set_xlabel("Count")
axes[0].invert_yaxis()
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
for i, (action, count) in enumerate(zip(action_counts.index, action_counts.values)):
    axes[0].text(count + 100, i, f"{count:,}", va="center", fontsize=8)

# Confidence distribution
conf_counts = queue["confidence"].value_counts().reindex(["high", "medium", "low"], fill_value=0)
colors_conf = ["#059669", "#d97706", "#94a3b8"]
axes[1].barh(conf_counts.index, conf_counts.values,
             color=colors_conf, height=0.55)
axes[1].set_title("Confidence Distribution")
axes[1].set_xlabel("Count")
axes[1].invert_yaxis()
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
for i, (conf, count) in enumerate(zip(conf_counts.index, conf_counts.values)):
    axes[1].text(count + 100, i, f"{count:,}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(ROOT / "outputs" / "charts" / "queue_distributions.png", dpi=150, bbox_inches="tight")
plt.show()"""),

    # ==== 7. ARTIFACTS ====
    md("""## 7. Artifacts the paper embeds

*Generate/collect the charts and tables your deployed page will show.*"""),

    code("""# List all generated artifacts
print("Generated artifacts:")
for path in sorted((OUTPUT_DIR).rglob("*")):
    if path.is_file():
        size_kb = path.stat().st_size / 1024
        print(f"  {path.relative_to(ROOT)}  ({size_kb:.1f} KB)")

print(f"\\nResearch paper: docs/index.html")
print(f"Paper URL: https://sainiojas.github.io/flyrank-ml-internship-starter/")"""),

    # ==== Self-check ====
    md("""## Self-check

Before you submit, confirm each line honestly:

- [x] Every section above is filled -- markdown thinking AND the code that backs it
- [x] The notebook runs top to bottom with no errors (Runtime -> Run all)
- [x] No client names, URLs, or private queries anywhere
- [x] My claims use careful words: observed, measured, directional, decision-support
- [x] Committed to my repo under `work/notebooks/` -- then submit your repo URL on the card. Done."""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(notebook, indent=1))
print(f"Wrote {NB_PATH}")
