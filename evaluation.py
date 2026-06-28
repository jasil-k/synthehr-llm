# evaluation.py
"""
Phase 5: Statistical Fidelity Evaluation for SynthEHR-LLM

Compares a generated synthetic cohort against approximate real-world
population baselines, to demonstrate that the synthetic data isn't just
"plausible-looking" but statistically reasonable.

IMPORTANT: BASELINE_* values below are APPROXIMATE, ILLUSTRATIVE reference
figures based on general epidemiological knowledge (not pulled from a
specific cited real-world dataset). They exist to give the fidelity
comparison something concrete to compare against. In a production /
research context you would replace these with numbers from a real
source (CDC NHANES, WHO, etc.) and cite it.
"""

import math
from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import ks_2samp

from generator import PatientRecord, CODES_DB

# ════════════════════════════════════════════════════════════
# BASELINE REFERENCE STATS (approximate, illustrative — see module docstring)
# ════════════════════════════════════════════════════════════

BASELINE_DEMOGRAPHICS = {
    "male_pct": 49.0,
    "smoker_pct": 14.0,      # matches generator.py's own smoker weighting
    "age_mean": 55.0,        # rough midpoint for a 30-80 adult population
    "age_std": 14.0,
}

# Approximate adult population prevalence (%) for each ICD-10 code.
BASELINE_CONDITION_PREVALENCE = {
    "E11": 12.0,      # Type 2 diabetes
    "I10": 45.0,      # Hypertension
    "N18": 14.0,      # CKD
    "I25": 7.0,       # CAD
    "J44": 6.0,       # COPD
    "J45": 8.0,       # Asthma
    "I50": 2.0,       # Heart failure
    "E78": 25.0,      # Hyperlipidemia
    "M79.3": 1.0,     # Panniculitis
    "F32": 8.0,       # Major depressive disorder
    "F41": 10.0,      # Anxiety disorder
    "K21": 20.0,      # GERD
    "M81": 10.0,      # Osteoporosis
    "G47.3": 10.0,    # Sleep apnea
    "Z87.891": 20.0,  # Hx of nicotine dependence
}

# Approximate joint (co-occurrence) prevalence (%) in the general population
# -- i.e. "% of the whole population with BOTH conditions", not "% of
# diabetics who also have HTN".
BASELINE_COMORBIDITY_PAIRS = {
    ("E11", "I10"): 7.0,   # Diabetes + Hypertension
    ("I10", "I25"): 4.0,   # Hypertension + Coronary Artery Disease
    ("J44", "J45"): 2.0,   # COPD + Asthma
}

MIN_RECOMMENDED_N = 15  # below this, show a loud "small sample" disclaimer

CYAN = "#38bdf8"
VIOLET = "#a78bfa"


# ════════════════════════════════════════════════════════════
# STATISTICAL HELPERS
# ════════════════════════════════════════════════════════════

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """
    Wilson score confidence interval for a proportion.

    Why Wilson and not the textbook "p ± 1.96*sqrt(p(1-p)/n)" formula?
    That textbook ("normal approximation") formula breaks down badly at
    small n -- it can produce nonsensical intervals (negative lower bounds,
    intervals that don't shrink near 0%/100%). Wilson's interval stays
    well-behaved even at n=5-10, which is exactly our regime. It's the
    same method most stats libraries use under the hood for "proportion
    confidence interval".

    Returns (point_estimate_pct, ci_low_pct, ci_high_pct), all on a 0-100 scale.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    phat = successes / n
    denom = 1 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    margin = (z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n)) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return phat * 100, low * 100, high * 100


def proportion_fidelity_score(synthetic_pct: float, baseline_pct: float) -> float:
    """
    Simple, explainable fidelity score for one proportion metric:
    100 minus the absolute percentage-point gap from baseline, floored at 0.

    Deliberately NOT a p-value. At n=5-10, a p-value would be either
    meaningless or misleadingly reassuring. This answers a more honest,
    presentable question: "how close, in plain percentage points, did we
    land to the real-world rate?" -- also far easier to explain in an
    interview than a hypothesis-test result.
    """
    gap = abs(synthetic_pct - baseline_pct)
    return max(0.0, 100.0 - gap)


def age_fidelity_score(synthetic_mean: float, baseline_mean: float,
                        points_lost_per_year: float = 2.0) -> float:
    """
    Same philosophy as proportion_fidelity_score, but for a continuous
    variable (age). Loses `points_lost_per_year` points per year of drift
    between synthetic and baseline mean age (default: -2 pts/year, so a
    50-year drift zeroes the score -- deliberately generous since age
    ranges are user-controlled per generation run).
    """
    gap_years = abs(synthetic_mean - baseline_mean)
    return max(0.0, 100.0 - gap_years * points_lost_per_year)


# ════════════════════════════════════════════════════════════
# COMPUTE STATS FROM THE GENERATED COHORT
# ════════════════════════════════════════════════════════════

def compute_cohort_stats(cohort: list[PatientRecord]) -> dict:
    """
    Walks the in-memory cohort and computes the same summary statistics
    that BASELINE_* describes, so both sides are directly comparable.
    """
    n = len(cohort)
    if n == 0:
        return {"n": 0}

    ages = [p.demographics.age_at_generation for p in cohort]
    male_count = sum(1 for p in cohort if p.demographics.gender == "Male")
    smoker_count = sum(1 for p in cohort if p.demographics.smoker)

    # Each condition counted once per patient who has it anywhere in
    # primary_conditions (presence/absence, not visit-count).
    condition_counts = Counter()
    patient_code_sets = []
    for p in cohort:
        codes_present = {c.code for c in p.primary_conditions}
        patient_code_sets.append(codes_present)
        for code in codes_present:
            condition_counts[code] += 1

    comorbidity_counts = {}
    for pair in BASELINE_COMORBIDITY_PAIRS:
        code_a, code_b = pair
        joint = sum(1 for codes in patient_code_sets if code_a in codes and code_b in codes)
        comorbidity_counts[pair] = joint

    return {
        "n": n,
        "ages": ages,
        "age_mean": float(np.mean(ages)),
        "age_std": float(np.std(ages, ddof=1)) if n > 1 else 0.0,
        "male_count": male_count,
        "smoker_count": smoker_count,
        "condition_counts": condition_counts,
        "comorbidity_counts": comorbidity_counts,
    }


# ════════════════════════════════════════════════════════════
# BUILD COMPARISON TABLES
# ════════════════════════════════════════════════════════════

def build_demographics_comparison(stats: dict) -> pd.DataFrame:
    n = stats["n"]
    rows = []

    male_pct, male_lo, male_hi = wilson_ci(stats["male_count"], n)
    rows.append({
        "Metric": "% Male",
        "Synthetic": round(male_pct, 1),
        "95% CI": f"{male_lo:.1f}–{male_hi:.1f}",
        "Baseline": BASELINE_DEMOGRAPHICS["male_pct"],
        "Score": round(proportion_fidelity_score(male_pct, BASELINE_DEMOGRAPHICS["male_pct"]), 1),
    })

    smoker_pct, smoker_lo, smoker_hi = wilson_ci(stats["smoker_count"], n)
    rows.append({
        "Metric": "% Smoker",
        "Synthetic": round(smoker_pct, 1),
        "95% CI": f"{smoker_lo:.1f}–{smoker_hi:.1f}",
        "Baseline": BASELINE_DEMOGRAPHICS["smoker_pct"],
        "Score": round(proportion_fidelity_score(smoker_pct, BASELINE_DEMOGRAPHICS["smoker_pct"]), 1),
    })

    rows.append({
        "Metric": "Mean Age",
        "Synthetic": round(stats["age_mean"], 1),
        "95% CI": f"±{stats['age_std']:.1f} (std)",
        "Baseline": BASELINE_DEMOGRAPHICS["age_mean"],
        "Score": round(age_fidelity_score(stats["age_mean"], BASELINE_DEMOGRAPHICS["age_mean"]), 1),
    })

    return pd.DataFrame(rows)


def build_condition_comparison(stats: dict) -> pd.DataFrame:
    n = stats["n"]
    rows = []
    for code, baseline_pct in BASELINE_CONDITION_PREVALENCE.items():
        count = stats["condition_counts"].get(code, 0)
        synth_pct, lo, hi = wilson_ci(count, n)
        rows.append({
            "Code": code,
            "Condition": CODES_DB[code]["description"],
            "Synthetic %": round(synth_pct, 1),
            "95% CI": f"{lo:.1f}–{hi:.1f}",
            "Baseline %": baseline_pct,
            "Score": round(proportion_fidelity_score(synth_pct, baseline_pct), 1),
        })
    return pd.DataFrame(rows).sort_values("Baseline %", ascending=False).reset_index(drop=True)


def build_comorbidity_comparison(stats: dict) -> pd.DataFrame:
    n = stats["n"]
    rows = []
    for pair, baseline_pct in BASELINE_COMORBIDITY_PAIRS.items():
        code_a, code_b = pair
        joint = stats["comorbidity_counts"].get(pair, 0)
        synth_pct, lo, hi = wilson_ci(joint, n)
        label = f"{code_a} + {code_b}"
        desc = f"{CODES_DB[code_a]['description']} + {CODES_DB[code_b]['description']}"
        rows.append({
            "Pair": label,
            "Conditions": desc,
            "Synthetic %": round(synth_pct, 1),
            "95% CI": f"{lo:.1f}–{hi:.1f}",
            "Baseline %": baseline_pct,
            "Score": round(proportion_fidelity_score(synth_pct, baseline_pct), 1),
        })
    return pd.DataFrame(rows)


def compute_overall_fidelity_score(demo_df: pd.DataFrame, cond_df: pd.DataFrame,
                                    comorb_df: pd.DataFrame) -> float:
    """
    Single headline number: the unweighted average of every individual
    metric's score across demographics, condition prevalence, and
    comorbidity tables. Unweighted on purpose -- "weighting by clinical
    importance" would just be another subjective judgment call, and an
    unweighted average is far easier to explain plainly.
    """
    all_scores = (
        list(demo_df["Score"]) + list(cond_df["Score"]) + list(comorb_df["Score"])
    )
    if not all_scores:
        return 0.0
    return float(np.mean(all_scores))


# ════════════════════════════════════════════════════════════
# OPTIONAL SUPPLEMENTARY: KS TEST FOR AGE (clearly caveated)
# ════════════════════════════════════════════════════════════

def age_ks_test(stats: dict, baseline_mean: float, baseline_std: float,
                 simulated_baseline_n: int = 2000, seed: int = 42) -> dict:
    """
    Supplementary check only. We don't have a real baseline AGE sample
    (just a mean/std), so we simulate one from a Normal distribution
    matching the baseline mean/std, then run a two-sample
    Kolmogorov-Smirnov test against the synthetic cohort's ages.

    CAVEAT (also shown in the UI): at n=5-15 synthetic patients, this test
    has very low statistical power -- it will rarely flag a real
    difference. A high p-value here is NOT proof of a good match, it's a
    secondary signal at best.
    """
    rng = np.random.default_rng(seed)
    simulated_baseline = rng.normal(baseline_mean, baseline_std, simulated_baseline_n)
    simulated_baseline = np.clip(simulated_baseline, 18, 100)
    stat, p_value = ks_2samp(stats["ages"], simulated_baseline)
    return {"ks_statistic": float(stat), "p_value": float(p_value)}


# ════════════════════════════════════════════════════════════
# PLOTLY CHART BUILDERS (dark theme, matches GLASS_CSS cyan/violet accents)
# ════════════════════════════════════════════════════════════

def _style_fig(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(color="#e5e7eb"),
    )
    return fig


def build_condition_figure(cond_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(name="Synthetic", x=cond_df["Code"], y=cond_df["Synthetic %"], marker_color=CYAN)
    fig.add_bar(name="Baseline", x=cond_df["Code"], y=cond_df["Baseline %"], marker_color=VIOLET)
    fig.update_layout(barmode="group", yaxis_title="Prevalence (%)")
    return _style_fig(fig, "Condition Prevalence: Synthetic vs Baseline")


def build_comorbidity_figure(comorb_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(name="Synthetic", x=comorb_df["Pair"], y=comorb_df["Synthetic %"], marker_color=CYAN)
    fig.add_bar(name="Baseline", x=comorb_df["Pair"], y=comorb_df["Baseline %"], marker_color=VIOLET)
    fig.update_layout(barmode="group", yaxis_title="Joint Prevalence (%)")
    return _style_fig(fig, "Comorbidity Co-occurrence: Synthetic vs Baseline")


def build_demographics_figure(demo_df: pd.DataFrame) -> go.Figure:
    pct_rows = demo_df[demo_df["Metric"] != "Mean Age"]
    fig = go.Figure()
    fig.add_bar(name="Synthetic", x=pct_rows["Metric"], y=pct_rows["Synthetic"], marker_color=CYAN)
    fig.add_bar(name="Baseline", x=pct_rows["Metric"], y=pct_rows["Baseline"], marker_color=VIOLET)
    fig.update_layout(barmode="group", yaxis_title="%")
    return _style_fig(fig, "Demographics: Synthetic vs Baseline")


# ════════════════════════════════════════════════════════════
# MAIN ENTRY POINT — call this from app.py
# ════════════════════════════════════════════════════════════

def render_evaluation_section(cohort: list[PatientRecord]) -> None:
    """
    Renders the entire 'Evaluation' tab. Call from app.py like:

        from evaluation import render_evaluation_section
        ...
        elif mode == "Evaluation":
            render_evaluation_section(st.session_state.cohort)
    """
    st.header("📊 Statistical Fidelity Evaluation")
    st.caption(
        "Compares the generated synthetic cohort against approximate, "
        "illustrative real-world population baselines (not a cited "
        "clinical dataset — see evaluation.py docstring)."
    )

    if not cohort:
        st.info("No cohort generated yet. Go to **Bulk Cohort** and generate "
                 "some patients first, then come back here.")
        return

    stats = compute_cohort_stats(cohort)
    n = stats["n"]

    if n < MIN_RECOMMENDED_N:
        st.warning(
            f"⚠️ Sample size is small (n={n}). All fidelity numbers below "
            "are **directional, not statistically certain** — confidence "
            "intervals will be wide, and the overall score should be read "
            "as 'roughly in the right ballpark' rather than a precise claim."
        )

    demo_df = build_demographics_comparison(stats)
    cond_df = build_condition_comparison(stats)
    comorb_df = build_comorbidity_comparison(stats)
    overall_score = compute_overall_fidelity_score(demo_df, cond_df, comorb_df)

    st.subheader("🏆 Overall Fidelity Score")
    st.metric(
        label=f"Across {len(demo_df) + len(cond_df) + len(comorb_df)} metrics, n={n} patients",
        value=f"{overall_score:.1f} / 100",
    )
    st.caption(
        "Score = average of (100 − |synthetic% − baseline%|) across every "
        "metric below. Not a p-value — a plain-English 'how close in "
        "percentage points' summary, chosen because formal significance "
        "tests are unreliable at this sample size."
    )

    st.divider()
    st.subheader("Demographics")
    st.plotly_chart(build_demographics_figure(demo_df), use_container_width=True)
    st.dataframe(demo_df, use_container_width=True, hide_index=True)

    ks = age_ks_test(stats, BASELINE_DEMOGRAPHICS["age_mean"], BASELINE_DEMOGRAPHICS["age_std"])
    st.caption(
        f"Supplementary KS test (age vs simulated baseline distribution): "
        f"statistic={ks['ks_statistic']:.3f}, p={ks['p_value']:.3f}. "
        "⚠️ Low statistical power at this sample size — a high p-value here "
        "is NOT proof of a good match, just an absence of strong evidence "
        "against it."
    )

    st.divider()
    st.subheader("Condition Prevalence")
    st.plotly_chart(build_condition_figure(cond_df), use_container_width=True)
    st.dataframe(cond_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Comorbidity Co-occurrence")
    st.plotly_chart(build_comorbidity_figure(comorb_df), use_container_width=True)
    st.dataframe(comorb_df, use_container_width=True, hide_index=True)