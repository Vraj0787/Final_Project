from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import webbrowser

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(script_dir: Path) -> Path:
    for candidate in [script_dir, *script_dir.parents]:
        if (candidate / "archive").exists():
            return candidate
    return script_dir


def find_data_path(project_root: Path, filename: str) -> Path:
    candidates = [
        SCRIPT_DIR / "archive" / filename,
        SCRIPT_DIR.parent / "data" / filename,
        project_root / "archive" / filename,
        project_root / "final_submission" / "data" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


PROJECT_ROOT = find_project_root(SCRIPT_DIR)


def determine_output_dir(project_root: Path, script_dir: Path) -> Path:
    if script_dir.name == "code" and script_dir.parent.name == "final_submission":
        return script_dir.parent / "outputs"
    return project_root / "final_submission" / "outputs"


OUTPUT_DIR = determine_output_dir(PROJECT_ROOT, SCRIPT_DIR)
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
CLEANED_DIR = OUTPUT_DIR / "cleaned_data"
TEXT_DIR = OUTPUT_DIR / "notes"
MPLCONFIG_DIR = OUTPUT_DIR / "mplconfig"

for directory in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, CLEANED_DIR, TEXT_DIR, MPLCONFIG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


LAYOFFS_PATH = find_data_path(PROJECT_ROOT, "layoffs_events.csv")
NEWS_PATH = find_data_path(PROJECT_ROOT, "news_sentiment.csv")
US_LABOR_PATH = find_data_path(PROJECT_ROOT, "us_labor_indicators.csv")
GLOBAL_LABOR_PATH = find_data_path(PROJECT_ROOT, "global_labor_indicators.csv")

RESEARCH_QUESTIONS = [
    "How did layoff activity evolve from 2020 to 2026, and how much of it came from AI companies?",
    "Do AI companies differ from non-AI companies in layoff size, workforce share affected, and funding profile?",
    "Which industries, countries, and company stages contributed the most recorded layoffs?",
    "Is funding associated with larger layoffs after accounting for the strong right-skew in layoff counts?",
    "How does layoff-focused news sentiment behave over time, and is it systematically different from non-layoff coverage?",
    "How do monthly layoff totals align with U.S. labor indicators such as unemployment, job openings, and jobless claims?",
    "Across countries, how are layoff totals related to unemployment, youth unemployment, and employment-to-population ratios?",
]

COUNTRY_FIXES = {
    "United Arab E…": "United Arab Emirates",
    "United Kingdo…": "United Kingdom",
}

INDUSTRY_FIXES = {
    "Infrastructu…": "Infrastructure",
    "Manufactur…": "Manufacturing",
    "Transportat…": "Transportation",
}


def clean_currency(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_percentage(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace("", np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def mean_confidence_interval(series: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    clean = series.dropna().astype(float)
    if len(clean) < 2:
        return (math.nan, math.nan)
    mean = clean.mean()
    sem = stats.sem(clean, nan_policy="omit")
    lower, upper = stats.t.interval(confidence, len(clean) - 1, loc=mean, scale=sem)
    return float(lower), float(upper)


def bootstrap_mean_difference(
    series_a: pd.Series, series_b: pd.Series, iterations: int = 5000, seed: int = 42
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    a = series_a.dropna().to_numpy(dtype=float)
    b = series_b.dropna().to_numpy(dtype=float)
    boot_diffs = np.empty(iterations)
    for i in range(iterations):
        boot_diffs[i] = rng.choice(a, size=len(a), replace=True).mean() - rng.choice(
            b, size=len(b), replace=True
        ).mean()
    return float(np.mean(boot_diffs)), float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))


def cohens_d(series_a: pd.Series, series_b: pd.Series) -> float:
    a = series_a.dropna().to_numpy(dtype=float)
    b = series_b.dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return math.nan
    pooled_var = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    if pooled_var <= 0:
        return math.nan
    return float((a.mean() - b.mean()) / math.sqrt(pooled_var))


def multiple_linear_regression(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> dict[str, float]:
    reg_df = df.dropna(subset=[y_col] + x_cols).copy()
    y = reg_df[y_col].to_numpy(dtype=float)
    X = reg_df[x_cols].to_numpy(dtype=float)
    X_design = np.column_stack([np.ones(len(reg_df)), X])
    beta, residuals, rank, singular_values = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta
    sse = float(np.sum((y - y_hat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r_squared = float(1 - sse / sst) if sst > 0 else math.nan
    return {
        "n_obs": int(len(reg_df)),
        "intercept": float(beta[0]),
        **{f"beta_{name}": float(value) for name, value in zip(x_cols, beta[1:])},
        "r_squared": r_squared,
        "rank": int(rank),
        "sse": sse,
    }


def scatter_with_trend(
    x: pd.Series,
    y: pd.Series,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    color: str,
) -> None:
    # Visualization method: scatter plot with a fitted linear trend line for relationship analysis.
    plot_df = pd.DataFrame({"x": x, "y": y}).dropna()
    if plot_df.empty:
        return
    fit = stats.linregress(plot_df["x"], plot_df["y"])
    x_line = np.linspace(plot_df["x"].min(), plot_df["x"].max(), 100)
    y_line = fit.intercept + fit.slope * x_line

    plt.figure(figsize=(8, 6))
    plt.scatter(plot_df["x"], plot_df["y"], alpha=0.7, color=color)
    plt.plot(x_line, y_line, color="#1a202c", linewidth=2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def write_research_questions() -> None:
    with (TEXT_DIR / "research_questions.txt").open("w", encoding="utf-8") as handle:
        for idx, question in enumerate(RESEARCH_QUESTIONS, start=1):
            handle.write(f"{idx}. {question}\n")


def load_layoffs() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(LAYOFFS_PATH)
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["layoff_count"] = pd.to_numeric(df["layoff_count"], errors="coerce")
    df["pct_workforce"] = clean_percentage(df["pct_workforce"])
    df["raised_mm"] = clean_currency(df["raised_mm"])
    df["is_ai_company"] = df["is_ai_company"].astype(bool)
    for column in ["company", "location", "industry", "stage", "country", "source_url"]:
        df[column] = df[column].fillna("Unknown").astype(str).str.strip()
    df["country"] = df["country"].replace(COUNTRY_FIXES).replace({"": "Unknown"})
    df["industry"] = df["industry"].replace(INDUSTRY_FIXES).replace({"": "Unknown"})
    df["stage"] = df["stage"].replace({"": "Unknown"})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["layoff_count_filled"] = df["layoff_count"].fillna(0)
    df["pct_workforce_filled"] = df["pct_workforce"].fillna(0)
    df["log_layoff_count"] = np.log1p(df["layoff_count"])
    df["log_raised_mm"] = np.log1p(df["raised_mm"])
    df.to_csv(CLEANED_DIR / "layoffs_events_cleaned.csv", index=False)
    return raw, df


def load_news() -> pd.DataFrame:
    df = pd.read_csv(NEWS_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    for column in ["title", "source", "description", "url", "sentiment_cat"]:
        df[column] = df[column].fillna("Unknown").astype(str).str.strip()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df.to_csv(CLEANED_DIR / "news_sentiment_cleaned.csv", index=False)
    return df


def load_us_labor() -> pd.DataFrame:
    df = pd.read_csv(US_LABOR_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df.to_csv(CLEANED_DIR / "us_labor_indicators_cleaned.csv", index=False)
    return df


def load_global_labor() -> pd.DataFrame:
    df = pd.read_csv(GLOBAL_LABOR_PATH)
    df["country_name"] = df["country_name"].astype(str).str.strip()
    df.to_csv(CLEANED_DIR / "global_labor_indicators_cleaned.csv", index=False)
    return df


def save_data_quality(raw_layoffs: pd.DataFrame, layoffs: pd.DataFrame, news: pd.DataFrame, us_labor: pd.DataFrame, global_labor: pd.DataFrame) -> None:
    # Data quality requirement: document missing values before and after cleaning for each dataset.
    frames = {
        "layoffs_events": (raw_layoffs, layoffs),
        "news_sentiment": (news, news),
        "us_labor_indicators": (us_labor, us_labor),
        "global_labor_indicators": (global_labor, global_labor),
    }
    rows: list[dict[str, object]] = []
    for dataset_name, (raw_df, clean_df) in frames.items():
        for column in raw_df.columns:
            rows.append(
                {
                    "dataset": dataset_name,
                    "column": column,
                    "raw_missing": int(raw_df[column].isna().sum()),
                    "clean_missing": int(clean_df[column].isna().sum()) if column in clean_df.columns else math.nan,
                    "dtype_after_cleaning": str(clean_df[column].dtype) if column in clean_df.columns else "missing",
                }
            )
    pd.DataFrame(rows).to_csv(TABLES_DIR / "data_quality_summary.csv", index=False)


def analyze_layoffs(layoffs: pd.DataFrame) -> dict[str, object]:
    # Descriptive statistics requirement: compare AI and non-AI companies across core layoff metrics.
    ai = layoffs[layoffs["is_ai_company"]]
    non_ai = layoffs[~layoffs["is_ai_company"]]

    summary = pd.DataFrame(
        [
            {"metric": "rows", "value": len(layoffs)},
            {"metric": "date_min", "value": layoffs["date"].min().date().isoformat()},
            {"metric": "date_max", "value": layoffs["date"].max().date().isoformat()},
            {"metric": "countries_covered", "value": layoffs["country"].nunique()},
            {"metric": "industries_covered", "value": layoffs["industry"].nunique()},
            {"metric": "stages_covered", "value": layoffs["stage"].nunique()},
            {"metric": "ai_company_share_pct", "value": round(layoffs["is_ai_company"].mean() * 100, 2)},
        ]
    )
    summary.to_csv(TABLES_DIR / "layoffs_overall_summary.csv", index=False)

    numeric = layoffs[["layoff_count", "pct_workforce", "raised_mm"]].describe(percentiles=[0.25, 0.5, 0.75]).round(2)
    numeric.to_csv(TABLES_DIR / "layoffs_numeric_summary.csv")

    grouped = (
        layoffs.groupby("is_ai_company")[["layoff_count", "pct_workforce", "raised_mm"]]
        .agg(["count", "mean", "median", "std"])
        .round(2)
    )
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.to_flat_index()]
    grouped = grouped.reset_index()
    grouped.to_csv(TABLES_DIR / "layoffs_ai_vs_non_ai_summary.csv", index=False)

    monthly = (
        layoffs.groupby(["month", "is_ai_company"])
        .agg(events=("company", "size"), total_layoffs=("layoff_count", "sum"), avg_pct=("pct_workforce", "mean"))
        .reset_index()
    )
    monthly.to_csv(TABLES_DIR / "layoffs_monthly_summary.csv", index=False)

    top_industries = (
        layoffs.groupby("industry")
        .agg(events=("company", "size"), total_layoffs=("layoff_count", "sum"), mean_layoff=("layoff_count", "mean"))
        .sort_values("total_layoffs", ascending=False)
        .head(12)
        .round(2)
        .reset_index()
    )
    top_industries.to_csv(TABLES_DIR / "layoffs_top_industries.csv", index=False)

    top_countries = (
        layoffs.groupby("country")
        .agg(events=("company", "size"), total_layoffs=("layoff_count", "sum"), mean_layoff=("layoff_count", "mean"))
        .sort_values("total_layoffs", ascending=False)
        .head(12)
        .round(2)
        .reset_index()
    )
    top_countries.to_csv(TABLES_DIR / "layoffs_top_countries.csv", index=False)

    top_stages = (
        layoffs.groupby("stage")
        .agg(events=("company", "size"), total_layoffs=("layoff_count", "sum"), mean_layoff=("layoff_count", "mean"))
        .sort_values("total_layoffs", ascending=False)
        .head(12)
        .round(2)
        .reset_index()
    )
    top_stages.to_csv(TABLES_DIR / "layoffs_top_stages.csv", index=False)

    layoff_ai = ai["layoff_count"].dropna()
    layoff_non_ai = non_ai["layoff_count"].dropna()
    pct_ai = ai["pct_workforce"].dropna()
    pct_non_ai = non_ai["pct_workforce"].dropna()

    top_industry_names = layoffs["industry"].value_counts().head(6).index.tolist()
    industry_subset = layoffs[layoffs["industry"].isin(top_industry_names)].dropna(subset=["layoff_count"])
    industry_groups = [industry_subset.loc[industry_subset["industry"] == name, "layoff_count"] for name in top_industry_names]

    funding_df = layoffs.dropna(subset=["raised_mm", "layoff_count", "log_raised_mm", "log_layoff_count"])
    # Inferential methods requirement: hypothesis tests, effect size, bootstrap interval, and rank correlation.
    tests = {
        "layoff_count_welch_ttest_p": float(stats.ttest_ind(layoff_ai, layoff_non_ai, equal_var=False).pvalue),
        "layoff_count_mannwhitney_p": float(stats.mannwhitneyu(layoff_ai, layoff_non_ai, alternative="two-sided").pvalue),
        "layoff_count_cohens_d": cohens_d(layoff_ai, layoff_non_ai),
        "layoff_count_bootstrap_diff_mean": bootstrap_mean_difference(layoff_ai, layoff_non_ai)[0],
        "layoff_count_bootstrap_diff_ci_low": bootstrap_mean_difference(layoff_ai, layoff_non_ai)[1],
        "layoff_count_bootstrap_diff_ci_high": bootstrap_mean_difference(layoff_ai, layoff_non_ai)[2],
        "pct_workforce_welch_ttest_p": float(stats.ttest_ind(pct_ai, pct_non_ai, equal_var=False).pvalue),
        "pct_workforce_mannwhitney_p": float(stats.mannwhitneyu(pct_ai, pct_non_ai, alternative="two-sided").pvalue),
        "pct_workforce_cohens_d": cohens_d(pct_ai, pct_non_ai),
        "industry_kruskal_p": float(stats.kruskal(*industry_groups).pvalue),
        "funding_spearman_rho": float(stats.spearmanr(funding_df["raised_mm"], funding_df["layoff_count"]).statistic),
        "funding_spearman_p": float(stats.spearmanr(funding_df["raised_mm"], funding_df["layoff_count"]).pvalue),
    }
    pd.DataFrame([tests]).round(6).to_csv(TABLES_DIR / "layoffs_statistical_tests.csv", index=False)

    ai_share_by_industry = pd.crosstab(layoffs["industry"], layoffs["is_ai_company"])
    ai_share_by_industry = ai_share_by_industry.loc[ai_share_by_industry.sum(axis=1).sort_values(ascending=False).head(8).index]
    chi2, p_value, _, _ = stats.chi2_contingency(ai_share_by_industry)
    pd.DataFrame(
        [{"chi_square_statistic": round(float(chi2), 4), "p_value": round(float(p_value), 6)}]
    ).to_csv(TABLES_DIR / "layoffs_industry_ai_chi_square.csv", index=False)

    # Graph requirement: time-series line chart for monthly layoffs by company type.
    plt.figure(figsize=(11, 6))
    monthly_pivot = monthly.pivot(index="month", columns="is_ai_company", values="total_layoffs").fillna(0)
    plt.plot(monthly_pivot.index, monthly_pivot.get(False, pd.Series(dtype=float)), label="Non-AI companies", linewidth=2.0)
    plt.plot(monthly_pivot.index, monthly_pivot.get(True, pd.Series(dtype=float)), label="AI companies", linewidth=2.0)
    plt.title("Monthly Recorded Layoffs by Company Type")
    plt.xlabel("Month")
    plt.ylabel("Total layoffs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "layoffs_monthly_trend_ai_vs_non_ai.png", dpi=220)
    plt.close()

    # Graph requirement: boxplot to compare the distribution of log-scaled layoff sizes.
    plt.figure(figsize=(8, 6))
    box_groups = [np.log1p(layoff_ai), np.log1p(layoff_non_ai)]
    plt.boxplot(box_groups, tick_labels=["AI", "Non-AI"], patch_artist=True)
    plt.title("Log-Scaled Layoff Count by Company Type")
    plt.ylabel("log(1 + layoff_count)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "layoffs_ai_vs_non_ai_boxplot.png", dpi=220)
    plt.close()

    # Graph requirement: histogram to show the overall layoff-count distribution.
    plt.figure(figsize=(10, 6))
    plt.hist(np.log1p(layoffs["layoff_count"].dropna()), bins=30, color="#2b6cb0", alpha=0.85)
    plt.title("Distribution of Recorded Layoff Counts")
    plt.xlabel("log(1 + layoff_count)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "layoffs_distribution_histogram.png", dpi=220)
    plt.close()

    # Graph requirement: bar chart for the industries with the highest recorded layoffs.
    plt.figure(figsize=(11, 6))
    plt.bar(top_industries["industry"], top_industries["total_layoffs"], color="#2b6cb0")
    plt.title("Top Industries by Total Recorded Layoffs")
    plt.xlabel("Industry")
    plt.ylabel("Total layoffs")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "layoffs_top_industries.png", dpi=220)
    plt.close()

    # Graph requirement: bar chart for the countries with the highest recorded layoffs.
    plt.figure(figsize=(11, 6))
    plt.bar(top_countries["country"], top_countries["total_layoffs"], color="#dd6b20")
    plt.title("Top Countries by Total Recorded Layoffs")
    plt.xlabel("Country")
    plt.ylabel("Total layoffs")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "layoffs_top_countries.png", dpi=220)
    plt.close()

    # Graph requirement: scatter plot with regression line for funding versus layoff size.
    plt.figure(figsize=(9, 6))
    x = funding_df["log_raised_mm"]
    y = funding_df["log_layoff_count"]
    fit = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    plt.scatter(x, y, alpha=0.4, s=18, color="#1a202c")
    plt.plot(x_line, fit.intercept + fit.slope * x_line, color="#c53030", linewidth=2)
    plt.title("Funding Raised vs Layoff Count")
    plt.xlabel("log(1 + raised_mm)")
    plt.ylabel("log(1 + layoff_count)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "layoffs_funding_vs_size.png", dpi=220)
    plt.close()

    return {
        "overall": summary.to_dict(orient="records"),
        "layoff_ai_ci": mean_confidence_interval(layoff_ai),
        "layoff_non_ai_ci": mean_confidence_interval(layoff_non_ai),
        "pct_ai_ci": mean_confidence_interval(pct_ai),
        "pct_non_ai_ci": mean_confidence_interval(pct_non_ai),
        "tests": tests,
    }


def analyze_news(news: pd.DataFrame) -> dict[str, object]:
    # Sentiment-analysis requirement: summarize how news tone differs across layoff and non-layoff coverage.
    summary = pd.DataFrame(
        [
            {"metric": "rows", "value": len(news)},
            {"metric": "date_min", "value": news["date"].min().date().isoformat()},
            {"metric": "date_max", "value": news["date"].max().date().isoformat()},
            {"metric": "layoff_news_share_pct", "value": round(news["is_layoff_news"].mean() * 100, 2)},
            {"metric": "hiring_news_share_pct", "value": round(news["is_hiring_news"].mean() * 100, 2)},
        ]
    )
    summary.to_csv(TABLES_DIR / "news_overall_summary.csv", index=False)

    sentiment_summary = (
        news.groupby(["sentiment_cat", "is_layoff_news"])
        .agg(article_count=("title", "size"), mean_sentiment=("sentiment", "mean"))
        .round(4)
        .reset_index()
    )
    sentiment_summary.to_csv(TABLES_DIR / "news_sentiment_summary.csv", index=False)

    monthly = (
        news.groupby("month")
        .agg(
            article_count=("title", "size"),
            mean_sentiment=("sentiment", "mean"),
            layoff_news_share=("is_layoff_news", "mean"),
            hiring_news_share=("is_hiring_news", "mean"),
        )
        .round(4)
        .reset_index()
    )
    monthly.to_csv(TABLES_DIR / "news_monthly_summary.csv", index=False)

    layoff_sentiment = news.loc[news["is_layoff_news"], "sentiment"]
    non_layoff_sentiment = news.loc[~news["is_layoff_news"], "sentiment"]
    # Inferential methods requirement: compare layoff-news sentiment against other coverage.
    tests = {
        "layoff_vs_nonlayoff_welch_p": float(stats.ttest_ind(layoff_sentiment, non_layoff_sentiment, equal_var=False).pvalue),
        "layoff_vs_nonlayoff_mannwhitney_p": float(
            stats.mannwhitneyu(layoff_sentiment, non_layoff_sentiment, alternative="two-sided").pvalue
        ),
        "layoff_vs_nonlayoff_cohens_d": cohens_d(layoff_sentiment, non_layoff_sentiment),
        "layoff_sentiment_ci_low": mean_confidence_interval(layoff_sentiment)[0],
        "layoff_sentiment_ci_high": mean_confidence_interval(layoff_sentiment)[1],
        "nonlayoff_sentiment_ci_low": mean_confidence_interval(non_layoff_sentiment)[0],
        "nonlayoff_sentiment_ci_high": mean_confidence_interval(non_layoff_sentiment)[1],
    }
    pd.DataFrame([tests]).round(6).to_csv(TABLES_DIR / "news_statistical_tests.csv", index=False)

    # Graph requirement: time-series line chart for monthly average news sentiment.
    plt.figure(figsize=(11, 6))
    plt.plot(monthly["month"], monthly["mean_sentiment"], linewidth=2.0, color="#2f855a")
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("Average Monthly News Sentiment")
    plt.xlabel("Month")
    plt.ylabel("Average sentiment")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "news_monthly_sentiment_trend.png", dpi=220)
    plt.close()

    # Graph requirement: bar chart for counts in each sentiment category.
    plt.figure(figsize=(8, 6))
    cat_counts = news["sentiment_cat"].value_counts().reindex(["negative", "neutral", "positive"]).fillna(0)
    plt.bar(cat_counts.index, cat_counts.values, color=["#c53030", "#718096", "#2f855a"])
    plt.title("News Sentiment Category Counts")
    plt.xlabel("Sentiment category")
    plt.ylabel("Article count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "news_sentiment_category_counts.png", dpi=220)
    plt.close()

    # Graph requirement: bar chart comparing average sentiment by news type.
    plt.figure(figsize=(8, 6))
    compare = pd.DataFrame(
        {
            "Layoff news": [layoff_sentiment.mean()],
            "Non-layoff news": [non_layoff_sentiment.mean()],
            "Hiring news": [news.loc[news["is_hiring_news"], "sentiment"].mean()],
        }
    ).transpose()
    plt.bar(compare.index, compare[0], color=["#c53030", "#4a5568", "#2f855a"])
    plt.title("Average Sentiment by News Type")
    plt.ylabel("Average sentiment")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "news_average_sentiment_by_type.png", dpi=220)
    plt.close()

    return {"tests": tests}


def analyze_us_labor(layoffs: pd.DataFrame, us_labor: pd.DataFrame) -> dict[str, object]:
    # Data integration requirement: merge monthly layoffs with U.S. labor-market indicators.
    layoffs_monthly = (
        layoffs.groupby("month")
        .agg(global_total_layoffs=("layoff_count", "sum"), global_event_count=("company", "size"))
        .reset_index()
    )
    us_layoffs_monthly = (
        layoffs[layoffs["country"] == "United States"]
        .groupby("month")
        .agg(us_total_layoffs=("layoff_count", "sum"), us_event_count=("company", "size"))
        .reset_index()
    )
    us_monthly = (
        us_labor.groupby("month")
        .agg(
            unemployment_rate=("unemployment_rate", "mean"),
            jolts_job_openings_k=("jolts_job_openings_k", "mean"),
            initial_jobless_claims_k=("initial_jobless_claims_k", "mean"),
            openings_per_unemployed=("openings_per_unemployed", "mean"),
            tech_emp_yoy_pct=("tech_emp_yoy_pct", "mean"),
            claims_4w_avg=("claims_4w_avg", "mean"),
        )
        .reset_index()
    )

    merged = us_monthly.merge(layoffs_monthly, on="month", how="left").merge(us_layoffs_monthly, on="month", how="left")
    merged[["global_total_layoffs", "global_event_count", "us_total_layoffs", "us_event_count"]] = merged[
        ["global_total_layoffs", "global_event_count", "us_total_layoffs", "us_event_count"]
    ].fillna(0)
    merged.to_csv(TABLES_DIR / "us_labor_monthly_merged.csv", index=False)

    corr_cols = [
        "global_total_layoffs",
        "us_total_layoffs",
        "unemployment_rate",
        "jolts_job_openings_k",
        "initial_jobless_claims_k",
        "openings_per_unemployed",
        "tech_emp_yoy_pct",
    ]
    corr = merged[corr_cols].corr(numeric_only=True).round(4)
    corr.to_csv(TABLES_DIR / "us_labor_correlation_matrix.csv")

    spearman_global = stats.spearmanr(
        merged["global_total_layoffs"], merged["unemployment_rate"], nan_policy="omit"
    )
    spearman_claims = stats.spearmanr(
        merged["global_total_layoffs"], merged["initial_jobless_claims_k"], nan_policy="omit"
    )
    spearman_openings = stats.spearmanr(
        merged["global_total_layoffs"], merged["openings_per_unemployed"], nan_policy="omit"
    )
    regression = multiple_linear_regression(
        merged,
        "global_total_layoffs",
        ["unemployment_rate", "initial_jobless_claims_k", "openings_per_unemployed"],
    )
    # Statistical methods requirement: Spearman correlations plus multivariable regression.
    tests = {
        "global_layoffs_vs_unemployment_rho": float(spearman_global.statistic),
        "global_layoffs_vs_unemployment_p": float(spearman_global.pvalue),
        "global_layoffs_vs_claims_rho": float(spearman_claims.statistic),
        "global_layoffs_vs_claims_p": float(spearman_claims.pvalue),
        "global_layoffs_vs_openings_rho": float(spearman_openings.statistic),
        "global_layoffs_vs_openings_p": float(spearman_openings.pvalue),
        **regression,
    }
    pd.DataFrame([tests]).round(6).to_csv(TABLES_DIR / "us_labor_statistical_tests.csv", index=False)

    # Graph requirement: standardized time-series comparison across layoffs and labor indicators.
    plt.figure(figsize=(11, 6))
    z_df = merged[["month", "global_total_layoffs", "unemployment_rate", "openings_per_unemployed"]].copy()
    for column in ["global_total_layoffs", "unemployment_rate", "openings_per_unemployed"]:
        z_df[column] = (z_df[column] - z_df[column].mean()) / z_df[column].std(ddof=0)
    plt.plot(z_df["month"], z_df["global_total_layoffs"], label="Global layoffs (z-score)", linewidth=2.0)
    plt.plot(z_df["month"], z_df["unemployment_rate"], label="US unemployment (z-score)", linewidth=2.0)
    plt.plot(z_df["month"], z_df["openings_per_unemployed"], label="Openings per unemployed (z-score)", linewidth=2.0)
    plt.title("Layoffs and U.S. Labor Conditions Over Time")
    plt.xlabel("Month")
    plt.ylabel("Standardized value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "us_labor_vs_layoffs_timeseries.png", dpi=220)
    plt.close()

    scatter_with_trend(
        merged["unemployment_rate"],
        np.log1p(merged["global_total_layoffs"]),
        FIGURES_DIR / "us_unemployment_vs_global_layoffs.png",
        "Global Layoffs vs U.S. Unemployment Rate",
        "U.S. unemployment rate",
        "log(1 + global recorded layoffs)",
        "#2b6cb0",
    )

    return {"tests": tests}


def analyze_global_labor(layoffs: pd.DataFrame, global_labor: pd.DataFrame) -> dict[str, object]:
    # Data integration requirement: connect country-year layoffs to global labor indicators.
    annual_country = (
        layoffs.groupby(["country", "year"])
        .agg(total_layoffs=("layoff_count", "sum"), event_count=("company", "size"))
        .reset_index()
    )
    merged = global_labor.merge(
        annual_country,
        left_on=["country_name", "year"],
        right_on=["country", "year"],
        how="left",
    )
    merged["total_layoffs"] = merged["total_layoffs"].fillna(0)
    merged["event_count"] = merged["event_count"].fillna(0)
    merged.to_csv(TABLES_DIR / "global_labor_layoff_merge.csv", index=False)

    country_summary = (
        merged.groupby("country_name")
        .agg(
            avg_unemployment=("unemployment_rate_pct", "mean"),
            avg_youth_unemployment=("youth_unemployment_pct", "mean"),
            avg_employment_to_pop=("employment_to_pop_pct", "mean"),
            total_layoffs=("total_layoffs", "sum"),
            total_events=("event_count", "sum"),
        )
        .sort_values("total_layoffs", ascending=False)
        .round(2)
        .reset_index()
    )
    country_summary.to_csv(TABLES_DIR / "global_country_summary.csv", index=False)

    corr = merged[
        ["total_layoffs", "event_count", "unemployment_rate_pct", "youth_unemployment_pct", "employment_to_pop_pct"]
    ].corr(numeric_only=True).round(4)
    corr.to_csv(TABLES_DIR / "global_labor_correlation_matrix.csv")

    rho_unemp = stats.spearmanr(merged["total_layoffs"], merged["unemployment_rate_pct"], nan_policy="omit")
    rho_youth = stats.spearmanr(merged["total_layoffs"], merged["youth_unemployment_pct"], nan_policy="omit")
    rho_emp = stats.spearmanr(merged["total_layoffs"], merged["employment_to_pop_pct"], nan_policy="omit")
    regression = multiple_linear_regression(
        merged,
        "total_layoffs",
        ["unemployment_rate_pct", "youth_unemployment_pct", "employment_to_pop_pct"],
    )
    # Statistical methods requirement: global correlation tests and multivariable regression.
    tests = {
        "layoffs_vs_unemployment_rho": float(rho_unemp.statistic),
        "layoffs_vs_unemployment_p": float(rho_unemp.pvalue),
        "layoffs_vs_youth_unemployment_rho": float(rho_youth.statistic),
        "layoffs_vs_youth_unemployment_p": float(rho_youth.pvalue),
        "layoffs_vs_employment_to_pop_rho": float(rho_emp.statistic),
        "layoffs_vs_employment_to_pop_p": float(rho_emp.pvalue),
        **regression,
    }
    pd.DataFrame([tests]).round(6).to_csv(TABLES_DIR / "global_labor_statistical_tests.csv", index=False)

    scatter_with_trend(
        merged["unemployment_rate_pct"],
        np.log1p(merged["total_layoffs"]),
        FIGURES_DIR / "global_unemployment_vs_layoffs.png",
        "Country-Year Layoffs vs Unemployment Rate",
        "Unemployment rate (%)",
        "log(1 + total layoffs)",
        "#805ad5",
    )

    scatter_with_trend(
        merged["youth_unemployment_pct"],
        np.log1p(merged["total_layoffs"]),
        FIGURES_DIR / "global_youth_unemployment_vs_layoffs.png",
        "Country-Year Layoffs vs Youth Unemployment",
        "Youth unemployment rate (%)",
        "log(1 + total layoffs)",
        "#d53f8c",
    )

    return {"tests": tests}


def write_project_brief(
    layoffs_results: dict[str, object],
    news_results: dict[str, object],
    us_results: dict[str, object],
    global_results: dict[str, object],
) -> None:
    lines = [
        "# Project Analysis Brief",
        "",
        "This file is a submission helper, not the final report. It summarizes the finished code outputs and the strongest findings your team can cite when writing the report.",
        "",
        "## Research Questions",
    ]
    lines.extend(f"{idx}. {question}" for idx, question in enumerate(RESEARCH_QUESTIONS, start=1))

    lines.extend(
        [
            "",
            "## Main Findings",
            (
                f"- AI-company layoff counts differed from non-AI companies. "
                f"Welch p = {layoffs_results['tests']['layoff_count_welch_ttest_p']:.4f}, "
                f"Mann-Whitney p = {layoffs_results['tests']['layoff_count_mannwhitney_p']:.4f}."
            ),
            (
                f"- Percent workforce affected also differed by company type. "
                f"Welch p = {layoffs_results['tests']['pct_workforce_welch_ttest_p']:.4f}."
            ),
            (
                f"- The top industries showed significantly different layoff-count distributions. "
                f"Kruskal-Wallis p = {layoffs_results['tests']['industry_kruskal_p']:.6f}."
            ),
            (
                f"- Funding raised was positively associated with layoff size. "
                f"Spearman rho = {layoffs_results['tests']['funding_spearman_rho']:.4f}, "
                f"p = {layoffs_results['tests']['funding_spearman_p']:.6f}."
            ),
            (
                f"- Layoff-focused news sentiment differed from non-layoff coverage. "
                f"Welch p = {news_results['tests']['layoff_vs_nonlayoff_welch_p']:.4f}."
            ),
            (
                f"- Monthly layoffs aligned with U.S. labor stress indicators. "
                f"Layoffs vs unemployment rho = {us_results['tests']['global_layoffs_vs_unemployment_rho']:.4f}; "
                f"layoffs vs jobless claims rho = {us_results['tests']['global_layoffs_vs_claims_rho']:.4f}."
            ),
            (
                f"- Country-year layoffs were related to labor-market conditions globally. "
                f"Layoffs vs unemployment rho = {global_results['tests']['layoffs_vs_unemployment_rho']:.4f}; "
                f"layoffs vs youth unemployment rho = {global_results['tests']['layoffs_vs_youth_unemployment_rho']:.4f}."
            ),
            "",
            "## Output Structure",
            "- `cleaned_data/`: cleaned CSV files used by the analysis.",
            "- `tables/`: summary tables, merged datasets, and statistical test outputs.",
            "- `figures/`: report-ready PNG charts.",
            "- `notes/`: research questions and this project brief.",
            "",
            "## Suggested Report Sections to Match the Code",
            "- Dataset overview and cleaning decisions",
            "- Descriptive statistics",
            "- AI vs non-AI company comparison",
            "- Industry, country, and funding analysis",
            "- News sentiment analysis",
            "- U.S. labor indicator analysis",
            "- Global labor indicator analysis",
            "- Final conclusions and limitations",
        ]
    )

    with (TEXT_DIR / "project_brief.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def write_methods_summary() -> None:
    lines = [
        "# Methods Summary",
        "",
        "This document summarizes the statistical methods implemented in `final_project_analysis.py`.",
        "",
        "## Data Cleaning",
        "- Parsed all date columns into pandas datetime format.",
        "- Converted `pct_workforce` from percentage strings into numeric percentages.",
        "- Converted `raised_mm` from currency-formatted strings into numeric millions of USD.",
        "- Standardized truncated country and industry names where needed.",
        "- Exported cleaned versions of every dataset to `final_submission/outputs/cleaned_data/`.",
        "",
        "## Descriptive Analysis",
        "- Overall dataset summaries.",
        "- Grouped summaries for AI vs non-AI companies.",
        "- Monthly, country-level, industry-level, and stage-level layoff aggregation.",
        "- Country-year merge between layoff events and global labor indicators.",
        "",
        "## Inferential Statistics",
        "- Welch two-sample t-tests for unequal-variance mean comparisons.",
        "- Mann-Whitney U tests for nonparametric two-group comparisons.",
        "- Bootstrap confidence intervals for AI vs non-AI mean differences.",
        "- Cohen's d effect-size estimates for two-group comparisons.",
        "- Kruskal-Wallis test for multi-industry layoff-count comparison.",
        "- Chi-square test for AI-company share across major industries.",
        "- Spearman correlations for monotonic relationships with skewed data.",
        "- Simple log-log regression for funding vs layoff size.",
        "- Multiple linear regression using `numpy.linalg.lstsq` for labor-indicator models.",
        "",
        "## Visualization",
        "- Time-series plots.",
        "- Boxplots.",
        "- Histograms.",
        "- Bar charts.",
        "- Scatter plots.",
        "",
        "## Reproducibility",
        "- All outputs are generated programmatically from the raw Kaggle dataset bundle.",
        "- The full pipeline can be rerun with `python3 final_project_analysis.py`.",
    ]
    with (TEXT_DIR / "methods_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def write_report_asset_map() -> None:
    lines = [
        "# Report Asset Map",
        "",
        "Use this as a guide when placing figures and tables into the final written report.",
        "",
        "## Research Question 1",
        "How did layoff activity evolve from 2020 to 2026, and how much of it came from AI companies?",
        "- Figure: `figures/layoffs_monthly_trend_ai_vs_non_ai.png`",
        "- Table: `tables/layoffs_monthly_summary.csv`",
        "",
        "## Research Question 2",
        "Do AI companies differ from non-AI companies in layoff size, workforce share affected, and funding profile?",
        "- Figure: `figures/layoffs_ai_vs_non_ai_boxplot.png`",
        "- Table: `tables/layoffs_ai_vs_non_ai_summary.csv`",
        "- Table: `tables/layoffs_statistical_tests.csv`",
        "",
        "## Research Question 3",
        "Which industries, countries, and company stages contributed the most recorded layoffs?",
        "- Figure: `figures/layoffs_top_industries.png`",
        "- Figure: `figures/layoffs_top_countries.png`",
        "- Table: `tables/layoffs_top_industries.csv`",
        "- Table: `tables/layoffs_top_countries.csv`",
        "- Table: `tables/layoffs_top_stages.csv`",
        "",
        "## Research Question 4",
        "Is funding associated with larger layoffs after accounting for the strong right-skew in layoff counts?",
        "- Figure: `figures/layoffs_funding_vs_size.png`",
        "- Table: `tables/layoffs_statistical_tests.csv`",
        "",
        "## Research Question 5",
        "How does layoff-focused news sentiment behave over time, and is it systematically different from non-layoff coverage?",
        "- Figure: `figures/news_monthly_sentiment_trend.png`",
        "- Figure: `figures/news_sentiment_category_counts.png`",
        "- Figure: `figures/news_average_sentiment_by_type.png`",
        "- Table: `tables/news_monthly_summary.csv`",
        "- Table: `tables/news_statistical_tests.csv`",
        "",
        "## Research Question 6",
        "How do monthly layoff totals align with U.S. labor indicators such as unemployment, job openings, and jobless claims?",
        "- Figure: `figures/us_labor_vs_layoffs_timeseries.png`",
        "- Figure: `figures/us_unemployment_vs_global_layoffs.png`",
        "- Table: `tables/us_labor_monthly_merged.csv`",
        "- Table: `tables/us_labor_correlation_matrix.csv`",
        "- Table: `tables/us_labor_statistical_tests.csv`",
        "",
        "## Research Question 7",
        "Across countries, how are layoff totals related to unemployment, youth unemployment, and employment-to-population ratios?",
        "- Figure: `figures/global_unemployment_vs_layoffs.png`",
        "- Figure: `figures/global_youth_unemployment_vs_layoffs.png`",
        "- Table: `tables/global_labor_layoff_merge.csv`",
        "- Table: `tables/global_country_summary.csv`",
        "- Table: `tables/global_labor_statistical_tests.csv`",
    ]
    with (TEXT_DIR / "report_asset_map.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def write_professor_facing_shortlist() -> None:
    lines = [
        "# Professor-Facing Shortlist",
        "",
        "This is the recommended subset of results to emphasize in the final report and presentation.",
        "",
        "## Best Figures to Include",
        "1. `figures/layoffs_monthly_trend_ai_vs_non_ai.png`",
        "Why: clearly shows the time trend and keeps the project centered on the main topic.",
        "2. `figures/layoffs_ai_vs_non_ai_boxplot.png`",
        "Why: visually supports the AI vs non-AI comparison and the hypothesis test results.",
        "3. `figures/layoffs_top_industries.png`",
        "Why: gives an intuitive view of where layoffs were concentrated.",
        "4. `figures/layoffs_funding_vs_size.png`",
        "Why: one of the strongest relationship plots in the project and statistically significant.",
        "5. `figures/news_average_sentiment_by_type.png`",
        "Why: adds an original supporting angle that helps the project stand out.",
        "6. `figures/us_labor_vs_layoffs_timeseries.png`",
        "Why: connects layoff activity to a broader labor-market context.",
        "7. `figures/global_unemployment_vs_layoffs.png`",
        "Why: useful as a global context figure, but lower priority than the six above.",
        "",
        "## Best Tables to Include",
        "1. `tables/layoffs_ai_vs_non_ai_summary.csv`",
        "2. `tables/layoffs_statistical_tests.csv`",
        "3. `tables/layoffs_top_industries.csv`",
        "4. `tables/news_statistical_tests.csv`",
        "5. `tables/us_labor_statistical_tests.csv`",
        "",
        "## Lower-Priority Assets",
        "- `global_labor_statistical_tests.csv`: useful if you want to mention that some global relationships were weak or not statistically strong.",
        "- `global_youth_unemployment_vs_layoffs.png`: acceptable, but usually not as strong as the unemployment figure.",
        "- Full correlation matrices: best used in an appendix rather than the main body.",
        "",
        "## Suggested Main Storyline",
        "1. Start with the growth and timing of layoff activity.",
        "2. Compare AI and non-AI companies.",
        "3. Show which industries were hit most.",
        "4. Show the funding relationship.",
        "5. Add one standout supporting angle: news sentiment or U.S. labor context.",
        "6. Mention global labor findings briefly as supporting context or limitations.",
    ]
    with (TEXT_DIR / "professor_facing_shortlist.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def write_manifest() -> None:
    manifest = {
        "main_script": "final_project_analysis.py",
        "primary_dataset": str(LAYOFFS_PATH.name),
        "supporting_datasets": [NEWS_PATH.name, US_LABOR_PATH.name, GLOBAL_LABOR_PATH.name],
        "outputs_root": str(OUTPUT_DIR.name),
        "figures_dir": str(FIGURES_DIR.name),
        "tables_dir": str(TABLES_DIR.name),
        "cleaned_data_dir": str(CLEANED_DIR.name),
        "notes_dir": str(TEXT_DIR.name),
    }
    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def write_run_gallery() -> Path:
    figure_files = sorted(FIGURES_DIR.glob("*.png"))
    table_files = sorted(TABLES_DIR.glob("*.csv"))
    note_files = sorted(TEXT_DIR.iterdir()) if TEXT_DIR.exists() else []
    gallery_path = OUTPUT_DIR / "run_gallery.html"
    figure_cards = "\n".join(
        [
            (
                f"<section class='card'><h2>{figure.name}</h2>"
                f"<img src='figures/{figure.name}' alt='{figure.stem}' />"
                f"<p><a href='figures/{figure.name}'>Open image</a></p></section>"
            )
            for figure in figure_files
        ]
    )
    table_links = "\n".join([f"<li><a href='tables/{table.name}'>{table.name}</a></li>" for table in table_files])
    note_links = "\n".join([f"<li><a href='notes/{note.name}'>{note.name}</a></li>" for note in note_files if note.is_file()])
    gallery_path.write_text(
        "\n".join(
            [
                "<!DOCTYPE html>",
                "<html lang='en'>",
                "<head>",
                "<meta charset='utf-8' />",
                "<meta name='viewport' content='width=device-width, initial-scale=1' />",
                "<title>Final Project Output Gallery</title>",
                "<style>",
                "body { font-family: Helvetica, Arial, sans-serif; margin: 0; background: #f7fafc; color: #1a202c; }",
                "header { padding: 24px 32px 8px; }",
                "main { padding: 0 32px 32px; }",
                ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }",
                ".card { background: white; border-radius: 14px; padding: 16px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); }",
                "img { width: 100%; height: auto; border-radius: 10px; background: #edf2f7; }",
                "ul { background: white; border-radius: 14px; padding: 18px 24px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); }",
                "a { color: #2b6cb0; text-decoration: none; }",
                "</style>",
                "</head>",
                "<body>",
                "<header>",
                "<h1>Final Project Output Gallery</h1>",
                f"<p>Outputs saved in: {OUTPUT_DIR}</p>",
                "</header>",
                "<main>",
                "<h2>Figures</h2>",
                f"<div class='grid'>{figure_cards}</div>",
                "<h2>Tables</h2>",
                f"<ul>{table_links}</ul>",
                "<h2>Notes</h2>",
                f"<ul>{note_links}</ul>",
                "</main>",
                "</body>",
                "</html>",
            ]
        ),
        encoding="utf-8",
    )
    return gallery_path


def open_run_gallery(gallery_path: Path) -> None:
    if os.environ.get("FINAL_PROJECT_OPEN_OUTPUTS", "1") == "0":
        return
    try:
        if os.uname().sysname == "Darwin":
            for browser_name in ["Safari", "Google Chrome"]:
                result = subprocess.run(
                    ["open", "-a", browser_name, str(gallery_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    return
            subprocess.run(
                ["open", str(FIGURES_DIR)],
                check=False,
                capture_output=True,
                text=True,
            )
        else:
            webbrowser.open(gallery_path.as_uri())
    except OSError:
        pass


def print_output_summary(gallery_path: Path) -> None:
    print(f"Submission-ready outputs saved in {OUTPUT_DIR}")
    print(f"Run gallery available at {gallery_path}")
    for figure_path in sorted(FIGURES_DIR.glob("*.png")):
        print(f"Figure: {figure_path}")


def build_final_submission_folder() -> None:
    submission_dir = PROJECT_ROOT / "final_submission"
    submission_code = submission_dir / "code"
    submission_data = submission_dir / "data"
    submission_outputs = submission_dir / "outputs"
    submission_highlights = submission_dir / "highlights"

    for directory in [submission_dir, submission_code, submission_data, submission_outputs, submission_highlights]:
        directory.mkdir(parents=True, exist_ok=True)

    files_to_copy = {
        PROJECT_ROOT / "final_project_analysis.py": submission_code / "final_project_analysis.py",
        PROJECT_ROOT / "EmploymentAnalysis.py": submission_code / "EmploymentAnalysis.py",
        PROJECT_ROOT / "main.py": submission_code / "main.py",
        PROJECT_ROOT / "README.md": submission_dir / "README.md",
        PROJECT_ROOT / "requirements.txt": submission_dir / "requirements.txt",
        LAYOFFS_PATH: submission_data / "layoffs_events.csv",
        NEWS_PATH: submission_data / "news_sentiment.csv",
        US_LABOR_PATH: submission_data / "us_labor_indicators.csv",
        GLOBAL_LABOR_PATH: submission_data / "global_labor_indicators.csv",
    }

    for src, dst in files_to_copy.items():
        dst.write_bytes(src.read_bytes())

    if OUTPUT_DIR != submission_outputs:
        for subfolder in ["cleaned_data", "figures", "tables", "notes"]:
            source_dir = OUTPUT_DIR / subfolder
            target_dir = submission_outputs / subfolder
            target_dir.mkdir(exist_ok=True)
            for item in source_dir.iterdir():
                if item.is_file():
                    (target_dir / item.name).write_bytes(item.read_bytes())

        manifest_target = submission_outputs / "manifest.json"
        manifest_target.write_bytes((OUTPUT_DIR / "manifest.json").read_bytes())

    highlight_items = [
        OUTPUT_DIR / "figures" / "layoffs_monthly_trend_ai_vs_non_ai.png",
        OUTPUT_DIR / "figures" / "layoffs_ai_vs_non_ai_boxplot.png",
        OUTPUT_DIR / "figures" / "layoffs_top_industries.png",
        OUTPUT_DIR / "figures" / "layoffs_funding_vs_size.png",
        OUTPUT_DIR / "figures" / "news_average_sentiment_by_type.png",
        OUTPUT_DIR / "figures" / "us_labor_vs_layoffs_timeseries.png",
        OUTPUT_DIR / "tables" / "layoffs_ai_vs_non_ai_summary.csv",
        OUTPUT_DIR / "tables" / "layoffs_statistical_tests.csv",
        OUTPUT_DIR / "tables" / "layoffs_top_industries.csv",
        OUTPUT_DIR / "tables" / "news_statistical_tests.csv",
        OUTPUT_DIR / "tables" / "us_labor_statistical_tests.csv",
        OUTPUT_DIR / "notes" / "professor_facing_shortlist.md",
    ]
    for item in highlight_items:
        if item.exists():
            (submission_highlights / item.name).write_bytes(item.read_bytes())

    submission_readme = submission_dir / "SUBMISSION_GUIDE.md"
    submission_readme.write_text(
        "\n".join(
            [
                "# Submission Guide",
                "",
                "## Run Command",
                "`python3 code/final_project_analysis.py`",
                "",
                "## Folder Contents",
                "- `code/`: final Python scripts",
                "- `data/`: original dataset files used by the project",
                "- `outputs/cleaned_data/`: cleaned data produced programmatically",
                "- `outputs/figures/`: figures for the final report",
                "- `outputs/tables/`: summary and statistical tables",
                "- `outputs/notes/`: methods summary, project brief, and report asset map",
                "- `highlights/`: the best professor-facing figures, tables, and shortlist notes",
                "",
                "## Recommended Files to Submit",
                "- everything inside this `final_submission/` folder",
                "- plus your team's final written report and presentation when ready",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    write_research_questions()
    raw_layoffs, layoffs = load_layoffs()
    news = load_news()
    us_labor = load_us_labor()
    global_labor = load_global_labor()

    save_data_quality(raw_layoffs, layoffs, news, us_labor, global_labor)
    layoffs_results = analyze_layoffs(layoffs)
    news_results = analyze_news(news)
    us_results = analyze_us_labor(layoffs, us_labor)
    global_results = analyze_global_labor(layoffs, global_labor)
    write_project_brief(layoffs_results, news_results, us_results, global_results)
    write_methods_summary()
    write_report_asset_map()
    write_professor_facing_shortlist()
    write_manifest()
    build_final_submission_folder()
    gallery_path = write_run_gallery()
    open_run_gallery(gallery_path)
    print_output_summary(gallery_path)


if __name__ == "__main__":
    main()
