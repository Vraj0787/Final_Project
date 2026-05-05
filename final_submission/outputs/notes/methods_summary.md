# Methods Summary

This document summarizes the statistical methods implemented in `final_project_analysis.py`.

## Data Cleaning
- Parsed all date columns into pandas datetime format.
- Converted `pct_workforce` from percentage strings into numeric percentages.
- Converted `raised_mm` from currency-formatted strings into numeric millions of USD.
- Standardized truncated country and industry names where needed.
- Exported cleaned versions of every dataset to `submission_outputs/cleaned_data/`.

## Descriptive Analysis
- Overall dataset summaries.
- Grouped summaries for AI vs non-AI companies.
- Monthly, country-level, industry-level, and stage-level layoff aggregation.
- Country-year merge between layoff events and global labor indicators.

## Inferential Statistics
- Welch two-sample t-tests for unequal-variance mean comparisons.
- Mann-Whitney U tests for nonparametric two-group comparisons.
- Bootstrap confidence intervals for AI vs non-AI mean differences.
- Cohen's d effect-size estimates for two-group comparisons.
- Kruskal-Wallis test for multi-industry layoff-count comparison.
- Chi-square test for AI-company share across major industries.
- Spearman correlations for monotonic relationships with skewed data.
- Simple log-log regression for funding vs layoff size.
- Multiple linear regression using `numpy.linalg.lstsq` for labor-indicator models.

## Visualization
- Time-series plots.
- Boxplots.
- Histograms.
- Bar charts.
- Scatter plots.

## Reproducibility
- All outputs are generated programmatically from the raw Kaggle dataset bundle.
- The full pipeline can be rerun with `python3 final_project_analysis.py`.