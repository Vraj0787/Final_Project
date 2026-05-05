# Project Analysis Brief

This file is a submission helper, not the final report. It summarizes the finished code outputs and the strongest findings your team can cite when writing the report.

## Research Questions
1. How did layoff activity evolve from 2020 to 2026, and how much of it came from AI companies?
2. Do AI companies differ from non-AI companies in layoff size, workforce share affected, and funding profile?
3. Which industries, countries, and company stages contributed the most recorded layoffs?
4. Is funding associated with larger layoffs after accounting for the strong right-skew in layoff counts?
5. How does layoff-focused news sentiment behave over time, and is it systematically different from non-layoff coverage?
6. How do monthly layoff totals align with U.S. labor indicators such as unemployment, job openings, and jobless claims?
7. Across countries, how are layoff totals related to unemployment, youth unemployment, and employment-to-population ratios?

## Main Findings
- AI-company layoff counts differed from non-AI companies. Welch p = 0.0247, Mann-Whitney p = 0.0744.
- Percent workforce affected also differed by company type. Welch p = 0.0316.
- The top industries showed significantly different layoff-count distributions. Kruskal-Wallis p = 0.000000.
- Funding raised was positively associated with layoff size. Spearman rho = 0.4357, p = 0.000000.
- Layoff-focused news sentiment differed from non-layoff coverage. Welch p = 0.0000.
- Monthly layoffs aligned with U.S. labor stress indicators. Layoffs vs unemployment rho = -0.3508; layoffs vs jobless claims rho = -0.3939.
- Country-year layoffs were related to labor-market conditions globally. Layoffs vs unemployment rho = -0.0102; layoffs vs youth unemployment rho = 0.0080.

## Output Structure
- `cleaned_data/`: cleaned CSV files used by the analysis.
- `tables/`: summary tables, merged datasets, and statistical test outputs.
- `figures/`: report-ready PNG charts.
- `notes/`: research questions and this project brief.

## Suggested Report Sections to Match the Code
- Dataset overview and cleaning decisions
- Descriptive statistics
- AI vs non-AI company comparison
- Industry, country, and funding analysis
- News sentiment analysis
- U.S. labor indicator analysis
- Global labor indicator analysis
- Final conclusions and limitations