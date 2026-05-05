# DS-A Final Project Analysis Package

This repository now includes a submission-ready analysis pipeline for the Kaggle dataset bundle:

- `archive/layoffs_events.csv`
- `archive/news_sentiment.csv`
- `archive/us_labor_indicators.csv`
- `archive/global_labor_indicators.csv`

## Main Script

Run:

```bash
python3 final_project_analysis.py
```

## What It Produces

The script writes all final code outputs to `submission_outputs/`:

- `cleaned_data/`: cleaned versions of every dataset used in the project
- `tables/`: descriptive statistics, merged datasets, and statistical test results
- `figures/`: PNG figures ready to place into the final report
- `notes/`: research questions and a project brief to help with report writing
- `manifest.json`: simple index of the generated package

## Analysis Scope

The pipeline covers:

1. Layoff trends over time from 2020 to 2026
2. AI vs non-AI company comparisons
3. Industry, country, and company-stage comparisons
4. Funding vs layoff-size analysis
5. News sentiment analysis
6. U.S. labor indicator analysis
7. Global labor indicator analysis

## Dependencies

Install the Python packages in `requirements.txt` if needed.
