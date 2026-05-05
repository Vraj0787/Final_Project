# Global AI Layoffs and Job Market Pulse Dataset

**Last updated:** 2026-04-18  |  Auto-refreshed on every notebook run

## Overview
Multi-source dataset covering 50+ countries - combining global tech/AI layoff events,
US Federal Reserve labor indicators (FRED), World Bank global labor data for 19 countries,
and a seeded + live-appended NewsAPI global headline sentiment archive.

## Output Files

| File | Rows | Description |
|------|------|-------------|
| layoffs_events.csv | 2470 | Global company-level layoff events (historical CSV + Playwright live scrape) |
| us_labor_indicators.csv | 392 | FRED: US unemployment, JOLTS, jobless claims, tech employment (weekly/monthly) |
| global_labor_indicators.csv | 114 | World Bank: unemployment, youth unemployment, employment ratio for 19 countries (annual) |
| news_sentiment.csv | 306 | Full global headline archive (seed + live-appended) with VADER sentiment scores |

---

## Column Reference

### layoffs_events.csv
| Column | Type | Description |
|--------|------|-------------|
| company | str | Company name |
| location | str | HQ city or region |
| layoff_count | int | Number of employees laid off |
| date | date | Parsed layoff date (YYYY-MM-DD) |
| pct_workforce | float | Percentage of total workforce affected (0–100) |
| industry | str | Industry sector (e.g. Retail, SaaS, AI) |
| source_url | str | URL of the source news article |
| stage | str | Funding stage at time of layoff (e.g. Series B, Public) |
| raised_mm | float | Total funding raised in millions USD |
| country | str | Country of company HQ |
| is_ai_company | bool | True if company is AI / ML / data-focused |

### us_labor_indicators.csv
| Column | Type | Description |
|--------|------|-------------|
| date | date | Observation date (weekly or monthly depending on series) |
| unemployment_rate | float | US civilian unemployment rate, % (FRED: UNRATE) |
| jolts_job_openings_k | float | JOLTS job openings, thousands (FRED: JTSJOL) |
| initial_jobless_claims_k | float | Initial jobless claims, thousands (FRED: ICSA) |
| information_sector_emp_k | float | Information sector employment, thousands (FRED: USINFO) |
| computer_math_emp_k | float | Computer and math occupations, thousands (FRED: LNS12032194) |
| financial_emp_k | float | Financial activities employment, thousands (FRED: USFIRE) |
| openings_per_unemployed | float | JOLTS openings divided by unemployment rate -- derived labor tightness ratio |
| tech_emp_yoy_pct | float | Information-sector employment year-over-year change, % -- derived |
| claims_4w_avg | float | 4-week rolling average of initial jobless claims -- derived |

### global_labor_indicators.csv
| Column | Type | Description |
|--------|------|-------------|
| year | int | Calendar year of observation |
| country_code | str | ISO 2-letter country code (e.g. US, IN, DE) |
| country_name | str | Full country name |
| unemployment_rate_pct | float | Unemployment rate, % of total labour force -- World Bank: SL.UEM.TOTL.ZS |
| youth_unemployment_pct | float | Youth unemployment rate, % ages 15-24 -- World Bank: SL.UEM.1524.ZS |
| employment_to_pop_pct | float | Employment-to-population ratio, % -- World Bank: SL.EMP.TOTL.SP.ZS |

Countries covered: USA, India, UK, Germany, Canada, Australia, Singapore, Brazil,
France, Netherlands, Sweden, Japan, China, South Korea, Israel, South Africa,
Nigeria, Indonesia, Mexico

### news_sentiment.csv
| Column | Type | Description |
|--------|------|-------------|
| date | date | Publication date normalised to midnight UTC |
| published_at | datetime | Raw UTC timestamp from NewsAPI |
| source | str | News outlet name (e.g. Reuters, Bloomberg) |
| title | str | Article headline |
| description | str | Article lede or short description |
| url | str | Direct link to the article |
| sentiment | float | VADER compound score -- range -1.0 to 1.0; negative below -0.05, positive above 0.05 |
| sentiment_cat | str | Bucketed label: negative / neutral / positive |
| is_layoff_news | bool | True if title + description match layoff-related keywords |
| is_hiring_news | bool | True if title + description match hiring-related keywords |

---

## Sources
- Layoffs.fyi: historical CSV (4,000+ events 2020-present) + Playwright live scrape on every run
- FRED API: Federal Reserve Bank of St. Louis -- 6 US labor series (free key required)
- World Bank API: api.worldbank.org -- global labor indicators, no API key required
- NewsAPI: newsapi.org -- global headlines, last 30 days per run; seed covers 2022-present

## Live Scraper Notes
layoffs.fyi is a JS-rendered React/Next.js site. The scraper uses Playwright (headless
Chromium) as its primary strategy -- it intercepts the Airtable API XHR calls the
React app fires on page load, giving us raw JSON before any rendering. Four strategies:

1. Playwright + Airtable/API intercept: captures exact JSON the React table uses
2. Playwright + rendered HTML: parses __NEXT_DATA__ or HTML table after full JS execution
3. requests + __NEXT_DATA__: fast, works on SSR/CDN-cached server responses
4. requests + HTML/inline scripts: last-resort pattern matching

If all strategies fail the notebook falls back to the historical CSV -- no data is lost.

## License
CC0 -- Public Domain
