# A/B Testing Analysis & Causal Inference — Landing Page Experiment
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![Dash](https://img.shields.io/badge/Dash-Dashboard-008DE4?logo=plotly&logoColor=white)](https://dash.plotly.com/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-Statistics-4B8BBE)](https://www.statsmodels.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/zhangluyuan/ab-testing)
> **Can a redesigned landing page increase conversions enough to justify a full launch?**  
> This project answers that question using a rigorous end-to-end experiment analysis combining frequentist hypothesis testing, Bayesian decision analysis, temporal segmentation, and business impact modeling — all presented through an interactive dashboard.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [1. Experiment Design & Sample Size Planning](#1-experiment-design--sample-size-planning)
  - [2. Data Loading & Integrity Validation](#2-data-loading--integrity-validation)
  - [3. Exploratory Data Analysis](#3-exploratory-data-analysis)
  - [4. Frequentist A/B Test](#4-frequentist-ab-test)
  - [5. Bayesian A/B Test](#5-bayesian-ab-test)
  - [6. Segmentation Analysis](#6-segmentation-analysis)
  - [7. Business Impact Quantification](#7-business-impact-quantification)
- [Dashboard Results](#dashboard-results)
- [Key Findings](#key-findings)
- [Final Recommendation](#final-recommendation)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Skills Demonstrated](#skills-demonstrated)

---

## Project Overview

This project is a production-quality A/B test analysis performed on the canonical **Udacity landing page experiment dataset**. It goes far beyond a simple p-value calculation — it applies the full decision-making framework used by data teams at companies like Netflix, Airbnb, Booking.com, and Google, including:

- Pre-experiment power analysis and sample size planning
- Data integrity validation before any inference
- Frequentist two-proportion z-testing with confidence intervals
- Bayesian posterior simulation with expected loss quantification
- Temporal segmentation and Simpson's Paradox checking
- Business impact modeling with break-even analysis
- A fully interactive Dash dashboard summarizing all findings

The notebook is written as an **executive-quality analysis report** — every statistical decision is tied back to a business consequence, and every chart has a plain-English interpretation designed for non-technical stakeholders.

---

## Business Problem

An e-commerce company redesigned its landing page and wants to know whether the new design increases the rate at which visitors complete a purchase (the **conversion rate**). The company ran a randomized controlled experiment: some users saw the existing page (control) and others saw the new design (treatment).

**The core decision:** should the company launch the new page to all users?

This is not just a statistical question — it is a capital allocation decision. A wrong answer in either direction has real costs:

- **Launching a bad design** degrades conversion, harms revenue, and wastes engineering capacity
- **Rejecting a good design** leaves growth on the table and delays competitive advantage

The goal of this analysis is to quantify the evidence clearly enough for leadership to make a rational, defensible decision.

---

## Dataset

| Property | Value |
|---|---|
| Source | [Udacity A/B Testing Dataset — Kaggle](https://www.kaggle.com/datasets/zhangluyuan/ab-testing) |
| File | `ab_data.csv` |
| Raw rows | 294,478 |
| Cleaned rows | 290,584 |
| Experiment window | Jan 02, 2017 → Jan 24, 2017 **(22 days)** |
| Columns | `user_id`, `timestamp`, `group`, `landing_page`, `converted` |

**Column descriptions:**

- `user_id` — unique visitor identifier (unit of experimentation)
- `timestamp` — when the user was exposed to the experiment
- `group` — assigned condition: `control` (old page) or `treatment` (new page)
- `landing_page` — actual page seen: `old_page` or `new_page`
- `converted` — binary outcome: `1` if the user converted, `0` if not

---

## Project Structure

```
ab-testing-causal-inference/
│
├── data/ab_data.csv                                      # Raw experiment dataset
├── ab_testing_causal_inference_report.ipynb         # Full analysis notebook
├── dashboard.py                                     # Interactive Dash dashboard                                                       
└── README.md                                        # This file
```

---

## Methodology

### 1. Experiment Design & Sample Size Planning

Before looking at any data, a proper experiment design requires defining statistical parameters upfront to avoid p-hacking and underpowered tests.

**Parameters chosen:**

| Parameter | Value | Reasoning |
|---|---|---|
| Baseline conversion rate | 12% | Consistent with dataset average |
| Minimum Detectable Effect (MDE) | 2% relative / **0.24% absolute** | Smallest lift worth acting on |
| Significance level (α) | 0.05 | 5% false positive tolerance — industry standard |
| Statistical power | 0.80 | 80% chance of detecting a real effect |

**Result:** The experiment required **228,644 users per group** (457,288 total) to be adequately powered for the chosen MDE. The actual cleaned sample of ~145,000 per group **falls short of this target** — meaning the experiment was underpowered.

> **Why this matters:** An underpowered experiment does not invalidate results, but a non-significant outcome could reflect insufficient data rather than a true null effect. This is disclosed explicitly throughout the analysis, and the Bayesian section provides a more nuanced picture by quantifying remaining uncertainty directly.

---

### 2. Data Loading & Integrity Validation

Data quality is not a cosmetic step in experimentation — it is part of causal validity. A randomized experiment is only interpretable if the group assignment matches the page actually served.

**Cleaning steps performed:**

| Step | Records affected | Reason |
|---|---|---|
| Exact duplicate rows | 0 found | No fully identical rows in raw data |
| Duplicate user IDs found | 3,894 | Users appearing more than once in the log |
| Mismatched assignment/exposure rows removed | **3,893** | Assigned to treatment but saw old page, or vice versa |
| Deduplication by user ID (keep first) | Remaining duplicates | One independent observation per user required |
| **Final cleaned dataset** | **290,584 rows** | Analysis-ready sample |

**Group balance after cleaning:**

| Group | Users | Conversions | Conversion Rate |
|---|---|---|---|
| Control | 145,274 | 17,489 | 12.0386% |
| Treatment | 145,310 | 17,264 | 11.8808% |
| Difference | **36 users (0.0124%)** | — | — |

Near-perfect balance confirms randomization worked correctly.

> **Why catching mismatches matters:** 3,893 rows had users assigned to one group but served the wrong page. Including them would corrupt the causal interpretation — you cannot cleanly attribute conversion differences to the page design if some "treatment" users never saw the treatment. This is the kind of check that separates rigorous experiment analysis from a notebook that simply runs functions on raw data.

---

### 3. Exploratory Data Analysis

Before formal testing, the analysis examines raw rates and temporal patterns to understand the data and detect anomalies.

**Experiment timing:**

| Property | Value |
|---|---|
| Start date | Jan 02, 2017 |
| End date | Jan 24, 2017 |
| Duration | **22 days** |
| Sample target met | **No** — 145,274 achieved vs 228,644 required |

**Daily trend analysis** showed that the two group lines cross each other repeatedly over 22 days, with neither consistently dominating. Both oscillate between approximately 11.5% and 12.5%. This pattern is consistent with sampling noise rather than a genuine treatment signal.

Crucially, there is **no novelty effect** — the treatment did not spike early and then decay back to baseline. It showed no advantage from day one, meaning even initial user curiosity was not triggered by the new design.

---

### 4. Frequentist A/B Test

A **one-sided two-proportion z-test** was used to formally test whether the treatment conversion rate exceeds the control.

**Hypotheses:**
- **H₀:** p_treatment ≤ p_control (new page does not improve conversion)
- **H₁:** p_treatment > p_control (new page improves conversion)

**Results:**

| Metric | Value | Interpretation |
|---|---|---|
| Control conversion rate | **12.0386%** | Baseline the new page must beat |
| Treatment conversion rate | **11.8808%** | New page actual performance |
| Observed absolute lift | **-0.1578%** | Treatment converted *less* than control |
| Observed relative lift | **-1.31%** | New page is 1.31% worse in relative terms |
| Z-statistic | **-1.3109** | Negative — treatment moves in the wrong direction |
| One-sided p-value | **0.9051** | Far above α=0.05 — no evidence against the null |
| 95% CI lower bound | **-0.3938%** | Could be losing up to 0.39% conversion |
| 95% CI upper bound | **+0.0781%** | Maximum plausible gain is only 0.08% |

**Confidence interval interpretation:** The entire interval sits almost entirely in negative territory. The data are far more consistent with the treatment hurting conversion than helping it. Zero barely sits at the far right edge of the interval — a positive lift is technically possible but extremely unlikely.

> **Common misconception clarified in the notebook:** A p-value of 0.9051 does NOT mean there is a 90% chance the null hypothesis is true. It means that IF the null were true, we would observe results at least this extreme 90.51% of the time. The p-value is a property of the testing procedure, not a direct probability about the hypothesis.

---

### 5. Bayesian A/B Test

The Bayesian analysis answers a more directly useful business question: **given everything observed, what is the probability the treatment is actually better, and what is the cost of making the wrong decision?**

**Setup:**
- Prior: **Beta(1, 1)** — weakly informative, treats all conversion rates as equally plausible
- Posterior simulation: **200,000 Monte Carlo draws** per variant
- Random seed: **42** (fully reproducible)

**Results:**

| Metric | Value | Interpretation |
|---|---|---|
| Posterior probability treatment wins | **9.59%** | 90.41% probability control is better |
| Expected loss if we launch treatment | **0.1637%** | Avg conversion lost if we launch and it is worse |
| Expected loss if we keep control | **0.0054%** | Avg conversion lost if we keep control and treatment was better |
| **Loss ratio** | **~30×** | Launching is 30 times riskier than keeping the old page |

**What the posterior chart shows:** The treatment posterior (red) sits almost entirely to the left of the control posterior (blue). The two distributions are clearly separated with minimal overlap — this is not an ambiguous, overlapping result. The data strongly and visually favor the control page.

> **The decisive business insight:** The expected loss asymmetry is the most powerful argument in the analysis. Launching the wrong page costs 0.1637% in conversion. Keeping the old page when the new one was actually better would only cost 0.0054%. That is a **30× difference in risk** — any rational expected-value framework says keep the old page.

---

### 6. Segmentation Analysis

Aggregate results can hide important heterogeneity. This section checks whether any user subgroup responds differently, and tests for **Simpson's Paradox** — where an overall trend reverses when data is broken into subgroups.

**Segmentation dimension:** Day of week (derived from `timestamp`)

**Full results from notebook — exact values:**

| Day | Control Rate | Treatment Rate | Winner | p-value | Significant at α=0.05? |
|---|---|---|---|---|---|
| Monday | 12.28% | 11.95% | Control | 0.8598 | ❌ No |
| **Tuesday** | **11.67%** | **12.23%** | **Treatment** | **0.0307** | **✅ Yes** |
| Wednesday | 12.18% | 11.88% | Control | 0.8203 | ❌ No |
| Thursday | 12.17% | 11.82% | Control | 0.8583 | ❌ No |
| Friday | 11.58% | 11.75% | Treatment | 0.2987 | ❌ No |
| Saturday | 12.46% | 11.71% | Control | 0.9891 | ❌ No |
| Sunday | 11.95% | 11.74% | Control | 0.7405 | ❌ No |

**Key observations:**

- Control outperforms treatment on **5 out of 7 days**
- Treatment outperforms control on **Tuesday** (significantly, p=0.0307) and **Friday** (not significantly, p=0.299)
- The notebook explicitly flags: *"1 segment shows a statistically significant treatment lift that deserves follow-up before a blanket launch decision"*
- **No full Simpson's Paradox** — the aggregate direction (control wins) is consistent with the majority of days

> **Important nuance on the Tuesday result:** With 7 simultaneous significance tests at α=0.05, we expect roughly 0.35 false positives by chance alone. The Tuesday result (p=0.0307) is significant but only barely, and it is a single day out of seven. This does not justify a full launch, but it does justify a focused **confirmatory experiment targeting Tuesday traffic** before dismissing the redesign entirely.

---

### 7. Business Impact Quantification

Statistical significance alone is not enough — every result must be translated into money to assess whether it is practically meaningful.

**Assumptions:**

| Assumption | Value |
|---|---|
| Monthly site visitors | 50,000 |
| Average order value | $35.00 |
| Redesign implementation cost | $20,000 |

**Results — exact values from notebook:**

| Metric | Value |
|---|---|
| Projected incremental orders / month | **-78.91 orders** |
| Projected monthly revenue impact | **-$2,761.92** |
| Projected annual revenue impact | **-$33,143.02** |
| Implementation cost (sunk) | **$20,000.00** |
| Break-even period | **Not reached** |

At the observed lift of -0.1578%, the new page would cost the business $2,762 per month and $33,143 per year in lost revenue — on top of the $20,000 already spent building it. The break-even point is never reached because the revenue impact is negative.

---

## Dashboard Results

The interactive Dash dashboard summarizes all findings in a single professional view built for both technical analysts and non-technical business stakeholders.

![Screenshot_19-3-2026_92826_127 0 0 1](https://github.com/user-attachments/assets/233c0c09-5259-4ee3-9a38-15c7b7e83588)


### Dashboard Components Explained

**① Verdict Banner — "DO NOT LAUNCH"**

The most prominent element. Displays the recommendation in large red text with a plain-English justification pulled directly from the Bayesian metrics:
> *"Treatment wins probability is only 9.6%. Launching is 30× riskier than keeping the control page. Expected loss if launched: 0.1637% vs 0.0054% if kept."*

The banner updates automatically — it turns green and shows "LAUNCH APPROVED" if a future dataset produces a win probability above 50%.

**② Six KPI Cards**

One card per critical metric, color-coded red for unfavorable values:
- **Control Rate: 12.039%** — the baseline in blue
- **Treatment Rate: 11.881%** — in red, below control
- **Absolute Lift: -0.158%** — red, moving in the wrong direction
- **Bayesian Win %: 9.6%** — red, far below the ≥95% threshold for confident launch
- **p-Value: 0.9051** — red, far above the <0.05 threshold
- **Loss Ratio: 30×** — amber, signaling extreme launch risk

**③ Daily Conversion Rate Over Time (Top Left)**

Line chart showing both groups daily across 22 days. The two lines cross repeatedly with no persistent gap — visual confirmation the result is noise, not signal. No novelty effect spike is visible at the start of the experiment.

**④ Bayesian Posterior Distributions (Top Right)**

The most decisive chart in the dashboard. The red treatment curve sits almost entirely to the left of the blue control curve. The clear separation between the two distributions visually represents the 90.41% probability that control is better. When posteriors look like this, the Bayesian evidence is conclusive.

**⑤ 95% Confidence Interval for Lift (Middle Left)**

The blue diamond point estimate sits clearly left of the amber zero reference line. The horizontal bar barely reaches zero on the right. The data are almost entirely consistent with the treatment being harmful, not helpful.

**⑥ Conversion Rate by Day of Week (Middle Right)**

Control (solid blue) sits above treatment (dashed red) on 5 of 7 days. Treatment edges control on Tuesday (the one significant segment) and Friday (not significant) — fully consistent with the notebook's segmentation output.

**⑦ Expected Loss by Decision (Bottom Left)**

The tall red bar (Launch Treatment = 0.1637%) dwarfs the tiny blue bar (Keep Control = 0.0054%). The visual makes the 30× risk asymmetry immediately legible without any arithmetic.

**⑧ Users per Experiment Group (Bottom Right)**

Confirms near-perfect traffic balance: 145,274 control vs 145,310 treatment. A 36-user gap (0.0124% imbalance) is negligible and consistent with proper randomization.

**⑨ Business Impact Summary Table**

All 16 metrics in a scrollable table. Rows highlighted in red flag unfavorable values. The bottom section — showing -$2,761.92 monthly, -$33,143.02 annual revenue impact, and "Not reached" for break-even — makes the financial case immediately clear to any stakeholder.

---

## Key Findings

1. **The treatment underperforms the control in aggregate.** Absolute lift is -0.1578%, meaning the new page converts 1.31% worse overall — it moves in the wrong direction.

2. **The frequentist test provides no evidence for launch.** p-value = 0.9051, with a 95% CI of [-0.3938%, +0.0781%] — almost entirely in negative territory.

3. **The Bayesian analysis is definitive.** Only 9.59% posterior probability that treatment beats control. Expected loss from launching is 30× higher than from keeping the old page.

4. **No novelty effect detected.** Treatment performed below control consistently from day one — no initial spike that later decayed.

5. **Tuesday is the one statistically significant segment** (p=0.0307, treatment +0.56pp over control). This deserves a focused confirmatory experiment, but does not justify a full launch across all users.

6. **Control wins on 5 of 7 days.** No Simpson's Paradox — aggregate and segment conclusions are largely consistent, with the Tuesday exception noted above.

7. **The financial impact is unambiguously negative.** Launching would cost ~$2,762/month and ~$33,143/year in lost revenue on top of $20,000 in build costs already spent. Break-even is never reached.

---

## Final Recommendation

> **⛔ DO NOT LAUNCH the new landing page based on current evidence.**

The analysis is consistent across every framework — frequentist, Bayesian, segmentation, and financial. The redesign does not improve overall conversion and carries a 30× higher expected cost if launched incorrectly.

**Recommended next steps:**

- **Keep the current page** as the production experience for all users
- **Run a focused Tuesday confirmatory experiment** — the one significant segment (p=0.0307) deserves a dedicated test before investing in a targeted weekday rollout
- **Instrument intermediate funnel metrics** — click-through, add-to-cart, checkout start, and page load time — to understand *where* the new design loses users
- **Commission a more ambitious redesign** — the current treatment may be too incremental; detecting a 0.24% absolute MDE requires a meaningfully differentiated design
- **Properly power the next experiment** — a follow-up test needs 228,644 users per group and should run for at least 2–3 full weekly cycles to capture day-of-week variation

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core analysis language |
| Pandas | Data loading, cleaning, aggregation |
| NumPy | Numerical operations, Bayesian simulation |
| Statsmodels | Power analysis, z-test, confidence intervals |
| Plotly | Interactive visualizations |
| Dash | Interactive web dashboard |
| Dash Bootstrap Components | Dashboard layout and styling |
| Jupyter Notebook | Analysis and reporting environment |

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Ayoub-Elkhaiari/A-B-Testing-Analysis-Causal-Inference-Landing-Page-Experiment.git
cd A-B-Testing-Analysis-Causal-Inference-Landing-Page-Experiment
```

**2. Install dependencies**
```bash
pip install pandas numpy statsmodels plotly dash dash-bootstrap-components jupyter
```

**3. Run the analysis notebook**
```bash
jupyter notebook ab_testing_causal_inference_report.ipynb
```
Run all cells top to bottom. The notebook is fully self-contained and reproducible.

**4. Run the interactive dashboard**
```bash
python dashboard.py
```
Then open `http://127.0.0.1:8050` in your browser.

> **Note:** Both the notebook and dashboard require `ab_data.csv` to be in the `data` folder.

---

## Skills Demonstrated

| Skill | Where applied |
|---|---|
| Experiment design & power analysis | Section 2 — sample size calculated before looking at data |
| Data integrity validation | Section 3 — 3,893 mismatch rows identified and removed |
| Frequentist hypothesis testing | Section 5 — two-proportion z-test, p-value, confidence interval |
| Bayesian inference | Section 6 — Beta-posterior simulation, expected loss, win probability |
| Multiple testing awareness | Section 7 — Tuesday significance interpreted in context of 7 simultaneous tests |
| Causal reasoning | Throughout — assignment vs exposure distinction, Simpson's Paradox check |
| Statistical communication | Every section — plain English interpretation for non-technical audiences |
| Business impact translation | Section 8 — lift converted to monthly/annual revenue and break-even |
| Interactive dashboard development | `dashboard.py` — full Dash app with 6 charts, KPI cards, and data table |
| Python (Pandas, NumPy, Statsmodels, Plotly, Dash) | All sections |

---

## References

- Kohavi, Tang, and Xu — *Trustworthy Online Controlled Experiments* (Cambridge University Press)
- Georgiev — *Statistical Methods in Online A/B Testing*
- Gelman et al. — *Bayesian Data Analysis* (3rd edition)
- [Statsmodels documentation](https://www.statsmodels.org)
- [Plotly documentation](https://plotly.com/python/)
- [Udacity A/B Testing Course](https://www.udacity.com/course/ab-testing--ud257)

---

*Analysis based on Udacity A/B Testing Dataset · 290,584 users · Jan 02–24, 2017 · 22-day experiment · Frequentist + Bayesian + Segmentation + Business Impact framework*
