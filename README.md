# Cricket Performance Intelligence System (CPIS)

CPIS is an industry-grade, data-driven cricket analytics platform that evaluates, ranks, and predicts player performance using custom rating algorithms, situational impact modeling, linear programming optimization, and machine learning.

Traditional cricket statistics (like batting average, strike rate, and economy) fail to capture a player's actual impact on match outcomes because they ignore situational context (e.g., scoring runs in a high-pressure chase, bowling in the death overs, or playing on spin-friendly vs. batting-friendly pitches). CPIS resolves this by engineering advanced context-aware features and calculating proprietary performance indexes.

---

## 1. System Architecture

The modular system pipeline operates as follows:

```
[ Raw Data Ingestor ] -> Runs high-fidelity ball-by-ball simulator generating 5 seasons of T20 league matches
           │
           ▼
[ ETL Cleaner & DB Ingestion ] -> Standarizes player names, duplicates, nulls, and populates SQLite
           │
           ▼
[ Analytics & Venue Engine ] -> Calculates MIS, Consistency, PPI, and Venue difficulty weights
           │
           ├─► [ ML Predictor ] (Random Forest, XGBoost, LightGBM) -> Forecasts next-season ratings
           ├─► [ MILP Optimizer ] (PuLP Linear Programming) -> Solves playing XI under role constraints
           └─► [ PDF & Excel Exporter ] -> Compiles and saves analytical report artifacts
           │
           ▼
[ Streamlit Web Dashboard ] -> Interactive dark-navy UI with Plotly charts & optimizer widgets
```

---

## 2. Advanced Mathematical Formulation

### A. Match Impact Score (MIS)
Measures a player's context-adjusted contribution to a match's outcome, scaled from $0$ to $100$.

$$\text{MIS} = w_1 \cdot \text{Batting Impact} + w_2 \cdot \text{Bowling Impact} + w_3 \cdot \text{Fielding Impact}$$

Where:
*   **Batting Impact** (calculated per match):
    $$\text{Batting Impact} = \text{Runs} \cdot \left(1 + 0.02 \cdot \max(0, \text{SR} - 120)\right) + 1.5 \cdot \text{Boundaries} + 2.0 \cdot \text{Pressure Runs} \cdot \text{Chase Difficulty}$$
    *   *Boundaries:* Count of fours and sixes.
    *   *Pressure Runs:* Runs scored in overs 16–20 or during a high-rate chase (Required Run Rate $> 9.0$ RPO).
*   **Bowling Impact** (calculated per match):
    $$\text{Bowling Impact} = \text{Wickets} \cdot 25 + (8.5 - \text{Economy}) \cdot 5 \cdot \text{Overs} + \text{Death Wickets} \cdot 15 + \text{Dot Balls} \cdot 1.5$$
    *   *Death Wickets:* Wickets taken in overs 16–20.
*   **Fielding Impact** (calculated per match):
    $$\text{Fielding Impact} = \text{Catches} \cdot 10 + \text{Stumpings} \cdot 15 + \text{Run Outs} \cdot 15$$
*   **Context Adjustments:** Final match impact scores are scaled up for playoffs ($1.2\times$) and final matches ($1.5\times$).

### B. Consistency Index (CI)
Measures player stability and reliability across matches:

$$\text{CI} = 100 \cdot \left(1 - \min\left(1, \frac{\text{Coefficient of Variation (CV)}}{\text{CV}_{\text{threshold}}}\right)\right) \cdot \text{Drop-off Factor}$$

*   $\text{CV} = \frac{\sigma}{\mu}$ (Standard Deviation of Match Impact Scores / Mean of Match Impact Scores)
*   $\text{CV}_{\text{threshold}} = 1.2$
*   **Drop-off Factor:** Penalizes players with fewer than 10 matches to ensure consistency ratings are only awarded to regular players.

### C. Pressure Performance Index (PPI)
Evaluates performance under pressure:

$$\text{PPI} = 0.4 \cdot \text{Chase Success} + 0.3 \cdot \text{Death Overs Perf} + 0.2 \cdot \text{Playoffs Impact} + 0.1 \cdot \text{Super Over Impact}$$

*   **Chase Success:** Ratio of runs scored in successful runs chases to overall runs.
*   **Death Overs Perf:** Economy rate and wicket-taking frequency (bowlers) or strike rate (batters) in overs 16–20.

### D. CPIS Player Rating
Composite player rating ($0\text{--}100$) used to rank players:

$$\text{CPIS Rating} = 0.35 \cdot \text{MIS} + 0.20 \cdot \text{Consistency Index} + 0.20 \cdot \text{PPI} + 0.15 \cdot \text{Batting Raw} + 0.10 \cdot \text{Bowling Raw}$$

---

## 3. Best Playing XI Optimizer (MILP)

The squad selection is modeled as a Mixed-Integer Linear Programming (MILP) problem solved using `PuLP`.

### Objective Function
$$\text{Maximize } \sum_{i \in \text{Players}} \text{Rating}_i \cdot x_i$$
Where $x_i \in \{0, 1\}$ represents the selection status of player $i$.

### Positional Constraints
*   **Team Size:** $\sum_{i} x_i = 11$
*   **Openers:** $2 \le \sum_{i \in \text{Openers}} x_i \le 2$
*   **Wicketkeeper:** $1 \le \sum_{i \in \text{WKs}} x_i \le 2$
*   **Middle Order/Finishers:** $3 \le \sum_{i \in \text{Middle}} x_i \le 5$
*   **All-Rounders:** $1 \le \sum_{i \in \text{ARs}} x_i \le 3$
*   **Bowlers:** $3 \le \sum_{i \in \text{Bowlers}} x_i \le 5$

---

## 4. Machine Learning & Forecasting

CPIS implements a predictive pipeline utilizing **Random Forest**, **XGBoost**, and **LightGBM** to forecast player stats for the upcoming season (2026).
*   **Features:** Historical stats in season $S$ (Runs, Strike Rate, Wickets, Economy, MIS, CI, PPI, CPIS Rating).
*   **Targets:** Stats in season $S+1$.
*   The system evaluates all three algorithms and automatically deploys the best-performing model based on the test set $R^2$ score.

---

## 5. Directory Structure

```
Cricket_Performance_Intelligence_System/
│
├── data/
│   ├── raw/                      # Raw simulated matches & deliveries CSVs
│   └── processed/                # Clean CSVs and SQLite Database
│
├── notebooks/                    # EDA, Feature Engineering & Player Ranking
│
├── sql/
│   └── queries.sql               # Advanced analytical database queries
│
├── src/
│   ├── ingestion/                # Data collection & simulation pipeline
│   ├── cleaning/                 # Missing value handling & quality reporting
│   ├── analytics/                # Custom metric engines (MIS, CI, PPI, Venue)
│   ├── ranking/                  # Player rating calculation & CSV ranking exports
│   ├── optimization/             # PuLP Playing XI optimizer solver
│   ├── predictive/               # Scikit-learn, XGBoost, LightGBM forecasting
│   └── reports/                  # Matplotlib & openpyxl report generator
│
├── dashboard/
│   └── app.py                    # Streamlit web application
│
├── models/                       # Directory containing trained ML pkl models
│
├── tests/                        # Automated unit tests (pytest)
│
├── requirements.txt              # Python packages dependencies
└── README.md                     # Documentation
```

---

## 6. Setup & Execution Guide

### Prerequisite Installation
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 1: Run Ingestion & High-Fidelity Simulation
Generates 5 seasons of T20 league matches (~300 matches, ~60k deliveries) containing real player names, playoff flags, and situational variables:
```bash
python src/ingestion/data_ingestor.py
```

### Step 2: Clean & Load Database
Processes duplicates, standardizes player names, and populates the SQLite database. Generates a data quality report under `reports/data_quality_report.md`:
```bash
python src/cleaning/data_cleaner.py
```

### Step 3: Run Analytics Metrics Pipeline
Computes Match Impact Scores, Consistency Index, Pressure Index, and venue stats:
```bash
python src/analytics/metrics.py
python src/analytics/venue.py
```

### Step 4: Calculate CPIS Player Ratings & Top 100 Rankings
Ranks players and exports results to `data/processed/player_rankings.csv`:
```bash
python src/ranking/player_rater.py
```

### Step 5: Run Machine Learning Models
Trains predictive models and outputs 2026 season forecasts to SQLite:
```bash
python src/predictive/models.py
```

### Step 6: Generate Visual Reports
Compiles visual PDF reports and multi-sheet Excel workbooks under `reports/`:
```bash
python src/reports/report_generator.py
```

### Step 7: Launch Streamlit Dashboard App
Run the web application locally:
```bash
python -m streamlit run dashboard/app.py
```

---

## 7. Verification & Tests
Verify the installation by running the unit test suite:
```bash
python -m pytest tests/
```
All unit tests should pass with ✅ status, validating the database schema, mathematical metric boundaries ($0\text{--}100$), and linear programming optimization constraints.
