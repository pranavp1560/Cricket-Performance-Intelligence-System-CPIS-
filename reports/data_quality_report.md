# Data Quality and Cleaning Report

**Generated on:** 2026-06-08 16:31:05

This report summarizes the data cleaning, standardization, and integrity validation performed on the raw match and ball-by-ball datasets.

## 1. Summary Statistics

| Dataset | Initial Rows | Cleaned Rows | Duplicates Removed | Missing Values Imputed |
| :--- | :---: | :---: | :---: | :---: |
| **Matches** | 302 | 300 | 2 | 2 |
| **Deliveries** | 63980 | 63962 | 18 | 0 (Orphaned) |

## 2. Data Cleaning Actions

### Matches Table
*   **Duplicate Removal**: Identified and dropped `2` duplicate match records based on `match_id` to guarantee primary key uniqueness.
*   **Missing Value Imputation**: Handled `2` matches with empty/null winners (e.g., weather washouts) by imputing the value `"No Result"` and set the winning margins to `0`.
*   **Schema Consistency**: Cast `match_id`, `season`, `is_playoff`, and `is_final` to integer types and parsed `date` columns to date objects.

### Deliveries Table
*   **Duplicate Removal**: Removed `18` duplicate ball-by-ball delivery entries.
*   **Name Standardization**: Standardized `266` player names to maintain historical consistency across various seasons.
    *   *Examples:* Mapped `"V. Kohli"`, `"V Kohli"` -> `"Virat Kohli"`, and `"J Bumrah"` -> `"Jasprit Bumrah"`.
*   **Wicket Fielding Null Imputation**: Replaced all missing value (`NaN`) entries in columns `dismissal_kind`, `player_dismissed`, and `fielder` with empty strings (`""`) to represent non-dismissal balls, enabling clean SQL joins and group-by aggregates.
*   **Chase Context Validation**: Filled null values in `runs_required`, `balls_remaining`, and `wickets_remaining` with default values (`0` runs, `120` balls, `10` wickets) for first innings.

## 3. Database Schema Verification

Cleaned data has been successfully ingested into the SQLite Database: **cricket_intelligence.db**.

Tables created:
*   `player_match_features`
*   `venue_classification`
*   `player_venue_stats`
*   `player_venue_insights`
*   `player_overall_metrics`
*   `player_predictions`
*   `matches`
*   `deliveries`

### Matches Schema Integrity
*   `match_id` (Primary Key, INTEGER)
*   `season` (INTEGER)
*   `date` (TEXT/DATE)
*   `venue` (TEXT)
*   `team1` / `team2` (TEXT)
*   `toss_winner` / `toss_decision` (TEXT)
*   `winner` (TEXT)
*   `win_by_runs` / `win_by_wickets` / `margin` (INTEGER)
*   `is_playoff` / `is_final` (INTEGER)

### Deliveries Schema Integrity
*   `match_id` (Foreign Key referencing `matches.match_id`, INTEGER)
*   `inning` (INTEGER)
*   `over` (INTEGER)
*   `ball` (INTEGER)
*   `batting_team` / `bowling_team` (TEXT)
*   `batsman` / `non_striker` / `bowler` (TEXT)
*   `batsman_runs` / `extra_runs` / `total_runs` (INTEGER)
*   `is_wicket` (INTEGER)
*   `dismissal_kind` / `player_dismissed` / `fielder` (TEXT)
*   `is_chase` (INTEGER)
*   `runs_required` / `balls_remaining` / `wickets_remaining` (INTEGER)
*   `is_super_over` (INTEGER)

**Report Status:** ✅ **PASSED**
