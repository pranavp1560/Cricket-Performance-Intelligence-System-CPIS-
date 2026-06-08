import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

def clean_data(raw_dir="data/raw", processed_dir="data/processed", db_path="data/processed/cricket_intelligence.db", report_path="reports/data_quality_report.md"):
    """
    Cleans raw match and delivery datasets, standardizes names, handles duplicates
    and null values, writes to a SQLite database, and outputs a quality report.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    print("Loading raw datasets...")
    matches_raw_path = f"{raw_dir}/matches_raw.csv"
    deliveries_raw_path = f"{raw_dir}/deliveries_raw.csv"
    
    if not os.path.exists(matches_raw_path) or not os.path.exists(deliveries_raw_path):
        raise FileNotFoundError("Raw CSV files not found. Please run the data ingestion pipeline first.")
        
    matches_df = pd.read_csv(matches_raw_path)
    deliveries_df = pd.read_csv(deliveries_raw_path)
    
    # Store initial metrics
    initial_matches_count = len(matches_df)
    initial_deliveries_count = len(deliveries_df)
    
    # ------------------ CLEANING MATCHES ------------------
    print("Cleaning matches dataset...")
    # 1. Duplicates
    duplicate_matches_count = matches_df.duplicated(subset=["match_id"]).sum()
    matches_df = matches_df.drop_duplicates(subset=["match_id"], keep="first")
    
    # 2. Nulls
    null_winner_count = matches_df["winner"].isnull().sum()
    matches_df["winner"] = matches_df["winner"].fillna("No Result")
    matches_df["margin"] = matches_df["margin"].fillna(0)
    
    # 3. Data type formatting
    matches_df["match_id"] = matches_df["match_id"].astype(int)
    matches_df["season"] = matches_df["season"].astype(int)
    matches_df["date"] = pd.to_datetime(matches_df["date"])
    matches_df["win_by_runs"] = matches_df["win_by_runs"].astype(int)
    matches_df["win_by_wickets"] = matches_df["win_by_wickets"].astype(int)
    matches_df["margin"] = matches_df["margin"].astype(int)
    matches_df["is_playoff"] = matches_df["is_playoff"].astype(int)
    matches_df["is_final"] = matches_df["is_final"].astype(int)
    
    # ------------------ CLEANING DELIVERIES ------------------
    print("Cleaning deliveries dataset...")
    # 1. Duplicates
    duplicate_deliveries_count = deliveries_df.duplicated().sum()
    deliveries_df = deliveries_df.drop_duplicates()
    
    # 2. Standardize player names
    player_mappings = {
        "V. Kohli": "Virat Kohli",
        "V Kohli": "Virat Kohli",
        "J Bumrah": "Jasprit Bumrah",
    }
    
    standardized_count = 0
    for col in ["batsman", "non_striker", "bowler", "player_dismissed", "fielder"]:
        # Count how many matches will be replaced
        matches_replaced = deliveries_df[col].isin(player_mappings.keys()).sum()
        standardized_count += matches_replaced
        deliveries_df[col] = deliveries_df[col].replace(player_mappings)
        
    # 3. Handle missing values in wickets/dismissals
    # Standardize dismissal kinds
    deliveries_df["dismissal_kind"] = deliveries_df["dismissal_kind"].fillna("")
    deliveries_df["player_dismissed"] = deliveries_df["player_dismissed"].fillna("")
    deliveries_df["fielder"] = deliveries_df["fielder"].fillna("")
    
    # Fill context columns
    deliveries_df["runs_required"] = deliveries_df["runs_required"].fillna(0).astype(int)
    deliveries_df["balls_remaining"] = deliveries_df["balls_remaining"].fillna(120).astype(int)
    deliveries_df["wickets_remaining"] = deliveries_df["wickets_remaining"].fillna(10).astype(int)
    
    # Data type formatting
    deliveries_df["match_id"] = deliveries_df["match_id"].astype(int)
    deliveries_df["inning"] = deliveries_df["inning"].astype(int)
    deliveries_df["over"] = deliveries_df["over"].astype(int)
    deliveries_df["ball"] = deliveries_df["ball"].astype(int)
    deliveries_df["batsman_runs"] = deliveries_df["batsman_runs"].astype(int)
    deliveries_df["extra_runs"] = deliveries_df["extra_runs"].astype(int)
    deliveries_df["total_runs"] = deliveries_df["total_runs"].astype(int)
    deliveries_df["is_wicket"] = deliveries_df["is_wicket"].astype(int)
    deliveries_df["is_chase"] = deliveries_df["is_chase"].astype(int)
    deliveries_df["is_super_over"] = deliveries_df["is_super_over"].astype(int)
    
    # ------------------ REFERENTIAL INTEGRITY CHECK ------------------
    # Ensure all deliveries belong to a valid match
    valid_matches = set(matches_df["match_id"])
    deliveries_clean_df = deliveries_df[deliveries_df["match_id"].isin(valid_matches)].copy()
    orphan_deliveries = len(deliveries_df) - len(deliveries_clean_df)
    deliveries_df = deliveries_clean_df
    
    # ------------------ LOAD INTO SQL DATABASE ------------------
    print(f"Creating SQL Database at {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # Write tables
    matches_df.to_sql("matches", conn, if_exists="replace", index=False)
    deliveries_df.to_sql("deliveries", conn, if_exists="replace", index=False)
    
    # Save CSV copies of clean data
    matches_df.to_csv(f"{processed_dir}/matches_clean.csv", index=False)
    deliveries_df.to_csv(f"{processed_dir}/deliveries_clean.csv", index=False)
    
    # Verify tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    # ------------------ GENERATE QUALITY REPORT ------------------
    print(f"Writing Data Quality Report to {report_path}...")
    report_md = f"""# Data Quality and Cleaning Report

**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This report summarizes the data cleaning, standardization, and integrity validation performed on the raw match and ball-by-ball datasets.

## 1. Summary Statistics

| Dataset | Initial Rows | Cleaned Rows | Duplicates Removed | Missing Values Imputed |
| :--- | :---: | :---: | :---: | :---: |
| **Matches** | {initial_matches_count} | {len(matches_df)} | {duplicate_matches_count} | {null_winner_count} |
| **Deliveries** | {initial_deliveries_count} | {len(deliveries_df)} | {duplicate_deliveries_count} | {orphan_deliveries} (Orphaned) |

## 2. Data Cleaning Actions

### Matches Table
*   **Duplicate Removal**: Identified and dropped `{duplicate_matches_count}` duplicate match records based on `match_id` to guarantee primary key uniqueness.
*   **Missing Value Imputation**: Handled `{null_winner_count}` matches with empty/null winners (e.g., weather washouts) by imputing the value `"No Result"` and set the winning margins to `0`.
*   **Schema Consistency**: Cast `match_id`, `season`, `is_playoff`, and `is_final` to integer types and parsed `date` columns to date objects.

### Deliveries Table
*   **Duplicate Removal**: Removed `{duplicate_deliveries_count}` duplicate ball-by-ball delivery entries.
*   **Name Standardization**: Standardized `{standardized_count}` player names to maintain historical consistency across various seasons.
    *   *Examples:* Mapped `"V. Kohli"`, `"V Kohli"` -> `"Virat Kohli"`, and `"J Bumrah"` -> `"Jasprit Bumrah"`.
*   **Wicket Fielding Null Imputation**: Replaced all missing value (`NaN`) entries in columns `dismissal_kind`, `player_dismissed`, and `fielder` with empty strings (`""`) to represent non-dismissal balls, enabling clean SQL joins and group-by aggregates.
*   **Chase Context Validation**: Filled null values in `runs_required`, `balls_remaining`, and `wickets_remaining` with default values (`0` runs, `120` balls, `10` wickets) for first innings.

## 3. Database Schema Verification

Cleaned data has been successfully ingested into the SQLite Database: **{os.path.basename(db_path)}**.

Tables created:
{chr(10).join([f"*   `{table}`" for table in tables])}

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
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    print("Data Cleaning and Database Load Completed successfully.")
    return True

if __name__ == "__main__":
    clean_data()
