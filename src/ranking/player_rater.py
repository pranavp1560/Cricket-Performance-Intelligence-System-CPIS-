import os
import sqlite3
import pandas as pd

def calculate_cpis_ratings(db_path="data/processed/cricket_intelligence.db", rankings_output="data/processed/player_rankings.csv"):
    """
    Calculates the composite CPIS Player Rating for each player:
    Rating = 0.35*MIS + 0.20*Consistency + 0.20*Pressure + 0.15*Batting + 0.10*Bowling.
    Ranks the players and outputs the Top 100 to a CSV.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}.")
        
    conn = sqlite3.connect(db_path)
    
    print("Loading player overall metrics...")
    metrics_df = pd.read_sql("SELECT * FROM player_overall_metrics", conn)
    
    # Calculate CPIS Rating
    # Formula:
    # 0.35 * MIS + 0.20 * Consistency + 0.20 * Pressure + 0.15 * Batting + 0.10 * Bowling
    metrics_df["cpis_rating"] = (
        0.35 * metrics_df["match_impact_score"] +
        0.20 * metrics_df["consistency_index"] +
        0.20 * metrics_df["pressure_performance_index"] +
        0.15 * metrics_df["batting_raw_rating"] +
        0.10 * metrics_df["bowling_raw_rating"]
    )
    
    # Round to 2 decimal places
    metrics_df["cpis_rating"] = metrics_df["cpis_rating"].round(2)
    
    # Sort and rank
    metrics_df = metrics_df.sort_values(by="cpis_rating", ascending=False).reset_index(drop=True)
    metrics_df["rank"] = metrics_df.index + 1
    
    # Write updated table back to SQLite database
    print("Saving updated metrics with CPIS Ratings to database...")
    metrics_df.to_sql("player_overall_metrics", conn, if_exists="replace", index=False)
    
    # Export Top 100 Rankings
    print(f"Exporting Top 100 Rankings to {rankings_output}...")
    top_100 = metrics_df.head(100)
    top_100.to_csv(rankings_output, index=False)
    
    # Print sample rankings
    print("\nTop 10 CPIS Player Rankings:")
    print(top_100[["rank", "player", "role", "matches", "match_impact_score", "consistency_index", "pressure_performance_index", "cpis_rating"]].head(10).to_string(index=False))
    
    conn.close()
    return True

if __name__ == "__main__":
    calculate_cpis_ratings()
