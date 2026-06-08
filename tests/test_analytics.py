import os
import sqlite3
import pandas as pd
import pytest
from src.optimization.xi_optimizer import PlayingXIOptimizer

DB_PATH = "data/processed/cricket_intelligence.db"

def test_database_tables():
    """
    Verifies that the cleaned SQLite database exists and contains all required tables.
    """
    assert os.path.exists(DB_PATH), "Database file does not exist."
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    required_tables = [
        "matches", "deliveries", "player_match_features", 
        "player_overall_metrics", "player_venue_stats", 
        "player_venue_insights", "venue_classification", "player_predictions"
    ]
    
    for table in required_tables:
        assert table in tables, f"Required table '{table}' is missing from the database."

def test_player_metrics_validity():
    """
    Verifies that calculated player performance metrics are within physical limits (0-100).
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM player_overall_metrics", conn)
    conn.close()
    
    assert len(df) > 0, "player_overall_metrics table is empty."
    
    # Test MIS boundaries
    assert df["match_impact_score"].min() >= 0.0, "MIS has negative values."
    assert df["match_impact_score"].max() <= 100.0, "MIS exceeds 100."
    
    # Test Consistency Index boundaries
    assert df["consistency_index"].min() >= 0.0, "Consistency Index has negative values."
    assert df["consistency_index"].max() <= 100.0, "Consistency Index exceeds 100."
    
    # Test Pressure Performance Index boundaries
    assert df["pressure_performance_index"].min() >= 0.0, "PPI has negative values."
    assert df["pressure_performance_index"].max() <= 100.0, "PPI exceeds 100."
    
    # Test CPIS composite rating boundaries
    assert df["cpis_rating"].min() >= 0.0, "CPIS Rating has negative values."
    assert df["cpis_rating"].max() <= 100.0, "CPIS Rating exceeds 100."

def test_playing_xi_optimizer():
    """
    Verifies that the mixed-integer playing XI optimizer works and satisfies key T20 selection constraints.
    """
    optimizer = PlayingXIOptimizer(db_path=DB_PATH)
    players_df = optimizer.load_players()
    
    # Run optimization under default constraints
    squad, captain, vice_captain = optimizer.optimize_xi(players_df)
    
    # Verifications
    assert len(squad) == 11, f"Optimizer selected {len(squad)} players instead of 11."
    assert captain in squad["player"].tolist(), "Captain is not in the selected squad."
    assert vice_captain in squad["player"].tolist(), "Vice-captain is not in the selected squad."
    assert captain != vice_captain, "Captain and Vice-captain are the same player."
    
    # Check wicketkeeper constraint
    wks_selected = squad[squad["role"] == "Wicketkeeper"]
    assert len(wks_selected) >= 1, "Optimal squad does not contain at least one Wicketkeeper."
    
    # Check opener constraint
    openers_selected = squad[squad["role"] == "Opener"]
    assert len(openers_selected) == 2, f"Optimal squad has {len(openers_selected)} openers instead of exactly 2."
