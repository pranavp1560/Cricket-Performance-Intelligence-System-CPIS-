import os
import sqlite3
import pandas as pd
import numpy as np

def run_venue_analytics(db_path="data/processed/cricket_intelligence.db"):
    """
    Computes venue-specific stats for each player and classifies venues.
    Saves tables 'player_venue_stats', 'player_venue_insights', and 'venue_classification' to the database.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}.")
        
    conn = sqlite3.connect(db_path)
    
    # Load raw tables
    matches_df = pd.read_sql("SELECT match_id, venue, winner, team1, team2 FROM matches", conn)
    deliveries_df = pd.read_sql("SELECT * FROM deliveries", conn)
    
    # Merge venue into deliveries for venue-grouped aggregations
    deliv_with_venue = deliveries_df.merge(matches_df[["match_id", "venue"]], on="match_id", how="left")
    
    # ------------------ VENUE CLASSIFICATION ------------------
    print("Classifying venues...")
    venue_groups = deliv_with_venue.groupby("venue")
    venue_classes = []
    
    for venue, v_data in venue_groups:
        total_runs = v_data["total_runs"].sum()
        total_balls = len(v_data[v_data["extra_runs"] == 0])
        if total_balls == 0:
            total_balls = len(v_data)
        rpo = (total_runs / total_balls) * 6.0
        
        total_wickets = v_data[
            (v_data["is_wicket"] == 1) & 
            (~v_data["dismissal_kind"].isin(["run out", "retired hurt"]))
        ].shape[0]
        
        runs_per_wicket = total_runs / total_wickets if total_wickets > 0 else total_runs
        
        # Determine classification
        # High RPO (> 8.2) & high runs per wicket (> 30) -> Batting-Friendly
        # Low RPO (< 7.8) or low runs per wicket (< 23) -> Bowling-Friendly
        # Otherwise -> Balanced
        if rpo > 8.1 and runs_per_wicket > 28.0:
            classification = "Batting-Friendly"
        elif rpo < 7.6 or runs_per_wicket < 22.0:
            classification = "Bowling-Friendly"
        else:
            classification = "Balanced"
            
        # Unique matches played at this venue
        venue_matches = deliv_with_venue[deliv_with_venue["venue"] == venue]["match_id"].nunique()
        avg_score_1st_inn = 0.0
        # Calculate average 1st innings score at this venue
        m_ids = matches_df[matches_df["venue"] == venue]["match_id"]
        v_1st_inn = deliveries_df[(deliveries_df["match_id"].isin(m_ids)) & (deliveries_df["inning"] == 1)]
        if venue_matches > 0:
            avg_score_1st_inn = v_1st_inn["total_runs"].sum() / venue_matches
            
        venue_classes.append({
            "venue": venue,
            "matches_played": venue_matches,
            "run_rate": rpo,
            "runs_per_wicket": runs_per_wicket,
            "avg_first_innings_score": avg_score_1st_inn,
            "classification": classification
        })
        
    venue_class_df = pd.DataFrame(venue_classes)
    venue_class_df.to_sql("venue_classification", conn, if_exists="replace", index=False)
    
    # ------------------ PLAYER VENUE STATS ------------------
    print("Computing player-specific venue stats...")
    
    # Batting venue stats
    bat_venue = deliv_with_venue.groupby(["batsman", "venue"])
    bat_venue_rows = []
    
    for (batsman, venue), group in bat_venue:
        if batsman == "":
            continue
        runs = group["batsman_runs"].sum()
        balls = len(group[group["extra_runs"] == 0])
        if balls == 0:
            balls = len(group)
        sr = (runs / balls) * 100 if balls > 0 else 0
        matches = group["match_id"].nunique()
        
        # count outs
        outs = group["player_dismissed"].notnull().sum() # actually this includes non-batsman dismissal if column has player_dismissed
        # More precise out count: count rows in this match where player_dismissed is this batsman
        outs = (group["player_dismissed"] == batsman).sum()
        avg = runs / outs if outs > 0 else runs
        
        bat_venue_rows.append({
            "player": batsman,
            "venue": venue,
            "bat_matches": matches,
            "bat_runs": runs,
            "bat_balls": balls,
            "bat_average": avg,
            "bat_strike_rate": sr
        })
    bat_v_df = pd.DataFrame(bat_venue_rows)
    
    # Bowling venue stats
    bowl_venue = deliv_with_venue.groupby(["bowler", "venue"])
    bowl_venue_rows = []
    
    for (bowler, venue), group in bowl_venue:
        if bowler == "":
            continue
        balls = len(group[group["extra_runs"] == 0])
        if balls == 0:
            balls = len(group)
        runs_conceded = group["total_runs"].sum()
        overs = balls / 6.0
        econ = (runs_conceded / overs) if overs > 0 else 0
        wickets = group[
            (group["is_wicket"] == 1) & 
            (~group["dismissal_kind"].isin(["run out", "retired hurt"]))
        ].shape[0]
        matches = group["match_id"].nunique()
        bowl_avg = runs_conceded / wickets if wickets > 0 else runs_conceded
        
        bowl_venue_rows.append({
            "player": bowler,
            "venue": venue,
            "bowl_matches": matches,
            "bowl_balls": balls,
            "bowl_runs_conceded": runs_conceded,
            "bowl_wickets": wickets,
            "bowl_economy": econ,
            "bowl_average": bowl_avg
        })
    bowl_v_df = pd.DataFrame(bowl_venue_rows)
    
    # Merge batting and bowling venue stats
    all_players = set(bat_v_df["player"]).union(set(bowl_v_df["player"]))
    all_venues = set(matches_df["venue"])
    
    player_venue_rows = []
    
    # Quick maps
    bat_map = bat_v_df.set_index(["player", "venue"]).to_dict(orient="index")
    bowl_map = bowl_v_df.set_index(["player", "venue"]).to_dict(orient="index")
    
    for p in all_players:
        for v in all_venues:
            key = (p, v)
            b_info = bat_map.get(key, {})
            bw_info = bowl_map.get(key, {})
            
            if not b_info and not bw_info:
                continue
                
            player_venue_rows.append({
                "player": p,
                "venue": v,
                "bat_matches": b_info.get("bat_matches", 0),
                "bat_runs": b_info.get("bat_runs", 0),
                "bat_balls_faced": b_info.get("bat_balls", 0),
                "bat_average": b_info.get("bat_average", 0.0),
                "bat_strike_rate": b_info.get("bat_strike_rate", 0.0),
                "bowl_matches": bw_info.get("bowl_matches", 0),
                "bowl_balls_bowled": bw_info.get("bowl_balls", 0),
                "bowl_runs_conceded": bw_info.get("bowl_runs_conceded", 0),
                "bowl_wickets": bw_info.get("bowl_wickets", 0),
                "bowl_economy": bw_info.get("bowl_economy", 0.0),
                "bowl_average": bw_info.get("bowl_average", 99.9)
            })
            
    pv_df = pd.DataFrame(player_venue_rows)
    pv_df.to_sql("player_venue_stats", conn, if_exists="replace", index=False)
    
    # ------------------ VENUE INSIGHT GENERATION ------------------
    print("Generating venue insights for each player...")
    player_insights = []
    venue_class_dict = venue_class_df.set_index("venue")["classification"].to_dict()
    
    for player, p_data in pv_df.groupby("player"):
        # We classify if the player is primarily a batsman or bowler
        # Let's count total runs vs total wickets
        tot_runs = p_data["bat_runs"].sum()
        tot_wicks = p_data["bowl_wickets"].sum()
        
        # Calculate composite score for each venue
        # Batting score = runs * (strike_rate/100)
        # Bowling score = wickets * 10 + max(0, 11 - economy) * 2
        p_data = p_data.copy()
        
        p_data["bat_score"] = p_data["bat_runs"] * (p_data["bat_strike_rate"] / 100.0)
        # Handle bowlers
        p_data["bowl_score"] = p_data["bowl_wickets"] * 20.0 + (12.0 - p_data["bowl_economy"]).clip(lower=0) * p_data["bowl_balls_bowled"] / 6.0
        
        if tot_runs > 150 and tot_wicks <= 5:
            # Batter
            p_data_valid = p_data[p_data["bat_matches"] >= 2]
            if len(p_data_valid) == 0:
                p_data_valid = p_data
            best_row = p_data_valid.sort_values(by="bat_score", ascending=False).iloc[0]
            worst_row = p_data_valid.sort_values(by="bat_score", ascending=True).iloc[0]
            
            best_venue = best_row["venue"]
            worst_venue = worst_row["venue"]
            
            best_desc = f"Excels here, averaging {best_row['bat_average']:.1f} runs at a strike rate of {best_row['bat_strike_rate']:.1f} ({venue_class_dict.get(best_venue, 'Balanced')} venue)."
            worst_desc = f"Struggles here, averaging just {worst_row['bat_average']:.1f} runs at a strike rate of {worst_row['bat_strike_rate']:.1f}."
            
        elif tot_wicks > 5 and tot_runs <= 100:
            # Bowler
            p_data_valid = p_data[p_data["bowl_matches"] >= 2]
            if len(p_data_valid) == 0:
                p_data_valid = p_data
            best_row = p_data_valid.sort_values(by="bowl_score", ascending=False).iloc[0]
            worst_row = p_data_valid.sort_values(by="bowl_score", ascending=True).iloc[0]
            
            best_venue = best_row["venue"]
            worst_venue = worst_row["venue"]
            
            best_desc = f"Dominates here with {best_row['bowl_wickets']} wickets and an economy of {best_row['bowl_economy']:.2f}."
            worst_desc = f"Expensive here, with an economy of {worst_row['bowl_economy']:.2f} and only {worst_row['bowl_wickets']} wickets."
            
        else:
            # All-rounder
            p_data["ar_score"] = p_data["bat_score"] + p_data["bowl_score"]
            p_data_valid = p_data[(p_data["bat_matches"] >= 2) | (p_data["bowl_matches"] >= 2)]
            if len(p_data_valid) == 0:
                p_data_valid = p_data
            best_row = p_data_valid.sort_values(by="ar_score", ascending=False).iloc[0]
            worst_row = p_data_valid.sort_values(by="ar_score", ascending=True).iloc[0]
            
            best_venue = best_row["venue"]
            worst_venue = worst_row["venue"]
            
            best_desc = f"Excellent all-round display: scored {best_row['bat_runs']} runs (SR {best_row['bat_strike_rate']:.1f}) and took {best_row['bowl_wickets']} wickets (Econ {best_row['bowl_economy']:.2f})."
            worst_desc = f"Subdued impact: scored {worst_row['bat_runs']} runs and took {worst_row['bowl_wickets']} wickets."
            
        player_insights.append({
            "player": player,
            "best_venue": best_venue,
            "best_venue_performance": best_desc,
            "worst_venue": worst_venue,
            "worst_venue_performance": worst_desc,
            "recommendation": f"For matches at {best_venue}, {player} is a must-have in the playing XI. {best_desc} Conversely, exercise caution at {worst_venue} as {player} {worst_desc.lower()}"
        })
        
    p_insights_df = pd.DataFrame(player_insights)
    p_insights_df.to_sql("player_venue_insights", conn, if_exists="replace", index=False)
    
    conn.close()
    print("Venue analytics completed successfully.")
    return True

if __name__ == "__main__":
    run_venue_analytics()
