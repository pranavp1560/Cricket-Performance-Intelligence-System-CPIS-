import os
import sqlite3
import pandas as pd
import numpy as np

def run_analytics_pipeline(db_path="data/processed/cricket_intelligence.db"):
    """
    Runs the feature engineering and analytics engine.
    Computes match-level features, aggregates player-level stats,
    and calculates Match Impact Score (MIS), Consistency Index (CI), and Pressure Performance Index (PPI).
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Please run cleaning script first.")
        
    conn = sqlite3.connect(db_path)
    
    print("Reading matches and deliveries from database...")
    matches_df = pd.read_sql("SELECT * FROM matches", conn)
    deliveries_df = pd.read_sql("SELECT * FROM deliveries", conn)
    
    print("Engineering Match Context Features (Match Pressure Index, Chase Difficulty)...")
    
    # 1. Match Pressure Index (MPI) for each delivery
    # Formula:
    # Inning 1: over_pct * wickets_pct
    # Inning 2: (runs_required / (balls_remaining + 1)) * (wickets_lost / 10)
    deliveries_df["wickets_lost"] = 10 - deliveries_df["wickets_remaining"]
    
    def calculate_delivery_mpi(row):
        over_factor = row["over"] / 19.0
        if row["inning"] == 1:
            # Wickets down increases pressure, later overs increases pressure
            return 0.3 * over_factor + 0.7 * (row["wickets_lost"] / 10.0)
        else:
            # Inning 2: Chase
            if row["balls_remaining"] <= 0:
                req_rate = row["runs_required"] * 6.0
            else:
                req_rate = (row["runs_required"] / row["balls_remaining"]) * 6.0
            req_rate_factor = min(3.0, max(0.0, req_rate / 9.0)) # normalized around 9 RPO
            wickets_factor = (row["wickets_lost"] + 1) / 10.0
            return 0.5 * req_rate_factor + 0.5 * wickets_factor
            
    deliveries_df["delivery_mpi"] = deliveries_df.apply(calculate_delivery_mpi, axis=1)
    
    # Write updated deliveries with MPI back to SQLite
    deliveries_df.to_sql("deliveries", conn, if_exists="replace", index=False)
    
    # 2. Extract unique lists of players
    all_players = set(deliveries_df["batsman"].unique()).union(
        set(deliveries_df["bowler"].unique())
    ).union(
        set(deliveries_df["fielder"].unique())
    )
    all_players = {p for p in all_players if p and p != ""}
    
    # Get match details for fast lookup
    matches_dict = matches_df.set_index("match_id").to_dict(orient="index")
    
    # Prepare list for player-match aggregation
    player_match_rows = []
    
    print("Aggregating player stats per match...")
    
    # Group deliveries by match and inning
    grouped_delivs = deliveries_df.groupby(["match_id", "inning"])
    
    for (match_id, inning), group in grouped_delivs:
        m_info = matches_dict.get(match_id, {})
        if not m_info:
            continue
            
        is_playoff = m_info.get("is_playoff", 0)
        is_final = m_info.get("is_final", 0)
        
        # Batting stats
        bat_group = group.groupby("batsman")
        bat_stats = {}
        for batsman, b_data in bat_group:
            runs = b_data["batsman_runs"].sum()
            # Balls faced excludes wides (extra_runs > 0 and batsman_runs == 0 is extra, but let's look at ball type)
            # Cricsheet standard: wide balls faced are not counted in batsman's balls faced
            # We can check extra_runs / dismissal kind or just count balls where extra_runs is 0 (or not wide)
            # In our simulation, we didn't specify extra type, but we set extra_runs=1 for wides/no-balls.
            # Let's count balls faced: count rows where extra_runs == 0.
            balls = len(b_data[b_data["extra_runs"] == 0])
            if balls == 0:
                balls = len(b_data) # fallback
            sr = (runs / balls) * 100 if balls > 0 else 0
            fours = (b_data["batsman_runs"] == 4).sum()
            sixes = (b_data["batsman_runs"] == 6).sum()
            dots = ((b_data["batsman_runs"] == 0) & (b_data["extra_runs"] == 0)).sum()
            
            # Out status: did the player get out in this inning?
            # Check if player_dismissed contains this batsman's name
            is_out = (group["player_dismissed"] == batsman).any()
            out_val = 1 if is_out else 0
            
            # Pressure Runs: runs in overs 15-19, or in chases when required RPO > 9.0
            pressure_runs = b_data[
                (b_data["over"] >= 15) | 
                ((b_data["inning"] == 2) & (b_data["runs_required"] / (b_data["balls_remaining"].replace(0, 1)/6) > 9.0))
            ]["batsman_runs"].sum()
            
            # Chase Difficulty Score (CDS): mean of delivery MPI for deliveries faced
            cds = b_data["delivery_mpi"].mean() if len(b_data) > 0 else 1.0
            
            bat_stats[batsman] = {
                "runs": runs,
                "balls": balls,
                "sr": sr,
                "fours": fours,
                "sixes": sixes,
                "dots": dots,
                "is_out": out_val,
                "pressure_runs": pressure_runs,
                "cds": cds
            }
            
        # Bowling stats
        bowl_group = group.groupby("bowler")
        bowl_stats = {}
        for bowler, b_data in bowl_group:
            # Balls bowled excludes wides/noballs (extra_runs == 0 or let's count rows where extra_runs == 0)
            balls_bowled = len(b_data[b_data["extra_runs"] == 0])
            if balls_bowled == 0:
                balls_bowled = len(b_data)
            runs_conceded = b_data["total_runs"].sum() # total runs off bowler's deliveries
            overs = balls_bowled / 6.0
            econ = (runs_conceded / overs) * 1 if overs > 0 else 0
            
            # Wickets: excluding run outs and retired hurt
            wickets = b_data[
                (b_data["is_wicket"] == 1) & 
                (~b_data["dismissal_kind"].isin(["run out", "retired hurt", "obstructing the field"]))
            ].shape[0]
            
            death_wickets = b_data[
                (b_data["is_wicket"] == 1) & 
                (b_data["over"] >= 15) &
                (~b_data["dismissal_kind"].isin(["run out", "retired hurt", "obstructing the field"]))
            ].shape[0]
            
            dot_balls = (b_data["total_runs"] == 0).sum()
            
            bowl_stats[bowler] = {
                "balls_bowled": balls_bowled,
                "runs_conceded": runs_conceded,
                "overs": overs,
                "econ": econ,
                "wickets": wickets,
                "death_wickets": death_wickets,
                "dot_balls": dot_balls
            }
            
        # Fielding stats (catches, stumpings, run outs)
        # Note: fielders are associated with wickets in this group
        field_stats = {}
        w_data = group[group["is_wicket"] == 1]
        for idx, row in w_data.iterrows():
            fielder = row["fielder"]
            dk = row["dismissal_kind"]
            if pd.isna(fielder) or fielder == "":
                continue
                
            if fielder not in field_stats:
                field_stats[fielder] = {"catches": 0, "stumpings": 0, "run_outs": 0}
                
            if dk == "caught":
                field_stats[fielder]["catches"] += 1
            elif dk == "stumped":
                field_stats[fielder]["stumpings"] += 1
            elif dk == "run out":
                field_stats[fielder]["run_outs"] += 1
                
        # Combine all active players in this match
        match_players = set(bat_stats.keys()).union(set(bowl_stats.keys())).union(set(field_stats.keys()))
        
        for player in match_players:
            p_bat = bat_stats.get(player, {"runs":0, "balls":0, "sr":0, "fours":0, "sixes":0, "dots":0, "is_out":0, "pressure_runs":0, "cds":1.0})
            p_bowl = bowl_stats.get(player, {"balls_bowled":0, "runs_conceded":0, "overs":0, "econ":0, "wickets":0, "death_wickets":0, "dot_balls":0})
            p_field = field_stats.get(player, {"catches":0, "stumpings":0, "run_outs":0})
            
            # Context multipliers
            context_mult = 1.0
            if is_final == 1:
                context_mult = 1.5
            elif is_playoff == 1:
                context_mult = 1.2
                
            # Batting Match Impact
            # Runs + Strike Rate Bonus + Boundary Bonus + Pressure Runs * Chase difficulty
            bat_runs = p_bat["runs"]
            bat_sr = p_bat["sr"]
            boundaries = p_bat["fours"] + p_bat["sixes"]
            sr_bonus = 0.02 * max(0, bat_sr - 120) * bat_runs
            boundary_bonus = 1.5 * p_bat["fours"] + 3.0 * p_bat["sixes"]
            pressure_bonus = 2.0 * p_bat["pressure_runs"] * p_bat["cds"]
            
            bat_impact = bat_runs + sr_bonus + boundary_bonus + pressure_bonus
            
            # Bowling Match Impact
            # Wickets * 25 + Economy Bonus + Death Wickets * 15 + Dot Balls * 1.5
            bowl_wickets = p_bowl["wickets"]
            bowl_econ = p_bowl["econ"]
            bowl_overs = p_bowl["overs"]
            
            # Economy bonus (lower than 8.5 gets positive points, higher gets negative points capped)
            econ_bonus = 0
            if bowl_overs > 0.0:
                econ_diff = 8.5 - bowl_econ
                econ_bonus = econ_diff * 5 * bowl_overs # scales with overs bowled
            
            wicket_points = bowl_wickets * 25
            death_points = p_bowl["death_wickets"] * 15
            dot_points = p_bowl["dot_balls"] * 1.5
            
            bowl_impact = wicket_points + econ_bonus + death_points + dot_points
            if bowl_overs == 0.0:
                bowl_impact = 0.0
                
            # Fielding Match Impact
            field_impact = p_field["catches"] * 10 + p_field["stumpings"] * 15 + p_field["run_outs"] * 15
            
            # Raw Match Impact
            raw_impact = bat_impact + bowl_impact + field_impact
            
            # Context Adjusted Match Impact
            adjusted_impact = raw_impact * context_mult
            
            player_match_rows.append({
                "match_id": match_id,
                "season": m_info.get("season", 2021),
                "player": player,
                "runs": p_bat["runs"],
                "balls_faced": p_bat["balls"],
                "strike_rate": p_bat["sr"],
                "fours": p_bat["fours"],
                "sixes": p_bat["sixes"],
                "dot_balls_faced": p_bat["dots"],
                "is_out": p_bat["is_out"],
                "pressure_runs": p_bat["pressure_runs"],
                "chase_difficulty": p_bat["cds"],
                "balls_bowled": p_bowl["balls_bowled"],
                "runs_conceded": p_bowl["runs_conceded"],
                "overs_bowled": p_bowl["overs"],
                "economy": p_bowl["econ"],
                "wickets": p_bowl["wickets"],
                "death_wickets": p_bowl["death_wickets"],
                "dot_balls_bowled": p_bowl["dot_balls"],
                "catches": p_field["catches"],
                "stumpings": p_field["stumpings"],
                "run_outs": p_field["run_outs"],
                "bat_impact": bat_impact,
                "bowl_impact": bowl_impact,
                "field_impact": field_impact,
                "raw_match_impact": raw_impact,
                "match_impact_score": adjusted_impact
            })
            
    pm_df = pd.DataFrame(player_match_rows)
    print(f"Computed match features for {len(pm_df)} player-match rows.")
    pm_df.to_sql("player_match_features", conn, if_exists="replace", index=False)
    
    # ------------------ COMPUTE OVERALL PLAYER METRICS ------------------
    print("Aggregating overall player metrics across all seasons...")
    player_groups = pm_df.groupby("player")
    
    overall_rows = []
    
    # We find max Match Impact Score across all player-match records to normalize MIS to 0-100 scale
    max_match_impact = pm_df["match_impact_score"].max()
    if pd.isna(max_match_impact) or max_match_impact == 0:
        max_match_impact = 100.0
        
    for player, data in player_groups:
        total_matches = data["match_id"].nunique()
        
        # Batting aggregates
        bat_matches = data[data["balls_faced"] > 0]
        total_runs = data["runs"].sum()
        total_balls = data["balls_faced"].sum()
        total_outs = data["is_out"].sum()
        bat_avg = total_runs / total_outs if total_outs > 0 else (total_runs if total_balls > 0 else 0)
        overall_sr = (total_runs / total_balls) * 100 if total_balls > 0 else 0
        total_fours = data["fours"].sum()
        total_sixes = data["sixes"].sum()
        total_dots_faced = data["dot_balls_faced"].sum()
        boundary_pct = ((total_fours + total_sixes) / total_balls * 100) if total_balls > 0 else 0
        dot_pct_faced = (total_dots_faced / total_balls * 100) if total_balls > 0 else 0
        
        # Chase performance (runs in 2nd innings)
        chase_runs = 0
        chase_balls = 0
        # Let's map inning to delivery data or check match-level details
        # For simplicity, if they faced balls and the match has chase_difficulty (which in 1st inning is 1.0, but in 2nd innings it changes)
        # We can query deliveries table to get precise chase runs
        # Or check deliveries_df:
        p_delivs = deliveries_df[deliveries_df["batsman"] == player]
        chase_runs = p_delivs[p_delivs["is_chase"] == 1]["batsman_runs"].sum()
        chase_balls = len(p_delivs[(p_delivs["is_chase"] == 1) & (p_delivs["extra_runs"] == 0)])
        chase_sr = (chase_runs / chase_balls) * 100 if chase_balls > 0 else 0
        
        # Bowling aggregates
        bowl_matches = data[data["balls_bowled"] > 0]
        total_balls_bowled = data["balls_bowled"].sum()
        total_runs_conceded = data["runs_conceded"].sum()
        total_wickets = data["wickets"].sum()
        total_overs = total_balls_bowled / 6.0
        overall_econ = (total_runs_conceded / total_overs) if total_overs > 0 else 0
        bowl_avg = total_runs_conceded / total_wickets if total_wickets > 0 else (total_runs_conceded if total_balls_bowled > 0 else 99.9)
        wickets_per_match = total_wickets / len(bowl_matches) if len(bowl_matches) > 0 else 0
        
        total_dots_bowled = data["dot_balls_bowled"].sum()
        dot_pct_bowled = (total_dots_bowled / total_balls_bowled * 100) if total_balls_bowled > 0 else 0
        
        # Death over efficiency (overs 15-19)
        # Check deliveries table
        p_bowl_delivs = deliveries_df[deliveries_df["bowler"] == player]
        death_balls = len(p_bowl_delivs[(p_bowl_delivs["over"] >= 15) & (p_bowl_delivs["extra_runs"] == 0)])
        death_runs = p_bowl_delivs[p_bowl_delivs["over"] >= 15]["total_runs"].sum()
        death_wickets = p_bowl_delivs[
            (p_bowl_delivs["over"] >= 15) & 
            (p_bowl_delivs["is_wicket"] == 1) &
            (~p_bowl_delivs["dismissal_kind"].isin(["run out", "retired hurt"]))
        ].shape[0]
        death_overs = death_balls / 6.0
        death_econ = (death_runs / death_overs) if death_overs > 0 else 0
        death_strike_rate = (death_balls / death_wickets) if death_wickets > 0 else 99.9
        
        # Fielding aggregates
        catches = data["catches"].sum()
        stumpings = data["stumpings"].sum()
        run_outs = data["run_outs"].sum()
        
        # 1. Match Impact Score (MIS)
        # Mean Match Impact Score across all games, normalized to a 0-100 scale
        mean_match_impact = data["match_impact_score"].mean()
        mis_score = min(100.0, max(0.0, (mean_match_impact / max_match_impact) * 100))
        
        # 2. Consistency Index (CI)
        # Formula: 100 * (1 - min(1, CV / CV_threshold)) * Drop-off Factor
        # Coefficient of Variation (CV) = std / mean of match impact scores
        std_impact = data["match_impact_score"].std()
        mean_impact = data["match_impact_score"].mean()
        
        if pd.isna(std_impact) or mean_impact <= 0:
            cv = 1.2 # fallback high volatility
        else:
            cv = std_impact / mean_impact
            
        cv_threshold = 1.2
        
        # Active Player Penalty (if played less than 5 matches in total)
        drop_off = 1.0
        if total_matches < 10:
            drop_off = total_matches / 10.0
            
        ci_score = 100.0 * (1.0 - min(1.0, cv / cv_threshold)) * drop_off
        
        # 3. Pressure Performance Index (PPI)
        # Formula: 0.4 * Chase Success Contribution + 0.3 * Death Overs Performance + 0.2 * Playoffs/Finals Impact + 0.1 * Super Over Impact
        
        # A: Chase Success Contribution: runs scored in chases where target > 150
        # Let's check deliveries for player in chases where inning == 2 and total match target > 150
        # In our simulation, we can identify chases by matches where team2 (chasing) won, or just check inning == 2
        # Let's check player's average runs in inning 2
        inning2_data = data[data["runs"] > 0] # actually data is aggregated at match level, so we check if there are 2nd inning runs
        # We can read from deliveries to see chase runs scored in successful chases
        chase_success_runs = 0
        # Join deliveries with matches
        merged_del = p_delivs.merge(matches_df[["match_id", "winner"]], on="match_id", how="left")
        # A successful chase is Inning 2 and Winner is the batting team
        successful_chase_runs = merged_del[
            (merged_del["inning"] == 2) & 
            (merged_del["batting_team"] == merged_del["winner"])
        ]["batsman_runs"].sum()
        
        chase_success_contrib = min(100.0, (successful_chase_runs / max(1.0, total_runs)) * 100 * 2.0) # normalise, capped
        
        # B: Death Overs Performance: death economy and wicket taking
        # Bowler: death econ < 8 is excellent (100 points), > 12 is poor (0 points)
        # Batter: death strike rate > 180 is excellent (100 points), < 120 is poor (0 points)
        death_perf = 50.0 # balanced base
        if total_balls_bowled > 0 and death_overs > 0:
            # Bowler death performance
            death_perf = max(0.0, min(100.0, (12.0 - death_econ) / 4.0 * 100))
        elif total_balls > 0:
            # Batter death performance
            # death strike rate = runs scored in death overs / balls faced in death overs
            death_bat_runs = p_delivs[p_delivs["over"] >= 15]["batsman_runs"].sum()
            death_bat_balls = len(p_delivs[(p_delivs["over"] >= 15) & (p_delivs["extra_runs"] == 0)])
            death_bat_sr = (death_bat_runs / death_bat_balls * 100) if death_bat_balls > 0 else 0
            if death_bat_sr > 0:
                death_perf = max(0.0, min(100.0, (death_bat_sr - 100.0) / 80.0 * 100))
                
        # C: Playoffs/Finals Impact
        playoff_matches = data[data["match_id"].isin(matches_df[matches_df["is_playoff"] == 1]["match_id"])]
        playoff_impact = 50.0 # base
        if len(playoff_matches) > 0:
            playoff_mean = playoff_matches["match_impact_score"].mean()
            regular_mean = data[~data["match_id"].isin(matches_df[matches_df["is_playoff"] == 1]["match_id"])]["match_impact_score"].mean()
            if regular_mean > 0:
                playoff_impact = min(100.0, max(0.0, (playoff_mean / regular_mean) * 50.0))
            else:
                playoff_impact = 100.0
                
        # D: Super Over Impact
        super_over_del = p_delivs[p_delivs["is_super_over"] == 1]
        super_over_impact = 50.0
        if len(super_over_del) > 0:
            so_runs = super_over_del["batsman_runs"].sum()
            so_wickets = p_bowl_delivs[
                (p_bowl_delivs["is_super_over"] == 1) & 
                (p_bowl_delivs["is_wicket"] == 1)
            ].shape[0]
            super_over_impact = min(100.0, (so_runs * 10 + so_wickets * 30) + 30.0)
            
        # Composite PPI
        ppi_score = 0.4 * chase_success_contrib + 0.3 * death_perf + 0.2 * playoff_impact + 0.1 * super_over_impact
        ppi_score = min(100.0, max(0.0, ppi_score))
        
        # Identify Primary Playing Role
        # Opener, Middle Order, Finisher, All-Rounder, Bowler, Wicketkeeper
        # We can look up in our original player lists, or infer based on data:
        # Let's count balls faced vs balls bowled.
        primary_role = "All-Rounder"
        if total_balls > 0 and total_balls_bowled == 0:
            # Batter or Wicketkeeper
            # Check if has stumpings or catches > 0.5 per match
            if stumpings > 0 or (catches / total_matches > 0.6 and total_runs > 100):
                primary_role = "Wicketkeeper"
            elif total_runs / total_matches > 20:
                # Batsman: check if they open or bat middle order
                # In simulation, we had specific names, but let's check:
                # If they face balls in early overs (over < 6) in most matches, they are Opener
                early_balls = p_delivs[p_delivs["over"] < 6].shape[0]
                if early_balls / max(1, len(p_delivs)) > 0.3:
                    primary_role = "Opener"
                else:
                    # Middle order or finisher
                    late_balls = p_delivs[p_delivs["over"] >= 15].shape[0]
                    if late_balls / max(1, len(p_delivs)) > 0.4:
                        primary_role = "Finisher"
                    else:
                        primary_role = "Middle Order"
            else:
                primary_role = "Middle Order"
        elif total_balls_bowled > 0 and total_balls == 0:
            primary_role = "Bowler"
        else:
            # Both batting and bowling
            if total_runs / total_matches > 15 and total_balls_bowled / total_matches > 6:
                primary_role = "All-Rounder"
            elif total_runs / total_matches > 15:
                # Batting all rounder or middle order
                primary_role = "Middle Order"
            else:
                primary_role = "Bowler"
                
        # Raw ratings for Batting/Bowling components (0-100 scale)
        # Batting Raw Rating = Average runs per match * SR factor
        bat_raw = min(100.0, (total_runs / total_matches) * (overall_sr / 120.0) * 1.5)
        # Bowling Raw Rating = Wickets per match * Economy factor
        bowl_raw = 0.0
        if total_balls_bowled > 0:
            econ_factor = max(0.1, (11.0 - overall_econ) / 4.0) # higher is better
            bowl_raw = min(100.0, wickets_per_match * 30.0 * econ_factor)
            
        overall_rows.append({
            "player": player,
            "role": primary_role,
            "matches": total_matches,
            "runs": total_runs,
            "balls_faced": total_balls,
            "batting_average": bat_avg,
            "strike_rate": overall_sr,
            "fours": total_fours,
            "sixes": total_sixes,
            "boundary_percentage": boundary_pct,
            "dot_percentage_faced": dot_pct_faced,
            "chase_runs": chase_runs,
            "chase_strike_rate": chase_sr,
            "balls_bowled": total_balls_bowled,
            "runs_conceded": total_runs_conceded,
            "wickets": total_wickets,
            "economy": overall_econ,
            "bowling_average": bowl_avg,
            "wickets_per_match": wickets_per_match,
            "dot_percentage_bowled": dot_pct_bowled,
            "death_economy": death_econ,
            "death_strike_rate": death_strike_rate,
            "catches": catches,
            "stumpings": stumpings,
            "run_outs": run_outs,
            "batting_raw_rating": bat_raw,
            "bowling_raw_rating": bowl_raw,
            "match_impact_score": mis_score,
            "consistency_index": ci_score,
            "pressure_performance_index": ppi_score
        })
        
    overall_df = pd.DataFrame(overall_rows)
    print(f"Aggregated metrics for {len(overall_df)} unique players.")
    
    # Save to SQLite
    overall_df.to_sql("player_overall_metrics", conn, if_exists="replace", index=False)
    
    conn.close()
    print("Analytics metrics pipeline completed successfully.")
    return True

if __name__ == "__main__":
    run_analytics_pipeline()
