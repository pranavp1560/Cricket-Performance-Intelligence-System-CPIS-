import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(output_dir="data/raw", num_seasons=5, seed=42):
    """
    Generates high-fidelity synthetic raw IPL/T20 data for analysis.
    Simulates seasons, matches, innings, overs, deliveries, with player skills,
    venue biases, chase contexts, playoff tension, and data quality anomalies.
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing player database and team mapping...")
    
    # 1. Define Teams
    teams = [
        "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bengaluru", 
        "Kolkata Knight Riders", "Delhi Capitals", "Rajasthan Royals", 
        "Gujarat Titans", "Sunrisers Hyderabad"
    ]
    
    # 2. Define Venues and their batting/bowling indices
    # Higher batting index = more runs, higher bowling index = spin/wickets friendly
    venues = {
        "Wankhede Stadium, Mumbai": {"bat_idx": 1.2, "bowl_idx": 0.95, "type": "Batting-Friendly"},
        "M. Chinnaswamy Stadium, Bengaluru": {"bat_idx": 1.35, "bowl_idx": 0.9, "type": "Batting-Friendly"},
        "MA Chidambaram Stadium, Chepauk": {"bat_idx": 0.85, "bowl_idx": 1.2, "type": "Spin-Friendly"},
        "Eden Gardens, Kolkata": {"bat_idx": 1.15, "bowl_idx": 1.0, "type": "Balanced"},
        "Narendra Modi Stadium, Ahmedabad": {"bat_idx": 1.1, "bowl_idx": 1.05, "type": "Balanced"},
        "Rajiv Gandhi International Stadium, Hyderabad": {"bat_idx": 1.05, "bowl_idx": 1.1, "type": "Balanced"},
        "Arun Jaitley Stadium, Delhi": {"bat_idx": 1.1, "bowl_idx": 1.0, "type": "Batting-Friendly"}
    }
    venue_names = list(venues.keys())
    
    # 3. Player Database with profiles
    # Structure: { name: (role, batting_skill [0-100], bowling_skill [0-100], is_wk) }
    # Roles: Opener, Middle Order, Finisher, All-Rounder, Bowler
    players = {
        # Batsmen
        "Virat Kohli": ("Opener", 92, 5, False, "Royal Challengers Bengaluru"),
        "Rohit Sharma": ("Opener", 88, 10, False, "Mumbai Indians"),
        "Shubman Gill": ("Opener", 90, 5, False, "Gujarat Titans"),
        "Yashasvi Jaiswal": ("Opener", 87, 5, False, "Rajasthan Royals"),
        "Jos Buttler": ("Opener", 91, 5, True, "Rajasthan Royals"),
        "Ruturaj Gaikwad": ("Opener", 89, 5, False, "Chennai Super Kings"),
        "Faf du Plessis": ("Opener", 86, 5, False, "Royal Challengers Bengaluru"),
        "Suryakumar Yadav": ("Middle Order", 93, 5, False, "Mumbai Indians"),
        "KL Rahul": ("Opener", 85, 5, True, "Delhi Capitals"),
        "Heinrich Klaasen": ("Middle Order", 91, 5, True, "Sunrisers Hyderabad"),
        "Nicholas Pooran": ("Middle Order", 88, 5, True, "Sunrisers Hyderabad"),
        "Dinesh Karthik": ("Finisher", 82, 5, True, "Royal Challengers Bengaluru"),
        "Rinku Singh": ("Finisher", 87, 5, False, "Kolkata Knight Riders"),
        "Shimron Hetmyer": ("Finisher", 80, 5, False, "Rajasthan Royals"),
        "David Miller": ("Finisher", 84, 5, False, "Gujarat Titans"),
        "Shivam Dube": ("Middle Order", 85, 45, False, "Chennai Super Kings"),
        "Tilak Varma": ("Middle Order", 83, 30, False, "Mumbai Indians"),
        
        # All-Rounders
        "Hardik Pandya": ("All-Rounder", 83, 78, False, "Mumbai Indians"),
        "Ravindra Jadeja": ("All-Rounder", 78, 88, False, "Chennai Super Kings"),
        "Glenn Maxwell": ("All-Rounder", 84, 70, False, "Royal Challengers Bengaluru"),
        "Marcus Stoinis": ("All-Rounder", 81, 72, False, "Lucknow Super Giants"), # Let's assume some team mappings
        "Andre Russell": ("All-Rounder", 86, 80, False, "Kolkata Knight Riders"),
        "Axar Patel": ("All-Rounder", 76, 85, False, "Delhi Capitals"),
        "Rashid Khan": ("All-Rounder", 70, 95, False, "Gujarat Titans"),
        "Sunil Narine": ("All-Rounder", 75, 90, False, "Kolkata Knight Riders"),
        
        # Wicket Keepers (primarily)
        "MS Dhoni": ("Finisher", 81, 5, True, "Chennai Super Kings"),
        "Rishabh Pant": ("Middle Order", 86, 5, True, "Delhi Capitals"),
        "Sanju Samson": ("Middle Order", 87, 5, True, "Rajasthan Royals"),
        "Ishan Kishan": ("Opener", 82, 5, True, "Mumbai Indians"),
        
        # Bowlers
        "Jasprit Bumrah": ("Bowler", 15, 98, False, "Mumbai Indians"),
        "Yuzvendra Chahal": ("Bowler", 10, 90, False, "Rajasthan Royals"),
        "Mohammed Shami": ("Bowler", 12, 89, False, "Gujarat Titans"),
        "Kagiso Rabada": ("Bowler", 14, 88, False, "Punjab Kings"),
        "Trent Boult": ("Bowler", 15, 91, False, "Rajasthan Royals"),
        "Bhuvneshwar Kumar": ("Bowler", 20, 84, False, "Sunrisers Hyderabad"),
        "Arshdeep Singh": ("Bowler", 12, 85, False, "Punjab Kings"),
        "Mitchell Starc": ("Bowler", 18, 87, False, "Kolkata Knight Riders"),
        "Kuldeep Yadav": ("Bowler", 11, 89, False, "Delhi Capitals"),
        "Tushar Deshpande": ("Bowler", 10, 78, False, "Chennai Super Kings"),
        "Mohit Sharma": ("Bowler", 12, 80, False, "Gujarat Titans"),
        "Varun Chakravarthy": ("Bowler", 8, 86, False, "Kolkata Knight Riders"),
        "Sandeeep Sharma": ("Bowler", 10, 81, False, "Rajasthan Royals"),
        "Mohammed Siraj": ("Bowler", 11, 83, False, "Royal Challengers Bengaluru"),
        "Pat Cummins": ("Bowler", 35, 88, False, "Sunrisers Hyderabad")
    }

    # Add other filler players to complete a roster of 100+ players
    filler_batters = [f"Batter_{i}" for i in range(1, 40)]
    filler_bowlers = [f"Bowler_{i}" for i in range(1, 40)]
    for fb in filler_batters:
        players[fb] = (np.random.choice(["Opener", "Middle Order", "Finisher"]), np.random.randint(60, 80), 5, np.random.rand() < 0.1, np.random.choice(teams))
    for fbw in filler_bowlers:
        players[fbw] = ("Bowler", np.random.randint(5, 25), np.random.randint(60, 80), False, np.random.choice(teams))
    
    player_names = list(players.keys())
    
    match_list = []
    delivery_list = []
    
    match_id_counter = 1001
    
    start_date = datetime(2021, 4, 9)
    
    print(f"Simulating {num_seasons} seasons of cricket matches...")
    for season_idx in range(num_seasons):
        season_year = 2021 + season_idx
        num_matches_in_season = 60 # standard IPL size
        season_start_date = start_date + timedelta(days=season_idx * 365)
        
        for match_in_season in range(num_matches_in_season):
            match_id = match_id_counter
            match_id_counter += 1
            
            # Match date
            match_date = (season_start_date + timedelta(days=match_in_season)).strftime("%Y-%m-%d")
            
            # Select Teams
            team_pair = np.random.choice(teams, size=2, replace=False)
            t1, t2 = team_pair[0], team_pair[1]
            
            # Toss
            toss_winner = np.random.choice([t1, t2])
            toss_decision = np.random.choice(["bat", "field"])
            
            # Venue
            venue = np.random.choice(venue_names)
            v_info = venues[venue]
            
            # Playoff / Final Flag
            is_playoff = 1 if match_in_season >= 56 else 0
            is_final = 1 if match_in_season == 59 else 0
            
            # Determine batting first and second
            if toss_decision == "bat":
                batting_first = toss_winner
                fielding_first = t2 if toss_winner == t1 else t1
            else:
                fielding_first = toss_winner
                batting_first = t2 if toss_winner == t1 else t1
            
            # Select playing squads (11 players for each team)
            # Pick players belonging to that team, or filler players if roster short
            t1_players = [p for p, info in players.items() if info[4] == t1 or (info[4] not in teams and np.random.rand() < 0.15)]
            t2_players = [p for p, info in players.items() if info[4] == t2 or (info[4] not in teams and np.random.rand() < 0.15)]
            
            # Ensure we have at least 11 players per team
            if len(t1_players) < 11:
                t1_players += list(np.random.choice([p for p in player_names if p not in t1_players], size=11-len(t1_players), replace=False))
            if len(t2_players) < 11:
                t2_players += list(np.random.choice([p for p in player_names if p not in t2_players], size=11-len(t2_players), replace=False))
                
            t1_squad = list(np.random.choice(t1_players, size=11, replace=False))
            t2_squad = list(np.random.choice(t2_players, size=11, replace=False))
            
            # Pick playing roles
            # Wicketkeeper
            t1_wks = [p for p in t1_squad if players[p][3]]
            t1_wk = t1_wks[0] if len(t1_wks) > 0 else np.random.choice(t1_squad)
            t2_wks = [p for p in t2_squad if players[p][3]]
            t2_wk = t2_wks[0] if len(t2_wks) > 0 else np.random.choice(t2_squad)
            
            # Bowlers (who will bowl in this match)
            # Bowlers are players whose role is Bowler or All-Rounder
            t1_bowlers = [p for p in t1_squad if players[p][0] in ["Bowler", "All-Rounder"]]
            if len(t1_bowlers) < 5:
                t1_bowlers += list(np.random.choice([p for p in t1_squad if p not in t1_bowlers], size=5-len(t1_bowlers), replace=False))
            t1_bowlers = list(np.random.choice(t1_bowlers, size=5, replace=False))
            
            t2_bowlers = [p for p in t2_squad if players[p][0] in ["Bowler", "All-Rounder"]]
            if len(t2_bowlers) < 5:
                t2_bowlers += list(np.random.choice([p for p in t2_squad if p not in t2_bowlers], size=5-len(t2_bowlers), replace=False))
            t2_bowlers = list(np.random.choice(t2_bowlers, size=5, replace=False))
            
            # Batting orders
            # Openers first, then middle order, then bowlers
            t1_batters_sorted = sorted(t1_squad, key=lambda p: {"Opener":0, "Middle Order":1, "All-Rounder":2, "Finisher":3, "Bowler":4}[players[p][0]])
            t2_batters_sorted = sorted(t2_squad, key=lambda p: {"Opener":0, "Middle Order":1, "All-Rounder":2, "Finisher":3, "Bowler":4}[players[p][0]])
            
            # Simulate Inning 1
            runs_1, wickets_1, deliveries_1 = simulate_inning(
                match_id, 1, batting_first, fielding_first, t1_batters_sorted if batting_first == t1 else t2_batters_sorted, 
                t2_bowlers if batting_first == t1 else t1_bowlers, t2_wk if batting_first == t1 else t1_wk, 
                v_info, is_playoff, is_final, None, delivery_list, players
            )
            
            # Simulate Inning 2
            runs_2, wickets_2, deliveries_2 = simulate_inning(
                match_id, 2, fielding_first, batting_first, t2_batters_sorted if batting_first == t1 else t1_batters_sorted,
                t1_bowlers if batting_first == t1 else t2_bowlers, t1_wk if batting_first == t1 else t2_wk,
                v_info, is_playoff, is_final, runs_1, delivery_list, players
            )
            
            # Determine Winner
            if runs_1 > runs_2:
                winner = batting_first
                win_by_runs = runs_1 - runs_2
                win_by_wickets = 0
                margin = win_by_runs
            elif runs_2 > runs_1:
                winner = fielding_first
                win_by_runs = 0
                win_by_wickets = 10 - wickets_2
                margin = win_by_wickets
            else:
                # Tie: Simulate Super Over
                winner = np.random.choice([t1, t2])
                win_by_runs = 0
                win_by_wickets = 0
                margin = 0
                # Record Super Over event in delivery database
                simulate_super_over(match_id, t1, t2, delivery_list, players)
            
            match_list.append({
                "match_id": match_id,
                "season": season_year,
                "date": match_date,
                "venue": venue,
                "team1": t1,
                "team2": t2,
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "winner": winner,
                "win_by_runs": win_by_runs,
                "win_by_wickets": win_by_wickets,
                "margin": margin,
                "is_playoff": is_playoff,
                "is_final": is_final
            })
            
    matches_df = pd.DataFrame(match_list)
    deliveries_df = pd.DataFrame(delivery_list)
    
    # 4. Inject anomalies for cleaning pipeline (Phase 2)
    print("Injecting typical data anomalies (nulls, duplicates, spelling variants)...")
    # Duplicate match_id rows (a small percentage)
    dup_matches = matches_df.sample(n=2, random_state=42).copy()
    matches_df = pd.concat([matches_df, dup_matches], ignore_index=True)
    
    # Duplicate deliveries (a small percentage)
    dup_deliveries = deliveries_df.sample(n=10, random_state=42).copy()
    deliveries_df = pd.concat([deliveries_df, dup_deliveries], ignore_index=True)
    
    # Null values in winner / margin for a couple of rows (washouts)
    washout_indices = matches_df.sample(n=2, random_state=42).index
    matches_df.loc[washout_indices, "winner"] = None
    matches_df.loc[washout_indices, "margin"] = None
    
    # Spelling variants of top players in raw delivery data
    # Virat Kohli -> V. Kohli, V Kohli
    # Jasprit Bumrah -> J Bumrah
    vk_indices = deliveries_df[deliveries_df["batsman"] == "Virat Kohli"].sample(frac=0.08, random_state=42).index
    deliveries_df.loc[vk_indices, "batsman"] = "V. Kohli"
    jb_indices = deliveries_df[deliveries_df["bowler"] == "Jasprit Bumrah"].sample(frac=0.08, random_state=42).index
    deliveries_df.loc[jb_indices, "bowler"] = "J Bumrah"
    
    # Missing values in dismissals (nan instead of '')
    deliveries_df["dismissal_kind"] = deliveries_df["dismissal_kind"].replace('', np.nan)
    deliveries_df["player_dismissed"] = deliveries_df["player_dismissed"].replace('', np.nan)
    deliveries_df["fielder"] = deliveries_df["fielder"].replace('', np.nan)
    
    # Export CSVs
    matches_df.to_csv(f"{output_dir}/matches_raw.csv", index=False)
    deliveries_df.to_csv(f"{output_dir}/deliveries_raw.csv", index=False)
    
    print(f"Data Generation Completed: {len(matches_df)} matches and {len(deliveries_df)} deliveries saved in {output_dir}/")
    return len(matches_df), len(deliveries_df)

def simulate_inning(match_id, inning, batting_team, bowling_team, squad_batting, squad_bowling, opponent_wk, venue_info, is_playoff, is_final, target, delivery_list, players_skills):
    """
    Simulates a realistic inning of 20 overs ball-by-ball.
    """
    runs = 0
    wickets = 0
    over = 0
    ball_in_over = 1
    
    # Keep track of active batsmen
    striker_idx = 0
    non_striker_idx = 1
    
    batsman_on_strike = squad_batting[striker_idx]
    batsman_non_striker = squad_batting[non_striker_idx]
    
    next_batsman_idx = 2
    
    bat_idx = venue_info["bat_idx"]
    bowl_idx = venue_info["bowl_idx"]
    
    # Adjust for playoff/final pressure
    pressure_mult = 1.05 if is_final else (1.02 if is_playoff else 1.0)
    
    # Bowler rotation: assign overs to the 5 bowlers
    # Bowler list: squad_bowling (5 players)
    # Bowlers will bowl 4 overs each
    bowler_overs = {b: 0 for b in squad_bowling}
    
    for over in range(20):
        # Pick bowler for this over
        # Death overs (16-20) bowled by best death bowlers (highest skill)
        # Powerplay (0-5) bowled by opening bowlers
        if over >= 16:
            # Sort bowlers by bowling skill desc, pick one who has overs left
            available_bowlers = [b for b, o in bowler_overs.items() if o < 4]
            if not available_bowlers:
                available_bowlers = squad_bowling
            bowler = sorted(available_bowlers, key=lambda b: players_skills[b][2], reverse=True)[0]
        else:
            available_bowlers = [b for b, o in bowler_overs.items() if o < 4]
            if not available_bowlers:
                available_bowlers = squad_bowling
            bowler = np.random.choice(available_bowlers)
            
        bowler_overs[bowler] += 1
        
        # Inning chase details
        is_chase = 1 if inning == 2 else 0
        
        ball_in_over = 1
        while ball_in_over <= 6:
            # Check if chase is complete
            if target is not None and runs > target:
                break
                
            # Ball-by-ball event probability modeling
            # Inputs: batsman skill, bowler skill, venue indices, match situation
            bat_skill = players_skills[batsman_on_strike][1]
            bowl_skill = players_skills[bowler][2]
            
            # Wicket probability
            # Base wicket probability is ~3.5% per ball, adjusted by skills
            p_wicket = 0.035 * (bowl_skill / 70) * (80 / bat_skill) * bowl_idx * pressure_mult
            # In death overs, batsmen take more risks -> higher wicket prob
            if over >= 16:
                p_wicket *= 1.4
            
            # Limit probability
            p_wicket = min(0.25, max(0.005, p_wicket))
            
            is_wicket_ball = np.random.rand() < p_wicket
            
            if is_wicket_ball:
                # Wicket event!
                wickets += 1
                player_dismissed = batsman_on_strike
                
                # Determine dismissal kind
                # catches (60%), bowled (15%), lbw (15%), run out (5%), stumped (5%)
                dismiss_choices = ["caught", "bowled", "lbw", "run out", "stumped"]
                dismiss_kind = np.random.choice(dismiss_choices, p=[0.6, 0.15, 0.15, 0.05, 0.05])
                
                fielder = ""
                if dismiss_kind == "caught":
                    # Random fielder from opposition team
                    fielder = np.random.choice([p for p in squad_bowling if p != bowler])
                elif dismiss_kind == "stumped":
                    fielder = opponent_wk
                elif dismiss_kind == "run out":
                    fielder = np.random.choice(squad_bowling)
                
                # Append delivery
                delivery_record = {
                    "match_id": match_id,
                    "inning": inning,
                    "over": over,
                    "ball": ball_in_over,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "batsman": batsman_on_strike,
                    "non_striker": batsman_non_striker,
                    "bowler": bowler,
                    "batsman_runs": 0,
                    "extra_runs": 0,
                    "total_runs": 0,
                    "is_wicket": 1,
                    "dismissal_kind": dismiss_kind,
                    "player_dismissed": player_dismissed,
                    "fielder": fielder,
                    "is_chase": is_chase,
                    "is_super_over": 0
                }
                
                if is_chase:
                    balls_rem = (19 - over) * 6 + (6 - ball_in_over)
                    delivery_record.update({
                        "runs_required": max(0, target + 1 - runs),
                        "balls_remaining": balls_rem,
                        "wickets_remaining": 10 - wickets
                    })
                    
                delivery_list.append(delivery_record)
                
                # Batsman changes
                if wickets >= 10:
                    break
                else:
                    # Next batsman in order
                    batsman_on_strike = squad_batting[next_batsman_idx]
                    next_batsman_idx += 1
            else:
                # Scoring ball
                # Model run scoring probability
                # Base scoring distributions
                # Opener: higher 4s and 6s, bowlers: higher dot balls (0 runs)
                # We calculate a score propensity
                run_propensity = (bat_skill / 70) * (80 / bowl_skill) * bat_idx
                # Adjust for over (powerplay or death overs have higher scoring intent)
                if over < 6 or over >= 16:
                    run_propensity *= 1.25
                
                # Probability distribution for runs (0, 1, 2, 4, 6, extra)
                # Adjust based on run_propensity
                if run_propensity > 1.3:
                    p_runs = [0.3, 0.35, 0.08, 0.15, 0.10, 0.02] # high strike rate
                elif run_propensity < 0.7:
                    p_runs = [0.55, 0.25, 0.05, 0.10, 0.03, 0.02] # bowler or defensive batting
                else:
                    p_runs = [0.42, 0.32, 0.08, 0.11, 0.05, 0.02] # balanced
                
                outcomes = [0, 1, 2, 4, 6, "extra"]
                run_idx = np.random.choice(len(outcomes), p=p_runs)
                run_outcome = outcomes[run_idx]
                
                b_runs = 0
                e_runs = 0
                
                if run_outcome == "extra":
                    # Wide or No ball: extra runs, and ball is re-bowled (so over ball count doesn't increment)
                    e_runs = 1
                    runs += e_runs
                    delivery_record = {
                        "match_id": match_id,
                        "inning": inning,
                        "over": over,
                        "ball": ball_in_over,
                        "batting_team": batting_team,
                        "bowling_team": bowling_team,
                        "batsman": batsman_on_strike,
                        "non_striker": batsman_non_striker,
                        "bowler": bowler,
                        "batsman_runs": 0,
                        "extra_runs": 1,
                        "total_runs": 1,
                        "is_wicket": 0,
                        "dismissal_kind": "",
                        "player_dismissed": "",
                        "fielder": "",
                        "is_chase": is_chase,
                        "is_super_over": 0
                    }
                    if is_chase:
                        balls_rem = (19 - over) * 6 + (6 - ball_in_over) + 1 # +1 since wide doesn't reduce balls remaining
                        delivery_record.update({
                            "runs_required": max(0, target + 1 - runs),
                            "balls_remaining": balls_rem,
                            "wickets_remaining": 10 - wickets
                        })
                    delivery_list.append(delivery_record)
                    continue # Re-bowl
                else:
                    b_runs = run_outcome
                    runs += b_runs
                    
                    delivery_record = {
                        "match_id": match_id,
                        "inning": inning,
                        "over": over,
                        "ball": ball_in_over,
                        "batting_team": batting_team,
                        "bowling_team": bowling_team,
                        "batsman": batsman_on_strike,
                        "non_striker": batsman_non_striker,
                        "bowler": bowler,
                        "batsman_runs": b_runs,
                        "extra_runs": 0,
                        "total_runs": b_runs,
                        "is_wicket": 0,
                        "dismissal_kind": "",
                        "player_dismissed": "",
                        "fielder": "",
                        "is_chase": is_chase,
                        "is_super_over": 0
                    }
                    if is_chase:
                        balls_rem = (19 - over) * 6 + (6 - ball_in_over)
                        delivery_record.update({
                            "runs_required": max(0, target + 1 - runs),
                            "balls_remaining": balls_rem,
                            "wickets_remaining": 10 - wickets
                        })
                    delivery_list.append(delivery_record)
                
                # Rotate strike on odd runs
                if b_runs in [1, 3]:
                    batsman_on_strike, batsman_non_striker = batsman_non_striker, batsman_on_strike
                    
            ball_in_over += 1
            
        # End of over: switch strike
        batsman_on_strike, batsman_non_striker = batsman_non_striker, batsman_on_strike
        
        # Check if inning ends early (chase complete or all out)
        if wickets >= 10:
            break
        if target is not None and runs > target:
            break
            
    return runs, wickets, len(delivery_list)

def simulate_super_over(match_id, t1, t2, delivery_list, players_skills):
    """
    Simulates a Super Over event for tied matches.
    """
    # 6 balls, 3 wickets limit
    teams = [t1, t2]
    np.random.shuffle(teams)
    super_bat, super_bowl = teams[0], teams[1]
    
    for inn in [3, 4]:
        bat_team = super_bat if inn == 3 else super_bowl
        bowl_team = super_bowl if inn == 3 else super_bat
        
        # Pick best batsman and bowler for each team
        t_batters = [p for p, info in players_skills.items() if info[4] == bat_team]
        t_bowlers = [p for p, info in players_skills.items() if info[4] == bowl_team]
        
        bat_name = sorted(t_batters, key=lambda x: players_skills[x][1], reverse=True)[0]
        non_st_name = sorted(t_batters, key=lambda x: players_skills[x][1], reverse=True)[1]
        bowl_name = sorted(t_bowlers, key=lambda x: players_skills[x][2], reverse=True)[0]
        
        for ball_idx in range(1, 7):
            p_wicket = 0.10 # super over pressure is high
            is_wicket = np.random.rand() < p_wicket
            
            if is_wicket:
                delivery_record = {
                    "match_id": match_id,
                    "inning": inn,
                    "over": 0,
                    "ball": ball_idx,
                    "batting_team": bat_team,
                    "bowling_team": bowl_team,
                    "batsman": bat_name,
                    "non_striker": non_st_name,
                    "bowler": bowl_name,
                    "batsman_runs": 0,
                    "extra_runs": 0,
                    "total_runs": 0,
                    "is_wicket": 1,
                    "dismissal_kind": "caught",
                    "player_dismissed": bat_name,
                    "fielder": non_st_name,
                    "is_chase": 1 if inn == 4 else 0,
                    "is_super_over": 1
                }
                delivery_list.append(delivery_record)
                break # Super over ends on wicket typically in simplification
            else:
                runs = np.random.choice([0, 1, 2, 4, 6], p=[0.2, 0.3, 0.1, 0.2, 0.2])
                delivery_record = {
                    "match_id": match_id,
                    "inning": inn,
                    "over": 0,
                    "ball": ball_idx,
                    "batting_team": bat_team,
                    "bowling_team": bowl_team,
                    "batsman": bat_name,
                    "non_striker": non_st_name,
                    "bowler": bowl_name,
                    "batsman_runs": runs,
                    "extra_runs": 0,
                    "total_runs": runs,
                    "is_wicket": 0,
                    "dismissal_kind": "",
                    "player_dismissed": "",
                    "fielder": "",
                    "is_chase": 1 if inn == 4 else 0,
                    "is_super_over": 1
                }
                delivery_list.append(delivery_record)

if __name__ == "__main__":
    generate_synthetic_data()
