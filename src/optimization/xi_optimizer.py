import os
import sqlite3
import pandas as pd
import pulp

class PlayingXIOptimizer:
    def __init__(self, db_path="data/processed/cricket_intelligence.db"):
        self.db_path = db_path
        
    def load_players(self, squad_filter=None):
        """
        Loads players and their roles/ratings from the database.
        Optionally filters by team.
        """
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found at {self.db_path}.")
            
        conn = sqlite3.connect(self.db_path)
        # Load player overall metrics
        query = "SELECT player, role, cpis_rating, consistency_index, match_impact_score, pressure_performance_index FROM player_overall_metrics"
        players_df = pd.read_sql(query, conn)
        conn.close()
        
        # If squad filter is provided (e.g. team name), we could filter.
        # Note: we can map players to their team using the deliveries table
        if squad_filter:
            # Find players who played for this team
            conn = sqlite3.connect(self.db_path)
            team_players = pd.read_sql(
                "SELECT DISTINCT batsman AS player FROM deliveries WHERE batting_team = ?",
                conn, params=(squad_filter,)
            )["player"].tolist()
            conn.close()
            players_df = players_df[players_df["player"].isin(team_players)].copy()
            
        return players_df

    def optimize_xi(self, players_df, constraints=None):
        """
        Runs Mixed-Integer Linear Programming to find the optimal playing XI.
        Constraints dict format:
        {
            'min_openers': 2, 'max_openers': 2,
            'min_middle_order': 3, 'max_middle_order': 5,
            'min_all_rounders': 1, 'max_all_rounders': 3,
            'min_bowlers': 3, 'max_bowlers': 5,
            'min_wks': 1, 'max_wks': 2
        }
        """
        # Default constraints
        if not constraints:
            constraints = {
                'min_openers': 2, 'max_openers': 2,
                'min_middle_order': 3, 'max_middle_order': 5,
                'min_all_rounders': 1, 'max_all_rounders': 3,
                'min_bowlers': 3, 'max_bowlers': 5,
                'min_wks': 1, 'max_wks': 2
            }
            
        # Initialize optimization problem
        prob = pulp.LpProblem("Best_Playing_XI", pulp.LpMaximize)
        
        # Define decision variables
        player_names = players_df["player"].tolist()
        x = pulp.LpVariable.dicts("select", player_names, cat="Binary")
        
        # Objective Function: Maximize sum of CPIS Ratings of selected players
        ratings = dict(zip(players_df["player"], players_df["cpis_rating"]))
        prob += pulp.lpSum([ratings[p] * x[p] for p in player_names]), "Maximize_CPIS_Rating"
        
        # Constraints
        # 1. Total players = 11
        prob += pulp.lpSum([x[p] for p in player_names]) == 11, "Total_11_Players"
        
        # Helper to group players by role
        roles = dict(zip(players_df["player"], players_df["role"]))
        
        openers = [p for p in player_names if roles[p] == "Opener"]
        # In T20 cricket, middle order and finishers can be combined as middle order batters
        middle_order = [p for p in player_names if roles[p] in ["Middle Order", "Finisher"]]
        all_rounders = [p for p in player_names if roles[p] == "All-Rounder"]
        bowlers = [p for p in player_names if roles[p] == "Bowler"]
        wks = [p for p in player_names if roles[p] == "Wicketkeeper"]
        
        # 2. Wicketkeepers
        prob += pulp.lpSum([x[p] for p in wks]) >= constraints.get('min_wks', 1), "Min_WKs"
        prob += pulp.lpSum([x[p] for p in wks]) <= constraints.get('max_wks', 2), "Max_WKs"
        
        # 3. Openers
        prob += pulp.lpSum([x[p] for p in openers]) >= constraints.get('min_openers', 2), "Min_Openers"
        prob += pulp.lpSum([x[p] for p in openers]) <= constraints.get('max_openers', 2), "Max_Openers"
        
        # 4. Middle Order / Finishers
        prob += pulp.lpSum([x[p] for p in middle_order]) >= constraints.get('min_middle_order', 3), "Min_Middle_Order"
        prob += pulp.lpSum([x[p] for p in middle_order]) <= constraints.get('max_middle_order', 5), "Max_Middle_Order"
        
        # 5. All-Rounders
        prob += pulp.lpSum([x[p] for p in all_rounders]) >= constraints.get('min_all_rounders', 1), "Min_All_Rounders"
        prob += pulp.lpSum([x[p] for p in all_rounders]) <= constraints.get('max_all_rounders', 3), "Max_All_Rounders"
        
        # 6. Bowlers
        prob += pulp.lpSum([x[p] for p in bowlers]) >= constraints.get('min_bowlers', 3), "Min_Bowlers"
        prob += pulp.lpSum([x[p] for p in bowlers]) <= constraints.get('max_bowlers', 5), "Max_Bowlers"
        
        # Solve the problem
        # Disable solver logging for clean output
        solver = pulp.PULP_CBC_CMD(msg=False)
        status = prob.solve(solver)
        
        if status != pulp.LpStatusOptimal:
            print("LP Solver could not find an optimal solution. Checking feasibility and attempting relaxed optimization...")
            # Fallback: Greedy selection if constraints are too tight
            # Sort players by cpis_rating
            sorted_players = players_df.sort_values(by="cpis_rating", ascending=False)
            selected = []
            
            # Ensure at least 1 wk, 2 openers, 3 middle, 1 all-rounder, 3 bowlers
            # Wicketkeeper
            wk_candidates = sorted_players[sorted_players["role"] == "Wicketkeeper"]
            if len(wk_candidates) > 0:
                selected.append(wk_candidates.iloc[0])
            # Openers
            op_candidates = sorted_players[sorted_players["role"] == "Opener"]
            selected.extend([op_candidates.iloc[i] for i in range(min(2, len(op_candidates)))])
            # Middle Order / Finishers
            mid_candidates = sorted_players[sorted_players["role"].isin(["Middle Order", "Finisher"])]
            selected.extend([mid_candidates.iloc[i] for i in range(min(3, len(mid_candidates)))])
            # All Rounders
            ar_candidates = sorted_players[sorted_players["role"] == "All-Rounder"]
            selected.extend([ar_candidates.iloc[i] for i in range(min(1, len(ar_candidates)))])
            # Bowlers
            bowl_candidates = sorted_players[sorted_players["role"] == "Bowler"]
            selected.extend([bowl_candidates.iloc[i] for i in range(min(3, len(bowl_candidates)))])
            
            # Fill remaining spots to reach 11
            selected_names = [p["player"] for p in selected]
            remaining = sorted_players[~sorted_players["player"].isin(selected_names)]
            needed = 11 - len(selected)
            if needed > 0 and len(remaining) >= needed:
                selected.extend([remaining.iloc[i] for i in range(needed)])
                
            optimal_squad = pd.DataFrame(selected)
        else:
            # Extract selected players
            selected_players = [p for p in player_names if x[p].varValue == 1.0]
            optimal_squad = players_df[players_df["player"].isin(selected_players)].copy()
            
        # Sort squad for visual batting order
        # Openers -> Middle Order -> Wicketkeeper (if not opener) -> All rounders -> Bowlers
        role_priority = {"Opener": 0, "Middle Order": 1, "Finisher": 2, "Wicketkeeper": 3, "All-Rounder": 4, "Bowler": 5}
        optimal_squad["priority"] = optimal_squad["role"].map(role_priority)
        optimal_squad = optimal_squad.sort_values(by=["priority", "cpis_rating"], ascending=[True, False]).reset_index(drop=True)
        optimal_squad = optimal_squad.drop(columns=["priority"])
        
        # Suggestions: Captain and Vice Captain
        # Captain: highest cpis_rating
        captain = optimal_squad.sort_values(by="cpis_rating", ascending=False).iloc[0]["player"]
        # Vice-Captain: highest consistency index in remaining players
        vc_candidates = optimal_squad[optimal_squad["player"] != captain]
        vice_captain = vc_candidates.sort_values(by="consistency_index", ascending=False).iloc[0]["player"]
        
        return optimal_squad, captain, vice_captain

if __name__ == "__main__":
    optimizer = PlayingXIOptimizer()
    players_df = optimizer.load_players()
    squad, capt, vc = optimizer.optimize_xi(players_df)
    
    print("\nOptimized Best Playing XI Squad:")
    print(squad[["player", "role", "cpis_rating", "consistency_index"]].to_string())
    print(f"\nSuggested Captain: {capt} (Highest CPIS Rating)")
    print(f"Suggested Vice-Captain: {vc} (Most Consistent Player)")
