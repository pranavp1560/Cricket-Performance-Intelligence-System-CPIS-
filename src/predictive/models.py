import os
import sqlite3
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

class CPISPredictiveModel:
    def __init__(self, db_path="data/processed/cricket_intelligence.db", models_dir="models"):
        self.db_path = db_path
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
    def build_season_dataset(self):
        """
        Aggregates player stats by season and builds the shift dataset
        where Features = Season S stats, Target = Season S+1 stats.
        """
        conn = sqlite3.connect(self.db_path)
        pm_df = pd.read_sql("SELECT * FROM player_match_features", conn)
        conn.close()
        
        print("Aggregating player stats by season...")
        # Group by player and season
        season_groups = pm_df.groupby(["player", "season"])
        
        season_stats = []
        
        # Max match impact score across all rows to normalize MIS per season
        max_impact = pm_df["match_impact_score"].max()
        
        for (player, season), data in season_groups:
            tot_runs = data["runs"].sum()
            tot_balls = data["balls_faced"].sum()
            tot_outs = data["is_out"].sum()
            bat_avg = tot_runs / tot_outs if tot_outs > 0 else tot_runs
            sr = (tot_runs / tot_balls * 100) if tot_balls > 0 else 0
            
            tot_wickets = data["wickets"].sum()
            tot_balls_bowled = data["balls_bowled"].sum()
            tot_runs_conceded = data["runs_conceded"].sum()
            tot_overs = tot_balls_bowled / 6.0
            econ = (tot_runs_conceded / tot_overs) if tot_overs > 0 else 0
            bowl_avg = tot_runs_conceded / tot_wickets if tot_wickets > 0 else tot_runs_conceded
            
            # Match Impact Score for this season
            mean_impact = data["match_impact_score"].mean()
            mis = min(100.0, (mean_impact / max_impact) * 100)
            
            # Consistency for this season
            std_impact = data["match_impact_score"].std()
            cv = (std_impact / mean_impact) if (mean_impact > 0 and not pd.isna(std_impact)) else 1.2
            ci = 100.0 * (1.0 - min(1.0, cv / 1.2))
            
            # Pressure index for this season (approximate based on match context mean)
            ppi = data["chase_difficulty"].mean() * 40.0 + (data["pressure_runs"].sum() / max(1.0, tot_runs)) * 40.0
            ppi = min(100.0, max(0.0, ppi))
            
            # Composite CPIS rating for this season
            bat_raw = min(100.0, (tot_runs / len(data)) * (sr / 120.0) * 1.5)
            bowl_raw = min(100.0, (tot_wickets / len(data)) * 30.0 * max(0.1, (11.0 - econ)/4.0)) if tot_overs > 0 else 0.0
            rating = 0.35 * mis + 0.20 * ci + 0.20 * ppi + 0.15 * bat_raw + 0.10 * bowl_raw
            
            season_stats.append({
                "player": player,
                "season": season,
                "runs": tot_runs,
                "balls_faced": tot_balls,
                "batting_average": bat_avg,
                "strike_rate": sr,
                "wickets": tot_wickets,
                "balls_bowled": tot_balls_bowled,
                "economy": econ,
                "bowling_average": bowl_avg,
                "mis": mis,
                "ci": ci,
                "ppi": ppi,
                "rating": rating
            })
            
        season_df = pd.DataFrame(season_stats)
        
        # Build the shift dataset
        print("Constructing predictive dataset features and labels...")
        features_list = []
        
        # We group by player
        for player, p_data in season_df.groupby("player"):
            p_data = p_data.sort_values(by="season")
            seasons = p_data["season"].tolist()
            
            for i in range(len(seasons) - 1):
                curr_season = seasons[i]
                next_season = seasons[i+1]
                
                curr_row = p_data[p_data["season"] == curr_season].iloc[0]
                next_row = p_data[p_data["season"] == next_season].iloc[0]
                
                features_list.append({
                    "player": player,
                    "curr_season": curr_season,
                    "next_season": next_season,
                    # Features
                    "runs_curr": curr_row["runs"],
                    "strike_rate_curr": curr_row["strike_rate"],
                    "batting_avg_curr": curr_row["batting_average"],
                    "wickets_curr": curr_row["wickets"],
                    "economy_curr": curr_row["economy"],
                    "bowling_avg_curr": curr_row["bowling_average"],
                    "mis_curr": curr_row["mis"],
                    "ci_curr": curr_row["ci"],
                    "ppi_curr": curr_row["ppi"],
                    "rating_curr": curr_row["rating"],
                    # Targets
                    "runs_next": next_row["runs"],
                    "wickets_next": next_row["wickets"],
                    "rating_next": next_row["rating"]
                })
                
        df_ml = pd.DataFrame(features_list)
        return season_df, df_ml

    def train_and_evaluate(self):
        """
        Trains RF, XGBoost, and LightGBM models for runs, wickets, and rating targets.
        Saves the best model and returns a metrics dictionary.
        """
        season_df, df_ml = self.build_season_dataset()
        if len(df_ml) < 20:
            print("Warning: ML Dataset size too small. Skipping model training.")
            return {}
            
        feature_cols = [
            "runs_curr", "strike_rate_curr", "batting_avg_curr",
            "wickets_curr", "economy_curr", "bowling_avg_curr",
            "mis_curr", "ci_curr", "ppi_curr", "rating_curr"
        ]
        
        targets = ["runs_next", "wickets_next", "rating_next"]
        
        X = df_ml[feature_cols]
        results_report = {}
        
        # We will train separate models for each target
        best_models = {}
        
        for target in targets:
            y = df_ml[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 1. Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            
            # 2. XGBoost
            xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.08, random_state=42)
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            
            # 3. LightGBM
            lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.08, random_state=42, verbose=-1)
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict(X_test)
            
            # Evaluate
            models_eval = {
                "Random Forest": rf_pred,
                "XGBoost": xgb_pred,
                "LightGBM": lgb_pred
            }
            
            target_metrics = {}
            for name, pred in models_eval.items():
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                mae = mean_absolute_error(y_test, pred)
                r2 = r2_score(y_test, pred)
                target_metrics[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
                
            results_report[target] = target_metrics
            
            # Determine best model based on R2
            best_name = max(target_metrics, key=lambda k: target_metrics[k]["R2"])
            best_model_obj = {"Random Forest": rf, "XGBoost": xgb_model, "LightGBM": lgb_model}[best_name]
            
            best_models[target] = (best_name, best_model_obj)
            
            # Save the best model for this target
            model_path = f"{self.models_dir}/predictor_{target}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(best_model_obj, f)
            print(f"Best model for {target} is {best_name} (R2={target_metrics[best_name]['R2']:.3f}). Saved to {model_path}.")
            
        # ------------------ GENERATE FUTURE PREDICTIONS (2026) ------------------
        # Get latest season (2025) stats for all players
        latest_season = season_df["season"].max()
        print(f"Generating 2026 Season Predictions using latest season ({latest_season}) stats...")
        
        latest_stats = season_df[season_df["season"] == latest_season].copy()
        
        # Construct features for prediction
        pred_features = latest_stats.rename(columns={
            "runs": "runs_curr",
            "strike_rate": "strike_rate_curr",
            "batting_average": "batting_avg_curr",
            "wickets": "wickets_curr",
            "economy": "economy_curr",
            "bowling_average": "bowling_avg_curr",
            "mis": "mis_curr",
            "ci": "ci_curr",
            "ppi": "ppi_curr",
            "rating": "rating_curr"
        })[feature_cols]
        
        predictions_df = pd.DataFrame({"player": latest_stats["player"]})
        
        for target in targets:
            _, model = best_models[target]
            preds = model.predict(pred_features)
            # Ensure predictions are physically realistic (no negative runs/wickets, cap ratings at 100)
            if "runs" in target:
                preds = np.clip(preds, 0, None).round().astype(int)
            elif "wickets" in target:
                preds = np.clip(preds, 0, None).round().astype(int)
            else:
                preds = np.clip(preds, 0, 100).round(2)
                
            predictions_df[target.replace("_next", "_predicted_2026")] = preds
            
        # Join with latest ratings for display comparison
        predictions_df = predictions_df.merge(
            latest_stats[["player", "runs", "wickets", "rating"]].rename(columns={
                "runs": "runs_actual_2025",
                "wickets": "wickets_actual_2025",
                "rating": "rating_actual_2025"
            }),
            on="player"
        )
        
        # Save to SQLite
        conn = sqlite3.connect(self.db_path)
        predictions_df.to_sql("player_predictions", conn, if_exists="replace", index=False)
        conn.close()
        
        # Save evaluation metrics as JSON/text summary for app usage
        metrics_path = f"{self.models_dir}/evaluation_metrics.pkl"
        with open(metrics_path, "wb") as f:
            pickle.dump(results_report, f)
            
        print("Model training and predictions run complete.")
        return results_report

if __name__ == "__main__":
    pm = CPISPredictiveModel()
    report = pm.train_and_evaluate()
    
    # Print metrics report
    print("\nModel Evaluation Report:")
    for target, metrics in report.items():
        print(f"\nTarget Variable: {target}")
        for model, vals in metrics.items():
            print(f"  * {model} -> R2: {vals['R2']:.3f}, MAE: {vals['MAE']:.2f}, RMSE: {vals['RMSE']:.2f}")
