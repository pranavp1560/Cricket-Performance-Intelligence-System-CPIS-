-- ====================================================================
-- CRICKET PERFORMANCE INTELLIGENCE SYSTEM (CPIS) - ADVANCED SQL QUERIES
-- ====================================================================

-- 1. TOP PERFORMERS: Top 10 Players by Composite CPIS Rating
-- Exposes the highest impact, consistent, and pressure-resilient players overall.
SELECT 
    rank,
    player,
    role,
    matches,
    match_impact_score AS mis,
    consistency_index AS ci,
    pressure_performance_index AS ppi,
    cpis_rating
FROM player_overall_metrics
ORDER BY cpis_rating DESC
LIMIT 10;


-- 2. VENUE INSIGHTS: Best Players at Wankhede Stadium, Mumbai
-- Evaluates batting averages and strike rates specifically for the high-scoring Wankhede Stadium.
-- A. Best Batters (Min 3 matches played at Wankhede)
SELECT 
    player,
    bat_matches AS matches_at_venue,
    bat_runs AS total_runs,
    ROUND(bat_average, 2) AS batting_average,
    ROUND(bat_strike_rate, 2) AS strike_rate
FROM player_venue_stats
WHERE venue = 'Wankhede Stadium, Mumbai' AND bat_matches >= 3
ORDER BY bat_runs DESC, bat_average DESC
LIMIT 5;

-- B. Best Bowlers (Min 3 matches played at Wankhede)
SELECT 
    player,
    bowl_matches AS matches_at_venue,
    bowl_wickets AS total_wickets,
    ROUND(bowl_economy, 2) AS economy_rate,
    ROUND(bowl_average, 2) AS bowling_average
FROM player_venue_stats
WHERE venue = 'Wankhede Stadium, Mumbai' AND bowl_matches >= 3
ORDER BY bowl_wickets DESC, bowl_economy ASC
LIMIT 5;


-- 3. CONSISTENCY ANALYSIS: Most Reliable and Stable Performers
-- Selects players who maintain high impact scores with low variance (high CI) across a solid sample size.
SELECT 
    player,
    role,
    matches,
    ROUND(consistency_index, 2) AS consistency_score,
    ROUND(match_impact_score, 2) AS avg_match_impact
FROM player_overall_metrics
WHERE matches >= 15
ORDER BY consistency_index DESC, match_impact_score DESC
LIMIT 10;


-- 4. TEAM ANALYSIS: Most Valuable Player (MVP) for Each IPL Franchise
-- Uses window functions (SQLite supports window functions since version 3.25.0) to select the player
-- with the highest CPIS rating in each team.
WITH PlayerTeams AS (
    -- Link players to the team they represented in deliveries
    SELECT DISTINCT
        p.player,
        p.role,
        p.cpis_rating,
        -- We take the most frequent batting/bowling team of the player
        COALESCE(
            (SELECT batting_team FROM deliveries WHERE batsman = p.player GROUP BY batting_team ORDER BY COUNT(*) DESC LIMIT 1),
            (SELECT bowling_team FROM deliveries WHERE bowler = p.player GROUP BY bowling_team ORDER BY COUNT(*) DESC LIMIT 1)
        ) AS team_name
    FROM player_overall_metrics p
),
RankedPlayers AS (
    SELECT 
        team_name,
        player,
        role,
        cpis_rating,
        ROW_NUMBER() OVER (PARTITION BY team_name ORDER BY cpis_rating DESC) as rnk
    FROM PlayerTeams
    WHERE team_name IS NOT NULL AND team_name != ''
)
SELECT 
    team_name,
    player AS mvp_player,
    role,
    cpis_rating AS mvp_cpis_rating
FROM RankedPlayers
WHERE rnk = 1
ORDER BY cpis_rating DESC;


-- 5. DEATH OVERS SPECIALISTS: Bowlers Excelling under Death Overs Pressure (Overs 15-19)
-- Finds bowlers with the lowest economy rates and highest wicket counts in death overs.
SELECT 
    player,
    matches,
    ROUND(death_economy, 2) AS death_economy_rate,
    ROUND(death_strike_rate, 2) AS death_balls_per_wicket,
    wickets AS total_wickets
FROM player_overall_metrics
WHERE balls_bowled >= 150 AND death_economy > 0
ORDER BY death_economy ASC, death_strike_rate ASC
LIMIT 10;


-- 6. CHASE MASTERS: Batters with Highest Average in Successful Run Chases
-- Aggregates batsman statistics in successful run chases (inning == 2 and batting_team == winner) from raw deliveries.
SELECT 
    d.batsman AS player,
    COUNT(DISTINCT d.match_id) AS successful_chase_matches,
    SUM(d.batsman_runs) AS total_chase_runs,
    ROUND(CAST(SUM(d.batsman_runs) AS REAL) / SUM(CASE WHEN d.is_wicket = 1 AND d.player_dismissed = d.batsman THEN 1 ELSE 0 END), 2) AS chase_average,
    ROUND(CAST(SUM(d.batsman_runs) AS REAL) / COUNT(CASE WHEN d.extra_runs = 0 THEN 1 END) * 100, 2) AS chase_strike_rate
FROM deliveries d
JOIN matches m ON d.match_id = m.match_id
WHERE d.inning = 2 AND m.winner = d.batting_team
GROUP BY d.batsman
HAVING total_chase_runs >= 150 AND SUM(CASE WHEN d.is_wicket = 1 AND d.player_dismissed = d.batsman THEN 1 ELSE 0 END) > 0
ORDER BY chase_average DESC, total_chase_runs DESC
LIMIT 10;
