import pandas as pd
import numpy as np
import joblib

# Load datasets
teams = pd.read_csv("./Data/teams.csv")  
merged = pd.read_csv("./Data/merged_data.csv") 
model = joblib.load("./Artifacts/trained_model.pkl")
scaler = joblib.load("./Artifacts/scaler.pkl") 


merged['GAME_DATE'] = pd.to_datetime(merged['GAME_DATE'])

def get_latest_team_stats_home(team_name):
    """Retrieve the latest average win rate and shooting percentage for a given team."""
    team_id = teams.loc[teams['ABBREVIATION'] == team_name, 'TEAM_ID'].values
    if len(team_id) == 0:
        raise ValueError(f"Team {team_name} not found!")
    team_id = team_id[0]
    
    # Get the most recent stats for the team
    team_stats = merged[merged['TEAM_ID_home'] == team_id].sort_values(by='GAME_DATE', ascending=False)
    
    if team_stats.empty:
        raise ValueError(f"No ranking data found for team {team_name}!")


    latest_stats = team_stats.iloc[0]  

    home_avg_pts = merged.groupby('TEAM_ID_home')['PTS_home'].transform(lambda x: x.rolling(5, min_periods=1).mean())

    return latest_stats['W_PCT_home'], latest_stats['FG_PCT_home'],home_avg_pts


def get_latest_team_stats_away(team_name):
    """Retrieve the latest average win rate and shooting percentage for a given team."""
    team_id = teams.loc[teams['ABBREVIATION'] == team_name, 'TEAM_ID'].values
    if len(team_id) == 0:
        raise ValueError(f"Team {team_name} not found!")
    team_id = team_id[0]
    
    # Get the most recent stats for the team
    team_stats = merged[merged['TEAM_ID_away'] == team_id].sort_values(by='GAME_DATE', ascending=False)
    
    if team_stats.empty:
        raise ValueError(f"No ranking data found for team {team_name}!")


    latest_stats = team_stats.iloc[0]  

    away_avg_pts = merged.groupby('TEAM_ID_home')['PTS_home'].transform(lambda x: x.rolling(5, min_periods=1).mean())


    return latest_stats['W_PCT_away'], latest_stats['FG_PCT_away'],away_avg_pts


def predict_match(home_team, away_team):
    """Predicts match outcome given only home and away team names."""
    
    # Get the latest statistics for both teams
    home_win_pct, home_fg_pct, home_avg_pts = get_latest_team_stats_home(home_team)
    away_win_pct, away_fg_pct, away_avg_pts = get_latest_team_stats_away(away_team)

    
    
    input_data = pd.DataFrame({
        'W_PCT_home': home_win_pct,
        'W_PCT_away': away_win_pct,
        'win_rate_diff': home_win_pct - away_win_pct,
        'FG_PCT_home': home_fg_pct,
        'FG_PCT_away': away_fg_pct,
        'fg_pct_diff': home_fg_pct - away_fg_pct,
        'home_avg_pts': home_avg_pts,
        'away_avg_pts': away_avg_pts
    })
    

    # Scale input if scaler was used during training
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_prob = model.predict_proba(input_data_scaled)[:, 1]  # Probability of home team winning
    
    # print winner witch is a team with more probability 
    winner = home_team if prediction[0] == 1 else away_team
    if prediction[0] == 0:
        prediction_prob = 1 - prediction_prob
    print(f"\n🏀 Prediction: {winner} will win with probability {prediction_prob[0]:.2f}")
    
    return winner, prediction_prob[0]

available_teams = teams['ABBREVIATION'].unique()
print("\n🏀 Available Teams:")
print(", ".join(available_teams))

home_team = input("\nEnter Home Team Abbreviation (e.g., LAL, BOS): ").upper()
away_team = input("Enter Away Team Abbreviation (e.g., GSW, MIA): ").upper()

predict_match(home_team, away_team)
