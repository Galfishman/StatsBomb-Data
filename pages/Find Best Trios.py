import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# Load data
data = pd.read_csv('https://raw.githubusercontent.com/Galfishman/StatsBomb-Data/main/pages/2024-05-24T06-27_export.csv')

# Streamlit app title and description
st.title("Best Player Pairs or Trios")
st.markdown("""
**Note:** I suggest not selecting overlapping metrics. The algorithm ranks player pairs or trios by their rank in each metric, so it's unnecessary to select multiple metrics for the same action. For example, for Dribbling, I suggest picking either Dribble or Carry, not both.
""")

# Sidebar for competition selection
competitions = data['competition_name'].unique().tolist()
selected_competitions = st.sidebar.multiselect("Select Competitions", options=competitions, default=competitions)

# Filter data by selected competitions
data = data[data['competition_name'].isin(selected_competitions)]

# Filter data by minimum minutes played
min_minutes_played = st.sidebar.slider("Filter by Minimum Minutes Played:", min_value=0, max_value=int(data['minutes'].max()), step=1, value=500)
data = data[(data['minutes'] >= min_minutes_played)]

# Define position similarity mapping
position_mapping = {
    'Center Backs': ['Centre Back', 'Left Centre Back', 'Right Centre Back'],
    'Full Backs': ['Left Back', 'Left Wing Back', 'Right Back', 'Right Wing Back'],
    'Midfielders': ['Left Defensive Midfielder', 'Left Centre Midfielder', 'Centre Attacking Midfielder', 'Centre Defensive Midfielder', 'Left Midfielder', 'Right Centre Midfielder', 'Right Defensive Midfielder', 'Right Midfielder'],
    'Wingers': ['Left Attacking Midfielder', 'Left Wing', 'Right Attacking Midfielder', 'Right Wing'],
    'Strikers': ['Centre Forward', 'Left Centre Forward', 'Right Centre Forward'],
    'GK': ['Goalkeeper']
}

# Position selection
positions = list(position_mapping.keys())
selected_positions = st.sidebar.multiselect("Select Positions", options=positions, default=positions)

# Filter data by selected positions
selected_position_values = [pos for group in selected_positions for pos in position_mapping[group]]
data = data[data['position_name'].isin(selected_position_values)]
data['player_team'] = data['player_name'] + ' - ' + data['team_name']

# Option to choose pairs or trios
selection_type = st.sidebar.radio("Select Pair or Trio", options=["Pair", "Trio"])

# List of metrics to choose from (excluding the first 18 columns)
metrics = data.columns[18:].tolist()

# Create metric selection
selected_metrics = st.multiselect("Select Metrics", options=metrics)

if selected_metrics:
    # Filter data based on selected players and metrics
    filtered_data = data[['player_team'] + selected_metrics]

    st.write(filtered_data)

    # Create combinations of players based on selection type
    if selection_type == "Pair":
        player_combinations = list(combinations(filtered_data['player_team'].unique(), 2))
    else:
        player_combinations = list(combinations(filtered_data['player_team'].unique(), 3))

    # Function to calculate and display scores for each combination
    def display_combination_scores(df, combinations, metrics):
        combination_scores = []
        for combination in combinations:
            combination_data = df[df['player_team'].isin(combination)]
            scores = combination_data[metrics].mean()
            scores_df = pd.DataFrame(scores).T
            combination_names = " & ".join(combination)
            scores_df['Combination'] = combination_names
            combination_scores.append(scores_df)

        # Concatenate all scores into a single DataFrame
        all_scores_df = pd.concat(combination_scores, ignore_index=True)

        # Rank each metric
        for metric in metrics:
            all_scores_df[f"{metric}_rank"] = all_scores_df[metric].rank(ascending=False, method='min')

        # Calculate the rank sum for each combination
        rank_columns = [f"{metric}_rank" for metric in metrics]
        all_scores_df['Rank_Sum'] = all_scores_df[rank_columns].sum(axis=1)

        # Sort by rank sum (ascending)
        all_scores_df = all_scores_df.sort_values(by='Rank_Sum', ascending=True).reset_index(drop=True)

        return all_scores_df

    # Calculate scores for each combination
    combination_scores_df = display_combination_scores(filtered_data, player_combinations, selected_metrics)

    # Display the scores and ranks
    st.write(combination_scores_df)

    # Find the best combination
    best_combination_row = combination_scores_df.iloc[0]
    best_combination = best_combination_row['Combination']
    st.write(f"The best combination is {best_combination} with a rank sum of {best_combination_row['Rank_Sum']}")
else:
    st.warning("Please select at least one metric.")
