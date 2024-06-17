import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# Load data
data = pd.read_csv('https://raw.githubusercontent.com/Galfishman/StatsBomb-Data/main/pages/2024-05-24T06-27_export.csv')

# Streamlit app title and description
st.title("Players Pair or Trio")
st.markdown("""
**Note:** I suggest not selecting overlapping metrics. The algorithm ranks player pairs or trios by their rank in each metric, so it's unnecessary to select multiple metrics for the same action. For example, for Dribbling, I suggest picking either Dribble or Carry, not both.
""")

# Sidebar for league selection
leagues = data['competition_name'].unique().tolist()
selected_leagues = st.sidebar.multiselect("Select Leagues", options=leagues, default=leagues)

# Filter data by selected leagues
data = data[data['competition_name'].isin(selected_leagues)]

# Filter data by minimum minutes played
min_minutes_played = st.sidebar.slider("Filter by Minimum Minutes Played:", min_value=0, max_value=int(data['minutes'].max()), step=1, value=500)
data = data[(data['minutes'] >= min_minutes_played)]
data['player_team'] = data['player_name'] + ' - ' + data['team_name']

# Create player selection
players = st.multiselect("Select Players", options=data['player_team'].unique())

# Validate player selection
if len(players) < 3:
    st.error("Please select at least 3 players.")
else:
    # Option to choose pairs or trios
    selection_type = st.sidebar.radio("Select Pair or Trio", options=["Pair", "Trio"])

    # List of metrics to choose from (excluding the first 18 columns)
    metrics = data.columns[18:].tolist()

    # Create metric selection
    selected_metrics = st.multiselect("Select Metrics", options=metrics)

    if selected_metrics:
        # Filter data based on selected players and metrics
        filtered_data = data[data['player_team'].isin(players)][['player_team'] + selected_metrics]

        st.write(filtered_data)

        # Create combinations of players based on selection type
        if selection_type == "Pair":
            player_combinations = list(combinations(players, 2))
        else:
            player_combinations = list(combinations(players, 3))

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
