import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import plotly.express as px


data = pd.read_csv('https://raw.githubusercontent.com/Galfishman/StatsBomb-Data/main/pages/2024-05-24T06-27_export.csv')


st.title("Players Pair")
st.markdown("**Note:** I suggeest to not select a metrics that will over lap. Becuause the algoritm work on rank the players pairs by thier rank in each metric, so we dont need to select some metric to one actions. (For Example for Dribbling i suggeest picking Obv for Dribble and Carry.)")

min_minutes_played = st.sidebar.slider("Filter by Minimum Minutes Played:", min_value=0, max_value=int(data['minutes'].max()), step=1, value=500)
data = data[(data['minutes'] >= min_minutes_played)]
data['player_team'] = data['player_name'] + ' - ' + data['team_name']

# Create player selection
players = st.multiselect("Select Players", options=data['player_team'].unique())
# Optional: Split the selected values back into player_name and team_name if needed

# Validate player selection
if len(players) < 3 or len(players) > 6:
    st.error("Please select between 3 and 6 players.")
else:
    # List of metrics to choose from (excluding the first 18 columns)
    metrics = data.columns[18:].tolist()

    # Create metric selection
    selected_metrics = st.multiselect("Select Metrics", options=metrics)

    if selected_metrics:
        # Filter data based on selected players and metrics
        filtered_data = data[data['player_team'].isin(players)][['player_team'] + selected_metrics]

        st.write(filtered_data)

        # Create pairwise combinations of players
        player_pairs = list(combinations(players, 2))

        # Function to calculate and display scores for each pair
        def display_pair_scores(df, pairs, metrics):
            pair_scores = []
            for pair in pairs:
                pair_data = df[df['player_team'].isin(pair)]
                scores = pair_data[metrics].sum()
                scores_df = pd.DataFrame(scores).T
                scores_df['Pair'] = f"{pair[0]} & {pair[1]}"
                total_score = scores.sum()
                scores_df['Total'] = total_score
                pair_scores.append(scores_df)

            # Concatenate all scores into a single DataFrame
            all_scores_df = pd.concat(pair_scores, ignore_index=True)

            # Rank each metric
            for metric in metrics:
                all_scores_df[f"{metric}_rank"] = all_scores_df[metric].rank(ascending=False, method='min')

            # Calculate the rank sum for each pair
            rank_columns = [f"{metric}_rank" for metric in metrics]
            all_scores_df['Rank_Sum'] = all_scores_df[rank_columns].sum(axis=1)

            # Sort by rank sum (ascending) and total score (descending)
            all_scores_df = all_scores_df.sort_values(by=['Rank_Sum', 'Total'], ascending=[True, False]).reset_index(drop=True)

            return all_scores_df

        # Calculate scores for each pair
        pair_scores_df = display_pair_scores(filtered_data, player_pairs, selected_metrics)

        # Display the scores and ranks
        st.write(pair_scores_df)

        # Find the best pair
        best_pair_row = pair_scores_df.iloc[0]
        best_pair = best_pair_row['Pair']
        st.write(f"The best pair is {best_pair} with a rank sum of {best_pair_row['Rank_Sum']} and a total score of {best_pair_row['Total']}")
    else:
        st.warning("Please select at least one metric.")