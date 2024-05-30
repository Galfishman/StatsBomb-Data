import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/Users/galfishman/Desktop/2024-05-24T06-27_export.csv')
data.columns = data.columns.str.replace('_', ' ')
# Round the 'minutes' column
data['minutes'] = data['minutes'].round()

# Create Streamlit interface
st.title('Player Similarity Finder')

# Filter by minutes range
min_minutes, max_minutes = st.sidebar.slider('Filter by Minutes Played:', min_value=int(data['minutes'].min()), max_value=int(data['minutes'].max()), value=(int(data['minutes'].min()), int(data['minutes'].max())), step=1)

# Filter data to include only the selected metrics and within the specified minutes range
selected_data = data[(data['minutes'] >= min_minutes) & (data['minutes'] <= max_minutes)]

# Player selection from filtered data
player = st.selectbox('Select a player:', selected_data['player name'].unique())

# Metric selection
metrics = st.multiselect('Select metrics:', data.columns.tolist()[18:])  # Exclude player_name from metric options

# Number of similar players
num_similar = st.slider('Number of similar players:', 5, 10, 5)

# Initialize dictionary to store weights
weights = {}

# If metrics are selected, show sliders for each metric to assign weights
if metrics:
    st.write("Assign weights to each metric:")
    for metric in metrics:
        weights[metric] = st.sidebar.slider(f'Weight for {metric}', 0.0, 1.0, 0.5)

# Filter data to include only the selected metrics
selected_data = selected_data[['player name', 'minutes', 'primary position'] + metrics]

# Handle NaN values by filling them with 0
selected_data = selected_data.fillna(0)

# Round 'minutes' column
selected_data['minutes'] = selected_data['minutes'].round()

# Get the index of the selected player and calculate similarity
try:
    if player and metrics:
        player_idx = selected_data[selected_data['player name'] == player].index[0]

        # Calculate weighted Euclidean distance
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(selected_data[metrics])
        weighted_data = data_scaled * np.array([weights[metric] for metric in metrics])
        distances = cdist(weighted_data, [weighted_data[player_idx]], metric='euclidean').flatten()

        # Get the indices of the most similar players
        similar_indices = distances.argsort()[1:num_similar+1]

        # Create a dataframe to display the results
        similar_players = selected_data.iloc[similar_indices]
        similar_players['distance'] = distances[similar_indices]

        # Select the columns to display
        display_columns = ['player name', 'minutes', 'primary position', 'distance'] + metrics

        # Add the chosen player to the top of the table
        chosen_player_row = selected_data[selected_data['player name'] == player]
        chosen_player_row['distance'] = 0.0  # Chosen player has 0 distance with themselves
        display_df = pd.concat([chosen_player_row[display_columns], similar_players[display_columns]], ignore_index=True)

        # Sort the dataframe by distance for the bar chart
        display_df_sorted = display_df.sort_values(by='distance')

        # Display the table with formatted metrics and custom styling
        st.write("### Similar Players (Transposed)")
        display_df_transposed = display_df.set_index('player name').transpose()

        # Round metrics to 3 decimal places and format as strings without trailing zeros
        display_df_rounded = display_df_transposed.applymap(lambda x: f'{x:.3f}' if isinstance(x, (int, float)) else x)

        # Styling the table
        st.write(display_df_rounded.style.set_properties(**{'font-weight': 'bold', 'color': 'white', 'background-color': 'black'}))

        # Display a bar chart of distances using matplotlib
        st.write("### Similarity Bar Chart")
        fig, ax = plt.subplots()
        ax.bar(display_df_sorted['player name'], display_df_sorted['distance'])
        ax.set_xlabel('Player Name')
        ax.set_ylabel('Distance')
        ax.set_title('Similarity Bar Chart')
        ax.tick_params(axis='x', rotation=45)
        ax.text(0.5, 0.96, 'Least is more similar', transform=ax.transAxes, ha='center')

        st.pyplot(fig)

except IndexError:
    st.warning(f"The selected player '{player}' is not found in the filtered data. Please adjust the filter or select another player.")

if not player:
    st.warning('Please select a player.')

if not metrics:
    st.warning('Please select metrics.')
