import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('https://raw.githubusercontent.com/Galfishman/StatsBomb-Data/main/pages/2024-05-24T06-27_export.csv')
data.columns = data.columns.str.replace('_', ' ')
# Round the 'minutes' column
data['minutes'] = data['minutes'].round()

# Create Streamlit interface
st.title('Player Similarity Finder')

# Define position similarity mapping
position_mapping = {
    'Center Backs': ['Centre Back', 'Left Centre Back', 'Right Centre Back'],
    'Full Backs': ['Left Back', 'Left Wing Back', 'Right Back', 'Right Wing Back'],
    'Midfielders': ['Left Defensive Midfielder', 'Left Centre Midfielder', 'Centre Attacking Midfielder', 'Centre Defensive Midfielder', 'Left Midfielder', 'Right Centre Midfielder', 'Right Defensive Midfielder', 'Right Midfielder'],
    'Wingers': ['Left Attacking Midfielder', 'Left Wing', 'Right Attacking Midfielder', 'Right Wing'],
    'Strikers': ['Centre Forward', 'Left Centre Forward', 'Right Centre Forward'],
    'GK': ['Goalkeeper']
}

# Filter by minutes range
min_minutes, max_minutes = st.sidebar.slider('Filter by Minutes Played:', 
                                             min_value=int(data['minutes'].min()), 
                                             max_value=int(data['minutes'].max()), 
                                             value=(int(data['minutes'].min()), int(data['minutes'].max())), 
                                             step=1)

# Filter data to include only the selected metrics and within the specified minutes range
selected_data = data[(data['minutes'] >= min_minutes) & (data['minutes'] <= max_minutes)]

# Initialize session state for player selection
if 'selected_player' not in st.session_state:
    st.session_state.selected_player = None

# Player selection from filtered data
if selected_data.empty:
    st.warning("No players found within the specified minutes range. Please adjust the filter.")
else:
    player_names = selected_data['player name'].unique()

    # Set the default player selection to the previously selected player, if available
    if st.session_state.selected_player in player_names:
        default_index = list(player_names).index(st.session_state.selected_player)
    else:
        default_index = 0

    player = st.selectbox('Select a player:', player_names, index=default_index)

    # Update session state with the selected player
    st.session_state.selected_player = player

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

    # Handle NaN values by filling them with 0
    selected_data = selected_data.fillna(0)

    # Round 'minutes' column
    selected_data['minutes'] = selected_data['minutes'].round()

    # Get the index of the selected player and calculate similarity
    try:
        if player and metrics:
            player_row = selected_data[selected_data['player name'] == player]
            player_idx = player_row.index[0]

            # Ensure metrics are scaled and weighted appropriately
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(selected_data[metrics])

            # Apply weights with handling of zero values
            weighted_data = data_scaled * np.array([max(weights[metric], 1e-6) for metric in metrics])

            # Get the primary position of the selected player
            player_primary_position = player_row['primary position'].values[0]

            # Get the positions to include in the comparison based on the mapping
            similar_positions = []
            for key, value in position_mapping.items():
                if player_primary_position in value:
                    similar_positions = value
                    break

            # Filter data to only include players with similar positions
            position_filtered_data = selected_data[selected_data['primary position'].isin(similar_positions) | selected_data['secondary position'].isin(similar_positions)]
            position_filtered_idx = position_filtered_data.index

            # Re-calculate scaled and weighted data for position-filtered data
            data_scaled_filtered = data_scaled[position_filtered_idx]
            weighted_data_filtered = weighted_data[position_filtered_idx]

            # Calculate distances
            distances = cdist(weighted_data_filtered, [weighted_data_filtered[position_filtered_idx.get_loc(player_idx)]], metric='euclidean').flatten()

            # Get the indices of the most similar players
            similar_indices = distances.argsort()[1:num_similar + 1]

            # Create a dataframe to display the results
            similar_players = position_filtered_data.iloc[similar_indices]
            similar_players['distance'] = distances[similar_indices]

            # Select the columns to display
            display_columns = ['player name', 'minutes', 'primary position', 'distance'] + metrics
            if 'secondary position' in selected_data.columns:
                display_columns.append('secondary position')

            # Add the chosen player to the top of the table
            chosen_player_row = player_row.copy()
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

            # Add Google search links as buttons
            st.write("### Google Search Links for Similar Players")
            for player_name in display_df_sorted['player name']:
                player_url = f"https://www.google.com/search?q={player_name.replace(' ', '+')}+football"
                if st.button(f'Search {player_name} on Google'):
                    st.write(f"[Search {player_name} on Google]({player_url})")

    except IndexError:
        st.warning(f"The selected player '{player}' is not found in the filtered data. Please adjust the filter or select another player.")

    if not player:
        st.warning('Please select a player.')

    if not metrics:
        st.warning('Please select metrics.')
