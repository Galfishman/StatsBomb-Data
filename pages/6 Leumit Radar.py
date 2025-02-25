import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from soccerplots.radar_chart import Radar
from mplsoccer import PyPizza
import matplotlib as mpl
from io import BytesIO
from mplsoccer import PyPizza, FontManager
import matplotlib as mpl
import math
from scipy import stats






###login
st.title("Leumit Radar")

# READ DATA
df = pd.read_csv('https://raw.githubusercontent.com/Galfishman/StatsBomb-Data/refs/heads/main/pages/Players%20Total.csv')


# Define the mapping of Short Position to Position 1
position_mapping = {
    'Center Backs': ['CD', 'RCD', 'LCD'],
    'Full Backs': ['RD', 'LD'],
    'Midfielders': ['CDM', 'RCDM', 'LCDM', 'LCM', 'RCM', 'CAM', 'RCAM', 'LCAM'],
    'Wingers': ['LM', 'RM', 'LAM', 'RAM'],
    'Strikers': ['CF', 'LCF', 'RCF']
}

st.sidebar.header("Please Filter Here:")
# Filter by position group
selected_position_group = st.sidebar.selectbox(
    "Filter by Position Group:",
    options=list(position_mapping.keys()),
)

# Filter by Minutes Played (Min)
min_minutes_played = st.sidebar.slider("Filter by Minimum Minutes Played:", min_value=0, max_value=int(df['Minutes played'].max()), step=1, value=0)

# Filter the DataFrame based on the selected position group and minimum minutes played
filtered_players = df[(df['Position'].isin(position_mapping[selected_position_group])) & (df['Minutes played'] >= min_minutes_played)]

# List of players based on the selected position group
players_list = filtered_players["Player"].unique()




Name = st.sidebar.selectbox(
    "Select the Player:",
    options=players_list,
)

Name2 = st.sidebar.selectbox(
    "Select other Player:",
    options=["League Average"] +filtered_players["Player"].unique().tolist(),
)


    
# List of all available parameters
all_params = list(df.columns[10:])

# Define preset parameters
attacking_params = [
    "Goals", "Assists", "Shots", "Shots on target", "Key passes",
    "Crosses", "Crosses accurate", "Passes into the penalty box",
    "Dribbles", "Dribbles successful", "xG", "Lost balls", "xA",
    "Progressive passes"
]
defensive_params = [
    "Defensive challenges", "Ball recoveries",
    "Ball recoveries in opponent's half", "Challenges won, %"
]

# Button selection for radar type
st.sidebar.subheader("Choose Radar Type:")
radar_type = st.sidebar.radio("Select Radar Type:", ["Attacking Radar", "Defensive Radar", "Create New Radar"])

# Select parameters based on radar type
if radar_type == "Attacking Radar":
    selected_params = attacking_params
elif radar_type == "Defensive Radar":
    selected_params = defensive_params
else:
    all_params = list(df.columns[10:])
    selected_params = st.sidebar.multiselect("Select Parameters:", options=all_params, default=attacking_params)

params = selected_params
st.sidebar.subheader("Metric Type:")
metric_type = st.sidebar.radio("Select Metric Type:", ["Total", "Per 90"])

# Adjust parameters based on the selected metric type
if metric_type == "Per 90":
    # Compute per 90 values
    for param in df.columns[10:]:
        if param != "Minutes played" and df[param].dtype in ['float64', 'int64']:
            df[param + " per 90"] = (df[param] / df["Minutes played"]) * 90
    # Add "per 90" suffix to selected parameters
    params = [param + " per 90" for param in params]
else:
    # Use total values (no changes needed)
    params = selected_params

# Re-filter the data
filtered_players = df[(df['Position'].isin(position_mapping[selected_position_group])) & (df['Minutes played'] >= min_minutes_played)]

# Calculate ranges dynamically based on metric type
ranges = []
for param in params:
    if metric_type == "Per 90":
        # Use only players who played enough minutes to avoid dividing by very small numbers
        valid_values = filtered_players[param].dropna()
    else:
        valid_values = df[param].dropna()
    
    # Determine the range based on valid values
    param_min = valid_values.min()
    param_max = valid_values.max()
    
    ranges.append((param_min, param_max))

# Recalculate a_values and b_values based on the selected metric type
a_values = filtered_players.loc[filtered_players['Player'] == Name, params].values.flatten().tolist()

if Name2 == "League Average":
    league_average_values = filtered_players[filtered_players['Player'] != Name][params].mean().tolist()
    b_values = league_average_values
else:
    b_values = filtered_players.loc[filtered_players['Player'] == Name2, params].values.flatten().tolist()

# Update the title dictionary with minutes played
minutes_name = "Minutes played"
minutes_player1 = filtered_players.loc[filtered_players['Player'] == Name, minutes_name].values[0]

Position_name = "Position"
Position_name1 = filtered_players.loc[filtered_players['Player'] == Name, Position_name].values[0]

Team_name = "Team"
team_name1 = filtered_players.loc[filtered_players['Player'] == Name, Team_name].values[0]

title = dict(
    title_name=f"{Name}\n{'Team: ' + team_name1}\n{Position_name1}\n{minutes_player1} Minutes Played",
    title_color='yellow',
    title_name_2=f"{Name2 if Name2 != 'League Average' else 'League Average'}",
    title_color_2='blue',
    title_fontsize=12,
)

# RADAR PLOT
radar = Radar(
    background_color="#121212",
    patch_color="#28252C",
    label_color="#FFFFFF",
    label_fontsize=10,
    range_color="#FFFFFF"
)

# plot radar
fig, ax = radar.plot_radar(
    ranges=ranges,
    params=params,
    values=[a_values, b_values],
    radar_color=['yellow', 'blue'],
    edgecolor="#222222",
    zorder=2,
    linewidth=1,
    title=title,
    alphas=[0.4, 0.4],
    compare=True
)

mpl.rcParams['figure.dpi'] = 1500


##########################################
##########################################
##########################################
##########################################


player_data = filtered_players.loc[filtered_players['Player'] == Name, selected_params].iloc[0]
values = [math.floor(stats.percentileofscore(filtered_players[param], player_data[param])) for param in selected_params]

# Create a table to display the statistic names and values
table_data = {'Statistic': selected_params, 'Value': player_data[selected_params]}
table_df = pd.DataFrame(table_data)
### PRECEPLIE PIZZA

baker = PyPizza(
    params=params,                  # list of parameters
    straight_line_color="white",  # color for straight lines
    straight_line_lw=2,             # linewidth for straight lines
    last_circle_lw=3,
    last_circle_color= 'white',                              # linewidth of last circle
    other_circle_lw=1,   
    other_circle_color='grey',           # linewidth for other circles
    other_circle_ls="-."            # linestyle for other circles
)

fig2, ax2 = baker.make_pizza(
    values,              # list of values
    figsize=(10, 10),      # adjust figsize according to your need
    param_location=105,  # where the parameters will be added
    kwargs_slices=dict(
        facecolor="cornflowerblue", edgecolor='white',
        zorder=2, linewidth=3
    ),                   # values to be used when plotting slices
    kwargs_params=dict(
        color="white", fontsize=10,
        va="center"
    ),                   # values to be used when adding parameter
    kwargs_values=dict(
        color="white", fontsize=12,
        zorder=3,
        bbox=dict(
            edgecolor="#000000", facecolor="cornflowerblue",
            boxstyle="round,pad=0.2", lw=1
        )
    )                    # values to be used when adding parameter-values
)

# Change background colors
fig2.patch.set_facecolor('#121212')  # Light grey figure background
ax2.set_facecolor('#121212')          # Light grey axes background


# Calculate the width and height of the title box
title = f"{Name} Percentile Rank\n{'Compare to all'} {selected_position_group} {'in'} {'Liga Leumit'}"
title_bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#000000", lw=1)
# Add the title box
fig2.text(0.515, 0.97, title, size=18, ha="center", color="#000000", bbox=title_bbox_props)



param_value_text = f"{Name}  Values\n"
for param, value in zip(selected_params, player_data):
    param_value_text += f"{param} - {value}\n"

# Adjust the vertical position of the param_value_text box based on the number of parameters
num_params = len(selected_params)
param_value_y = -0.11 - (num_params * 0.015)  # Adjust the value as needed to control the vertical position


# Display plots side by side with a gap
col1, col2 = st.columns([1, 1], gap='large') 
with col1:
    st.header("Values Radar ")
    st.pyplot(fig)
with col2:
    st.header("Percentile Rank")
    st.pyplot(fig2)






head_to_head_df = pd.DataFrame({
    'Player': [Name, Name2],
    **{param: [a_values[i], b_values[i]] for i, param in enumerate(params)}
})

# Transpose the DataFrame to have parameters as rows
head_to_head_df_transposed = head_to_head_df.set_index('Player').T

# Identify the highest value for each parameter across both players
max_values = head_to_head_df_transposed.max()

# Highlight the highest value in each row across both players
highlighted_df = head_to_head_df_transposed.style.format("{:.2f}").apply(lambda row: ['background-color: grey' if val == row.max() else '' for val in row], axis=1)

st.header("Head-to-Head Comparison")
st.table(highlighted_df)
