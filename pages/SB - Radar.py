import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from soccerplots.radar_chart import Radar
from mplsoccer import PyPizza
import matplotlib as mpl
from io import BytesIO
from mplsoccer import PyPizza, FontManager
import matplotlib as mpl
import requests as req
from requests_cache import CachedSession
from statsbombpy.config import CACHED_CALLS_SECS, HOSTNAME, VERSIONS
import math
from scipy import stats
import os
import requests
import matplotlib.pyplot as plt
from tempfile import mkdtemp


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = 'Palatino Linotype'

# Set your login credentials
credentials = st.secrets["credentials"]
# Set your login credentials
credentials = {"user": st.secrets.credentials.user, "passwd": st.secrets.credentials.passwd}

# Set environment variables for authentication (optional)
os.environ['SB_USERNAME'] = credentials['user']
os.environ['SB_PASSWORD'] = credentials['passwd']

session = CachedSession(cache_name=mkdtemp(), backend="sqlite", expire_after=CACHED_CALLS_SECS)


# The first thing we have to do is open the data. We use a parser SBopen available in mplsoccer.


def get_resource(url: str, creds: dict) -> list:
    auth = req.auth.HTTPBasicAuth(credentials["user"], credentials["passwd"])
    resp = session.get(url, auth=auth)
    if resp.status_code != 200:
        print(f"{url} -> {resp.status_code}")
        resp = []
    else:
        resp = resp.json()
    return resp

def competitions() -> list:
    url = f"https://data.statsbomb.com/api/v4/competitions"
    competitions_data = get_resource(url, {credentials["user"], credentials["passwd"]})
    competition_list = []
    for comp in competitions_data:
        print("Current competition:", comp)
        competition_dict = {
            "competition_name": comp["competition_name"],
            "country_name": comp["country_name"],
            "competition_id": comp["competition_id"],
            "season_name": comp["season_name"],
            "season_id": comp["season_id"]
        }
        competition_list.append(competition_dict)
    return competition_list

def seasons(competition_id: int) -> list:
    url = f"https://data.statsbomb.com/api/v4/competitions/{competition_id}/seasons"
    seasons_data = get_resource(url, {credentials["user"], credentials["passwd"]})
    season_list = []
    for season in seasons_data:
        season_dict = {
            "season_name": season["season_name"],
            "season_id": season["season_id"]
        }
        season_list.append(season_dict)
    return season_list

# Get list of competitions
competition_list = competitions()
countries = sorted(set(comp["country_name"] for comp in competition_list))
selected_country = st.sidebar.selectbox("Select Country", ["All"] + countries)


competitions_filtered = [comp for comp in competition_list if selected_country == "All" or comp["country_name"] == selected_country]
competition_names = sorted(set(comp["competition_name"] for comp in competitions_filtered))
selected_competition = st.sidebar.selectbox("Select Competition", ["All"] + competition_names)
selected_competition_id = next((comp["competition_id"] for comp in competitions_filtered if comp["competition_name"] == selected_competition), None)

# Get list of seasons for the selected competition
seasons_filtered = [comp for comp in competitions_filtered if selected_competition == "All" or comp["competition_name"] == selected_competition]
season_names = sorted(set(comp["season_name"] for comp in seasons_filtered))
selected_season = st.sidebar.selectbox("Select Season", ["All"] + season_names)
selected_season_id = next((comp["season_id"] for comp in seasons_filtered if comp["season_name"] == selected_season), None)


# Get player statistics for the selected competition and season
if selected_competition_id is not None and selected_season_id is not None:
    url = f"https://data.statsbomb.com/api/v4/competitions/{selected_competition_id}/seasons/{selected_season_id}/player-stats"
    df = get_resource(url, credentials)
else:
    st.warning("Please select both a competition and a season.")

df = pd.DataFrame(df)

columns_to_drop = [
    'player_season_average_space_received_in',
    'player_season_average_fhalf_space_received_in',
    'player_season_average_f3_space_received_in',
    'player_season_ball_receipts_in_space_10_ratio',
    'player_season_ball_receipts_in_space_2_ratio',
    'player_season_ball_receipts_in_space_5_ratio',
    'player_season_fhalf_ball_receipts_in_space_10_ratio',
    'player_season_fhalf_ball_receipts_in_space_2_ratio',
    'player_season_fhalf_ball_receipts_in_space_5_ratio',
    'player_season_f3_ball_receipts_in_space_10_ratio',
    'player_season_f3_ball_receipts_in_space_2_ratio',
    'player_season_f3_ball_receipts_in_space_5_ratio',
    'player_season_lbp_90',
    'player_season_lbp_completed_90',
    'player_season_lbp_ratio',
    'player_season_fhalf_lbp_completed_90',
    'player_season_fhalf_lbp_ratio',
    'player_season_f3_lbp_completed_90',
    'player_season_f3_lbp_ratio',
    'player_season_fhalf_lbp_90',
    'player_season_f3_lbp_90',
    'player_season_obv_lbp_90',
    'player_season_fhalf_obv_lbp_90',
    'player_season_f3_obv_lbp_90',
    'player_season_lbp_pass_ratio',
    'player_season_fhalf_lbp_pass_ratio',
    'player_season_f3_lbp_pass_ratio',
    'player_season_lbp_received_90',
    'player_season_fhalf_lbp_received_90',
    'player_season_f3_lbp_received_90',
    'player_season_average_lbp_to_space_distance',
    'player_season_fhalf_average_lbp_to_space_distance',
    'player_season_f3_average_lbp_to_space_distance',
    'player_season_lbp_to_space_10_received_90',
    'player_season_fhalf_lbp_to_space_10_received_90',
    'player_season_f3_lbp_to_space_10_received_90',
    'player_season_lbp_to_space_2_received_90',
    'player_season_fhalf_lbp_to_space_2_received_90',
    'player_season_f3_lbp_to_space_2_received_90',
    'player_season_lbp_to_space_5_received_90',
    'player_season_fhalf_lbp_to_space_5_received_90',
    'player_season_f3_lbp_to_space_5_received_90',
    'player_season_average_lbp_to_space_received_distance',
    'player_season_fhalf_average_lbp_to_space_received_distance',
    'player_season_f3_average_lbp_to_space_received_distance',
    'player_season_lbp_to_space_10_90',
    'player_season_fhalf_lbp_to_space_10_90',
    'player_season_f3_lbp_to_space_10_90',
    'player_season_lbp_to_space_2_90',
    'player_season_fhalf_lbp_to_space_2_90',
    'player_season_f3_lbp_to_space_2_90',
    'player_season_lbp_to_space_5_90',
    'player_season_fhalf_lbp_to_space_5_90',
    'player_season_f3_lbp_to_space_5_90',
    'player_season_360_minutes',
    'player_season_xgchain',
    'player_season_op_xgchain',
    'player_season_xgbuildup',
    'player_season_op_xgbuildup'
]

df = df.drop(columns=columns_to_drop)
df.columns = df.columns.str.replace('player_season_', '')
df.columns = df.columns.str.replace('_90', '')


###login
st.title("MTA RADAR StatsBomb Comparison | Data is per 90 min")

# Define the mapping of Short Position to Position 1
position_mapping = {
    'Center Backs': ['Centre Back','Left Centre Back','Right Centre Back'],
    'Full Backs': ['Left Back','Left Wing Back','Right Back','Right Wing Back'],
    'Midfielders': ['Left Defensive Midfielder','Left Centre Midfielder','Centre Attacking Midfielder','Centre Defensive Midfielder','Left Midfielder','Right Centre Midfielder','Right Defensive Midfielder','Right Midfielder'] ,
    'Wingers': ['Left Attacking Midfielder','Left Wing','Right Attacking Midfielder','Right Wing'],
    'Strikers':['Centre Forward','Left Centre Forward','Right Centre Forward'],
    'GK': ['Goalkeeper'],
}

st.sidebar.header("Filter Here:")

# Filter by position group
selected_position_group = st.sidebar.selectbox(
    "Filter by Position Group:",
    options=list(position_mapping.keys()),
)

# Filter by Minutes Played (Min)
min_minutes_played = st.sidebar.slider("Filter by Minimum Minutes Played:", min_value=0, max_value=int(df['minutes'].max()), step=1, value=500)

# Filter the DataFrame based on the selected position group and minimum minutes played
filtered_players = df[(df['primary_position'].isin(position_mapping[selected_position_group])) & (df['minutes'] >= min_minutes_played)]

# List of players based on the selected position group
players_list = filtered_players["player_name"].unique()

Name = st.sidebar.selectbox(
    "Select the Player:",
    options=players_list,
)

Name2 = st.sidebar.selectbox(
    "Select other Player:",
    options=["League Average"] +filtered_players["player_name"].unique().tolist(),
)


    
# List of all available parameters
all_params = list(df.columns[18:])

# Filtered parameters based on user selection
selected_params = st.sidebar.multiselect(
    "Select Parameters:",
    options=all_params,
    default=("np_xg", "np_shots", "key_passes", "passes_into_box", "touches_inside_box",'defensive_actions','obv'))  # Default value is all_params (all parameters selected)

params = selected_params

with st.expander("Show Players Table"):
    # Display the DataFrame with only selected parameters
    selected_columns = ['player_name','team_name','minutes'] + selected_params
    st.dataframe(filtered_players[selected_columns])

filtered_players.fillna(0, inplace=True)

# add ranges to list of tuple pairs
ranges = []
a_values = []
b_values = []

for x in params:
    a = min(df[x])
    a = a - (a * 0.2)

    b = max(df[x])
    b = b

    ranges.append((a, b))

for _, row in df.iterrows():
    if row['player_name'] == Name:
        a_values = row[params].tolist()
    if row['player_name'] == Name2:
        b_values = row[params].tolist()


if Name2 == "League Average":
    league_average_values = filtered_players[filtered_players['player_name'] != Name][params].mean().tolist()
    b_values = league_average_values
    title_name2 = "League Average"
else:
    player2_row = df[df['player_name'] == Name2]
    if not player2_row.empty:
        minutes_player2 = round(player2_row['minutes'].values[0])  # Round to nearest whole number
        Position_name2 = player2_row['primary_position'].values[0]
        team_name2 = player2_row['team_name'].values[0]
        b_values = player2_row[params].values[0].tolist()
        title_name2 = f"{Name2}\n{'Position: ' + Position_name2}\n{'Team:  ' + team_name2}\n{minutes_player2} Minutes Played"
    else:
        st.error(f"No data available for player: {Name2}")
        st.stop()

a_values = a_values[:]
b_values = b_values[:]
values = [a_values, b_values]

# Print values for troubleshooting
minutes_name = "minutes"
minutes_player1 = round(filtered_players.loc[filtered_players['player_name'] == Name, minutes_name].values[0])

Position_name = "primary_position"
Position_name1 = filtered_players.loc[filtered_players['player_name'] == Name, Position_name].values[0]

Team_name = "team_name"
team_name1 = filtered_players.loc[filtered_players['player_name'] == Name, Team_name].values[0]

# Update the title dictionary with minutes played
title = dict(
title_name = f"{Name}\n{'Position: ' +Position_name1} \n{'Team: ' + team_name1}\n{minutes_player1} Minutes Played",
    title_color='yellow',
    title_name_2= title_name2,
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
    fig_size = (12,8),
    ranges=ranges,
    params=params,
    values=values,
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


player_data = filtered_players.loc[filtered_players['player_name'] == Name, selected_params].iloc[0]
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
    figsize=(12, 8),      # adjust figsize according to your need
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
title = f"{Name} Percentile Rank\n{'Compare to all'} {selected_position_group} {'in'} {'Ligat Haal'}"
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
    st.pyplot(fig,use_container_width=True)
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
