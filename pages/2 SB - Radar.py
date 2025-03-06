import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from soccerplots.radar_chart import Radar
from mplsoccer import PyPizza
import matplotlib as mpl
from io import BytesIO
import requests as req
from requests_cache import CachedSession
from statsbombpy.config import CACHED_CALLS_SECS, HOSTNAME, VERSIONS
import math
from scipy import stats
import os
import requests
from tempfile import mkdtemp

# Credentials and environment variables

# Set your login credentials
credentials = st.secrets["credentials"]
# Set your login credentials
credentials = {"user": st.secrets.credentials.user, "passwd": st.secrets.credentials.passwd}

plt.rcParams['font.family'] = 'Liberation Serif'
plt.rcParams['font.sans-serif'] = 'Palatino Linotype'

os.environ['SB_USERNAME'] = credentials['user']
os.environ['SB_PASSWORD'] = credentials['passwd']

session = CachedSession(cache_name=mkdtemp(), backend="sqlite", expire_after=CACHED_CALLS_SECS)

def get_resource(url: str, creds: dict) -> list:
    auth = req.auth.HTTPBasicAuth(credentials["user"], credentials["passwd"])
    resp = session.get(url, auth=auth)
    if resp.status_code != 200:
        print(f"{url} -> {resp.status_code}")
        return []
    return resp.json()

def competitions() -> list:
    url = f"https://data.statsbomb.com/api/v4/competitions"
    competitions_data = get_resource(url, credentials)
    competition_list = []
    for comp in competitions_data:
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
    seasons_data = get_resource(url, credentials)
    season_list = []
    for season in seasons_data:
        season_dict = {
            "season_name": season["season_name"],
            "season_id": season["season_id"]
        }
        season_list.append(season_dict)
    return season_list

# --- Data Pull and Cleaning ---
competition_list = competitions()
countries = sorted(set(comp["country_name"] for comp in competition_list))
selected_country = st.sidebar.selectbox("Select Country", ["All"] + countries)

competitions_filtered = [comp for comp in competition_list if selected_country == "All" or comp["country_name"] == selected_country]
competition_names = sorted(set(comp["competition_name"] for comp in competitions_filtered))
selected_competition = st.sidebar.selectbox("Select Competition", ["All"] + competition_names)
selected_competition_id = next((comp["competition_id"] for comp in competitions_filtered if comp["competition_name"] == selected_competition), None)

seasons_filtered = [comp for comp in competitions_filtered if selected_competition == "All" or comp["competition_name"] == selected_competition]
season_names = sorted(set(comp["season_name"] for comp in seasons_filtered))
selected_season = st.sidebar.selectbox("Select Season", ["All"] + season_names)
selected_season_id = next((comp["season_id"] for comp in seasons_filtered if comp["season_name"] == selected_season), None)

if selected_competition_id is not None and selected_season_id is not None:
    url = f"https://data.statsbomb.com/api/v4/competitions/{selected_competition_id}/seasons/{selected_season_id}/player-stats"
    data = get_resource(url, credentials)
else:
    url = f"https://data.statsbomb.com/api/v4/competitions/{1211}/seasons/{317}/player-stats"
    data = get_resource(url, credentials)

df = pd.DataFrame(data)

# Drop unwanted columns and clean up names
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

# --- Mapping: Change internal column names to display names ---
display_metric_names = {
    "np_xg": "NP xG",
    "np_shots": "NP Shots",
    "op_xa": "OP XA",
    "op_passes_into_box": "OP Passes Into Box",
    "op_key_passes": "OP Key Passes",
    "crossing_ratio": "Crossing Ratio",
    "touches_inside_box": "Touches Inside Box",
    "dribbles": "Dribbles",
    "turnovers": "Turnovers",
    "defensive_actions": "Defensive Actions",
    "obv_defensive_action": "OBV Defensive Action",
    "tackles_and_interceptions": "Tackles & Interceptions",
    "fhalf_ball_recoveries": "Fhalf Ball Recoveries",
    "counterpressures": "Counterpressures",
    "aggressive_actions": "Aggressive Actions",
    "dribbled_past": "Dribbled Past",
    "aerial_ratio": "Aerial Ratio",
    "fhalf_pressures": "Fhalf Pressures",
    "op_passes": "OP Passes",
    "passing_ratio": "Passing Ratio",
    "forward_pass_proportion": "Forward Pass Proportion",
    "obv_pass": "OBV Pass",
    "pressured_passing_ratio": "Pressured Passing Ratio",
    "unpressured_long_balls": "Unpressured Long Balls",
    "carries": "Carries",
    "key_passes": "Key Passes",
    "passes_into_box": "Passes Into Box",
}

# Now permanently rename the columns in the DataFrame
df = df.rename(columns=display_metric_names)


### Login Title
st.title("MTA RADAR StatsBomb Comparison | Data is per 90 min")

# --- Position Group Mapping ---
position_mapping = {
    'Center Backs': ['Centre Back', 'Left Centre Back', 'Right Centre Back'],
    'Full Backs': ['Left Back', 'Left Wing Back', 'Right Back', 'Right Wing Back'],
    'Midfielders': ['Left Defensive Midfielder', 'Left Centre Midfielder', 'Centre Attacking Midfielder', 'Centre Defensive Midfielder', 'Right Centre Midfielder', 'Right Defensive Midfielder'],
    'Wingers': ['Left Attacking Midfielder', 'Left Wing', 'Right Attacking Midfielder', 'Right Wing', 'Right Midfielder', 'Left Midfielder'],
    'Strikers': ['Centre Forward', 'Left Centre Forward', 'Right Centre Forward'],
    'GK': ['Goalkeeper'],
}

st.sidebar.header("Filter Here:")

# --- Radar Template Selection ---
radar_template = st.sidebar.selectbox("Select Radar Template:", options=["Attacking", "Defending for Attackers", "Defending for Defenders", "Attacking for Defenders", "Custom"])

# --- Filter by Position Group ---
selected_position_group = st.sidebar.selectbox("Filter by Position Group:", options=list(position_mapping.keys()))
min_minutes_played = st.sidebar.slider("Filter by Minimum Minutes Played:", min_value=0, max_value=int(df['minutes'].max()), step=1, value=0)

filtered_players = df[
    (df['primary_position'].isin(position_mapping[selected_position_group])) &
    (df['minutes'] >= min_minutes_played)
]

# --- Player Selection ---
players_list = filtered_players["player_name"].unique()
Name = st.sidebar.selectbox("Select the Player:", options=players_list)

player_b_teams = sorted(filtered_players["team_name"].unique())
selected_team_b = st.sidebar.selectbox("Filter Player B by Team:", options=["All"] + player_b_teams)

if selected_team_b == "All":
    player_b_options = ["League Average"] + filtered_players["player_name"].unique().tolist()
else:
    player_b_options = ["Team Average"] + filtered_players[filtered_players["team_name"] == selected_team_b]["player_name"].unique().tolist()
Name2 = st.sidebar.selectbox("Select other Player:", options=player_b_options)

# --- List of all available parameters (now using display names) ---
all_params = list(df.columns[18:])

# --- Set default parameters using the display names ---
if radar_template == "Attacking":
    default_params = ["NP xG", "NP Shots", "OP XA", "OP Passes Into Box", "OP Key Passes", "Crossing Ratio", "Touches Inside Box", "Dribbles", "Turnovers"]
elif radar_template == "Defending for Attackers":
    default_params = ["Defensive Actions", "OBV Defensive Action", "Tackles & Interceptions", "Fhalf Ball Recoveries", "Counterpressures", "Aggressive Actions"]
elif radar_template == "Defending for Defenders":
    default_params = ["Defensive Actions", "OBV Defensive Action", "Dribbled Past", "Tackles & Interceptions", "Aerial Ratio", "Fhalf Pressures", "Fhalf Ball Recoveries", "Counterpressures"]
elif radar_template == "Attacking for Defenders":
    default_params = ["OP Passes", "Passing Ratio", "Forward Pass Proportion", "OBV Pass", "Pressured Passing Ratio", "Unpressured Long Balls", "Carries"]
else:
    default_params = ["NP xG", "NP Shots", "Key Passes", "Passes Into Box", "Touches Inside Box", "Defensive Actions", "OBV Defensive Action"]

# If "Custom" is chosen, let the user select from all available parameters; otherwise use defaults.
if radar_template == "Custom":
    selected_params = st.sidebar.multiselect("Select Parameters:", options=all_params, default=default_params)
else:
    selected_params = default_params

params = selected_params  # These are now the display names used everywhere
with st.expander("Show Players Table"):
    selected_columns = ['player_name', 'team_name', 'minutes'] + params
    # If a specific team is selected for Player B, filter by that team.
    if selected_team_b != "All":
        players_table = filtered_players[filtered_players["team_name"] == selected_team_b][selected_columns].copy()
    else:
        players_table = filtered_players[selected_columns].copy()

    st.dataframe(players_table)

filtered_players.fillna(0, inplace=True)

# --- Create ranges for each parameter using the filtered players ---
ranges = []
for x in params:
    a = min(filtered_players[x])
    a = a - (a * 0.2)
    b = max(filtered_players[x])
    ranges.append((a, b))

# --- Get values for Player A and Player B using the display names ---
a_values = []
b_values = []
for _, row in df.iterrows():
    if row['player_name'] == Name:
        a_values = row[params].tolist()
    if row['player_name'] == Name2:
        b_values = row[params].tolist()

if Name2 == "League Average":
    league_average_values = filtered_players[filtered_players['player_name'] != Name][params].mean().tolist()
    b_values = league_average_values
    title_name2 = "League Average"
elif Name2 == "Team Average":
    team_average_values = filtered_players[filtered_players["team_name"] == selected_team_b][params].mean().tolist()
    b_values = team_average_values
    title_name2 = f"{selected_team_b} Team Average"
else:
    player2_row = df[df['player_name'] == Name2]
    if not player2_row.empty:
        minutes_player2 = round(player2_row['minutes'].values[0])
        Position_name2 = player2_row['primary_position'].values[0]
        team_name2 = player2_row['team_name'].values[0]
        if 'birth_date' in player2_row.columns:
            birth_date_B = player2_row['birth_date'].values[0]
            title_name2 = f"{Name2}\nBirth Date: {birth_date_B}\nPosition: {Position_name2}\nTeam: {team_name2}\n{minutes_player2} Minutes Played"
        else:
            title_name2 = f"{Name2}\nPosition: {Position_name2}\nTeam: {team_name2}\n{minutes_player2} Minutes Played"
        b_values = player2_row[params].values[0].tolist()
    else:
        st.error(f"No data available for player: {Name2}")
        st.stop()

values = [a_values, b_values]

# --- Retrieve details for Player A ---
minutes_player1 = round(filtered_players.loc[filtered_players['player_name'] == Name, "minutes"].values[0])
Position_name1 = filtered_players.loc[filtered_players['player_name'] == Name, "primary_position"].values[0]
team_name1 = filtered_players.loc[filtered_players['player_name'] == Name, "team_name"].values[0]
if 'birth_date' in filtered_players.columns:
    birth_date_A = filtered_players.loc[filtered_players['player_name'] == Name, 'birth_date'].values[0]
else:
    birth_date_A = "N/A"

title = dict(
    title_name=f"{Name}\nBirth Date: {birth_date_A}\nPosition: {Position_name1}\nTeam: {team_name1}\n{minutes_player1} Minutes Played",
    title_color='yellow',
    title_name_2=title_name2,
    title_color_2='blue',
    title_fontsize=12,
)

if radar_template == "Attacking":
    chosen_radar_color = ['yellow', 'blue']
elif radar_template == "Defending for Attackers":
    chosen_radar_color = ['yellow', 'blue']
elif radar_template == "Defending for Defenders":
    chosen_radar_color = ['yellow', 'blue']
elif radar_template == "Attacking for Defenders":
    chosen_radar_color = ['yellow', 'blue']
else:
    chosen_radar_color = ['yellow', 'blue']

# --- RADAR PLOT ---
radar = Radar(
    background_color="#121212",
    patch_color="#28252C",
    label_color="#FFFFFF",
    label_fontsize=10,
    range_color="#FFFFFF"
)

fig, ax = radar.plot_radar(
    fig_size=(12, 8),
    ranges=ranges,
    params=params,
    values=values,
    radar_color=chosen_radar_color,
    edgecolor="#222222",
    zorder=2,
    linewidth=1,
    title=title,
    alphas=[0.4, 0.4],
    compare=True
)

mpl.rcParams['figure.dpi'] = 1500

# --- Percentile Radar Plot (PyPizza) ---
player_data = filtered_players.loc[filtered_players['player_name'] == Name, params].iloc[0]
values_percentile = [math.floor(stats.percentileofscore(filtered_players[param], player_data[param])) for param in params]

baker = PyPizza(
    params=params,
    straight_line_color="white",
    straight_line_lw=2,
    last_circle_lw=3,
    last_circle_color='white',
    other_circle_lw=1,
    other_circle_color='grey',
    other_circle_ls="-."
)

fig2, ax2 = baker.make_pizza(
    values_percentile,
    figsize=(12, 8),
    param_location=105,
    kwargs_slices=dict(
        facecolor="cornflowerblue", edgecolor='white',
        zorder=2, linewidth=3
    ),
    kwargs_params=dict(
        color="white", fontsize=10,
        va="center"
    ),
    kwargs_values=dict(
        color="white", fontsize=12,
        zorder=3,
        bbox=dict(
            edgecolor="#000000", facecolor="cornflowerblue",
            boxstyle="round,pad=0.2", lw=1
        )
    )
)

fig2.patch.set_facecolor('#121212')
ax2.set_facecolor('#121212')

title_text = f"Percentile Comparison\n{Name} vs {title_name2}\n({selected_position_group} | {selected_competition})"
title_bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#000000", lw=1)
fig2.text(0.515, 0.97, title_text, size=18, ha="center", color="#000000", bbox=title_bbox_props)

col1, col2 = st.columns([1, 1], gap='large') 
with col1:
    st.header("Values Radar")
    st.pyplot(fig, use_container_width=True)
with col2:
    st.header("Percentile Rank")
    st.pyplot(fig2)

head_to_head_df = pd.DataFrame({
    'Player': [Name, Name2],
    **{param: [a_values[i], b_values[i]] for i, param in enumerate(params)}
})
head_to_head_df_transposed = head_to_head_df.set_index('Player').T
highlighted_df = head_to_head_df_transposed.style.format("{:.2f}").apply(
    lambda row: ['background-color: grey' if val == row.max() else '' for val in row],
    axis=1
)
st.header("Head-to-Head Comparison")
st.table(highlighted_df)
