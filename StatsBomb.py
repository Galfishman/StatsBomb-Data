import streamlit as st
import warnings
from tempfile import mkdtemp
import requests as req
from requests_cache import CachedSession
import pandas as pd
from mplsoccer import Pitch, VerticalPitch, FontManager, Sbapi
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patheffects as path_effects
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import os
from matplotlib import patches
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
import statsbombpy.entities as ents
from statsbombpy import sb

# Configure plot styles
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = 'Palatino Linotype'

# Retrieve credentials from Streamlit secrets
credentials = st.secrets["credentials"]

# Set environment variables for authentication (optional)
os.environ['SB_USERNAME'] = credentials['user']
os.environ['SB_PASSWORD'] = credentials['passwd']

# Initialize the Sbapi parser
parser = Sbapi(credentials["user"], credentials["passwd"])

# Create a cached session to store API responses
session = CachedSession(cache_name=mkdtemp(), backend="sqlite", expire_after=300)  # Cache for 5 minutes

class NoAuthWarning(UserWarning):
    """Warning raised when no user credentials are provided."""
    pass

def get_resource(url: str, creds: dict) -> list:
    """Fetch resources from the StatsBomb API."""
    auth = req.auth.HTTPBasicAuth(credentials["user"], credentials["passwd"])
    resp = session.get(url, auth=auth)
    if resp.status_code != 200:
        print(f"{url} -> {resp.status_code}")
        resp = []
    else:
        resp = resp.json()
    return resp

def competitions() -> list:
    """Fetch competition data from the StatsBomb API."""
    url = "https://data.statsbomb.com/api/v4/competitions"
    competitions_data = get_resource(url, {credentials["user"], credentials["passwd"]})
    competition_list = [
        {
            "competition_name": comp["competition_name"],
            "country_name": comp["country_name"],
            "competition_id": comp["competition_id"],
            "season_name": comp["season_name"],
            "season_id": comp["season_id"]
        }
        for comp in competitions_data
    ]
    return competition_list

def matches(competition_id: int, season_id: int, match_week: int = None, team: str = None) -> list:
    """Fetch match data for a specific competition and season."""
    url = f"https://data.statsbomb.com/api/v6/competitions/{competition_id}/seasons/{season_id}/matches"
    matches_data = get_resource(url, {credentials["user"], credentials["passwd"]})
    matches_list = []
    for match in matches_data:
        if match_week and match.get("match_week", 0) != match_week:
            continue
        home_team_name = match["home_team"]["home_team_name"]
        away_team_name = match["away_team"]["away_team_name"]
        if team and team not in (home_team_name, away_team_name):
            continue
        match_dict = {
            "match_id": match["match_id"],
            "home_team": home_team_name,
            "away_team": away_team_name,
            "match_date": match["match_date"],
            "home_score": match["home_score"],
            "away_score": match["away_score"],
            "competition_name": match["competition"]["competition_name"],
            "season_name": match["season"]["season_name"],
            "competition_stage_name": match["competition_stage"]["name"],
            "stadium_name": match["stadium"]["name"]
        }
        matches_list.append(match_dict)
    return matches_list

def events(match_id: int) -> list:
    """Fetch event data for a specific match."""
    url = f"https://data.statsbomb.com/api/v8/events/{match_id}"
    events_data = get_resource(url, {credentials["user"], credentials["passwd"]})
    return events_data

def plot_heatmap(df, pitch, End_Zone_Display, PasserPick, TeamPick):
    """Plot Gaussian smoothed heatmap for passes."""
    fig_gaussian, ax_gaussian = pitch.draw(figsize=(8, 16))
    fig_gaussian.set_facecolor('black')
    df_pressure = df

    if End_Zone_Display:
        bin_statistic_gaussian = pitch.bin_statistic(df_pressure.end_x, df_pressure.end_y, statistic='count', bins=(20, 20))
    else:
        bin_statistic_gaussian = pitch.bin_statistic(df_pressure.x, df_pressure.y, statistic='count', bins=(20, 20))
    bin_statistic_gaussian['statistic'] = gaussian_filter(bin_statistic_gaussian['statistic'], 1)
    pcm_gaussian = pitch.heatmap(bin_statistic_gaussian, ax=ax_gaussian, cmap='hot', edgecolors='black')

    cbar_gaussian = fig_gaussian.colorbar(pcm_gaussian, ax=ax_gaussian, shrink=0.25)
    cbar_gaussian.outline.set_edgecolor('#efefef')
    cbar_gaussian.ax.yaxis.set_tick_params(color='#efefef')
    plt.setp(plt.getp(cbar_gaussian.ax.axes, 'yticklabels'), color='#efefef')

    ax_title_gaussian = ax_gaussian.set_title(f"{PasserPick} Heat Map" if PasserPick != "All" else f"{TeamPick} Heat Map", fontsize=16, color='#efefef')

    st.pyplot(fig_gaussian)

def plot_pass_zones(df_pass, pitch, End_Zone_Display, PasserPick, TeamPick):
    """Plot pass zones heatmap."""
    fig_pass, ax_pass = pitch.draw(figsize=(8, 16))
    fig_pass.set_facecolor('black')

    if End_Zone_Display:
        bin_statistic_pass = pitch.bin_statistic_positional(df_pass.end_x, df_pass.end_y, statistic='count', positional='full', normalize=True)
    else:
        bin_statistic_pass = pitch.bin_statistic_positional(df_pass.x, df_pass.y, statistic='count', positional='full', normalize=True)

    pitch.heatmap_positional(bin_statistic_pass, ax=ax_pass, cmap='coolwarm', edgecolors='#22312b')
    pitch.scatter(df_pass.x, df_pass.y, c='white', s=2, ax=ax_pass, alpha=0.2)
    labels = pitch.label_heatmap(bin_statistic_pass, color='white', fontsize=11, ax=ax_pass, ha='center', va='center', str_format='{:0.0%}', path_effects=path_effects)

    ax_title = ax_pass.set_title(f"{PasserPick} Passes zones" if PasserPick != "All" else f"{TeamPick} Passes zones", fontsize=20, pad=10, color='white')
    st.pyplot(fig_pass)

def plot_pass_flow(df_pass, pitch, PasserPick, TeamPick):
    """Plot pass flow map."""
    fig, ax = pitch.draw(figsize=(10, 20), constrained_layout=True, tight_layout=False)
    fig.set_facecolor('black')

    bins = (6, 4)
    bs_heatmap = pitch.bin_statistic(df_pass.x, df_pass.y, statistic='count', bins=bins)
    hm = pitch.heatmap(bs_heatmap, ax=ax, cmap='Blues')
    fm = pitch.flow(df_pass.x, df_pass.y, df_pass.end_x, df_pass.end_y, color='grey', arrow_type='average', arrow_length=15, bins=bins, ax=ax)

    ax_title = ax.set_title(f"{PasserPick} Pass flow map" if PasserPick != "All" else f"{TeamPick} Pass flow map", fontsize=20, pad=-20, color='white')
    st.pyplot(fig)

def plot_attack_zones(df_pass, PasserPick, TeamPick):
    """Plot attack zones with pass and carry entries."""
    def plot_circle(number, x_co_ord, y_co_ord, size, ax):
        circ = patches.Circle((x_co_ord, y_co_ord), size, facecolor='grey', ec="black", lw=3, alpha=0.7, zorder=10)
        ax.add_patch(circ)
        ax.text(s=f"{number}", x=x_co_ord, y=y_co_ord, size=20, color='white', ha="center", va="center", zorder=11, fontweight='bold')

    def plot_percentage(df_pass, ax):
        total = len(df_pass)
        left = len(df_pass[df_pass["end_y"] < 26.6])
        centre = len(df_pass[(df_pass["end_y"] >= 26.6) & (df_pass["end_y"] < 53.3)])
        right = len(df_pass[df_pass["end_y"] >= 53.3])
        left_per = int((left / total) * 100)
        centre_per = int((centre / total) * 100)
        right_per = int((right / total) * 100)

        plot_circle(str(left_per) + "%", 12, 95, 5, ax)
        plot_circle(str(centre_per) + "%", 40, 95, 5, ax)
        plot_circle(str(right_per) + "%", 68, 95, 5, ax)

    pitch = VerticalPitch(pitch_color='grass', line_color='white', stripe_color='#c2d59d', pad_bottom=0.5, half=True)
    fig, ax = pitch.draw(figsize=(8, 10))
    plot_percentage(df_pass, ax)
    plt.title(f"{PasserPick} Attack sides percentage" if PasserPick != "All" else f"{TeamPick} Attack sides percentage", fontsize=20, pad=20, color='black')
    st.pyplot(fig)

def main():
    """Main function to run the Streamlit app."""
    # Get the list of competitions
    competitions_data = competitions()
    competition_list = [comp['competition_name'] for comp in competitions_data]
    competition_list.insert(0, "Select Competition")

    # Streamlit sidebar widgets for user input
    selected_competition = st.sidebar.selectbox('Select Competition', competition_list)
    if selected_competition == "Select Competition":
        st.warning("Please select a competition.")
        return

    # Filter competitions data for the selected competition
    selected_competition_info = next(comp for comp in competitions_data if comp['competition_name'] == selected_competition)
    season_list = [selected_competition_info['season_name']]
    season_list.insert(0, "Select Season")

    selected_season = st.sidebar.selectbox('Select Season', season_list)
    if selected_season == "Select Season":
        st.warning("Please select a season.")
        return

    match_week = st.sidebar.number_input('Enter Matchweek', min_value=1, value=1, step=1)
    team = st.sidebar.text_input('Enter Team')

    # Get the list of matches for the selected competition, season, match week, and team
    matches_data = matches(selected_competition_info['competition_id'], selected_competition_info['season_id'], match_week, team)
    match_list = [f"{match['home_team']} vs {match['away_team']}" for match in matches_data]
    match_list.insert(0, "Select Match")

    selected_match = st.sidebar.selectbox('Select Match', match_list)
    if selected_match == "Select Match":
        st.warning("Please select a match.")
        return

    # Filter matches data for the selected match
    selected_match_info = next(match for match in matches_data if f"{match['home_team']} vs {match['away_team']}" == selected_match)
    match_id = selected_match_info['match_id']

    # Get event data for the selected match
    events_data = events(match_id)
    df_event = pd.json_normalize(events_data, sep='_')
    df = df_event

    home_team_name = selected_match_info["home_team"]
    away_team_name = selected_match_info["away_team"]

    st.title('Match Event Data')
    st.header(f"{selected_match_info['home_team']} ({selected_match_info['home_score']} - {selected_match_info['away_score']}) {selected_match_info['away_team']}")
    st.write(selected_match_info['match_date'])
    st.write(f"Stadium: {selected_match_info['stadium_name']}")

    TeamPick = st.sidebar.selectbox('Select Team', df['team_name'].unique())

    df_pass = df[(df['team_name'] == TeamPick) & (df['type_name'] == "Pass") & (~df['outcome_name'].isin(['Pass Offside', 'Out', 'Incomplete', 'Unknown']))]
    df_pass_pressure = df[(df['team_name'] == TeamPick) & ((df['type_name'] == "Pass") | (df['type_name'] == "Carry")) & (df['under_pressure'] == 1)]
    press_df_team = df_event[df_event['team_name'] == TeamPick]
    press_df = press_df_team[(press_df_team['type_name'] == 'Pressure')]
    df_shots = df[(df['team_name'] == TeamPick) & (df['type_name'] == "Shot")]

    passers = df_pass['player_name'].unique().tolist()
    passers.insert(0, "All")

    PasserPick = st.sidebar.selectbox('Select Player', passers)

    if PasserPick != "All":
        df_pass = df_pass[df_pass['player_name'] == PasserPick]
        df_pass_pressure = df_pass_pressure[df_pass_pressure['player_name'] == PasserPick]
        press_df = press_df[press_df['player_name'] == PasserPick]

    End_Zone_Display = st.sidebar.checkbox('Display End Zone')
    if End_Zone_Display:
        st.write('Plots Show the Passes End Zones!')

    pitch = Pitch(pitch_type='statsbomb', line_zorder=2, pitch_color='black', line_color='white')

    # Call the plotting functions
    plot_heatmap(df_pass, pitch, End_Zone_Display, PasserPick, TeamPick)
    plot_pass_zones(df_pass, pitch, End_Zone_Display, PasserPick, TeamPick)
    plot_pass_flow(df_pass, pitch, PasserPick, TeamPick)
    plot_attack_zones(df_pass, PasserPick, TeamPick)

if __name__ == "__main__":
    main()
