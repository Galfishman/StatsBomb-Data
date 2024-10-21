import streamlit as st
import warnings
from tempfile import mkdtemp
import requests as req
from requests_cache import CachedSession
import streamlit as st
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import Pitch
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
from mplsoccer import VerticalPitch, FontManager
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase
import matplotlib.patheffects as pe 
from io import BytesIO
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import FancyArrowPatch
from mplsoccer.pitch import VerticalPitch
import streamlit_authenticator as stauth
from adjustText import adjust_text
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator, FuncFormatter
import statsbombpy.entities as ents
from statsbombpy.config import CACHED_CALLS_SECS, HOSTNAME, VERSIONS
from mplsoccer import Sbopen
from mplsoccer import Sbapi
from statsbombpy import sb
import plottable as pltb
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.formatters import decimal_to_percent
import matplotlib.cm as cm
import matplotlib.pyplot as plt  # Import matplotlib
from plottable.table import Table, ColumnDefinition
import os
from matplotlib import patches
import streamlit_authenticator as stauth  # pip install streamlit-authenticator
import pickle
from pathlib import Path



plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = 'Palatino Linotype'
credentials = st.secrets["credentials"]


# Set your login credentials
credentials = {"user": st.secrets.credentials.user, "passwd": st.secrets.credentials.passwd}

# Set environment variables for authentication (optional)
os.environ['SB_USERNAME'] = credentials['user']
os.environ['SB_PASSWORD'] = credentials['passwd']

# The first thing we have to do is open the data. We use a parser SBopen available in mplsoccer.
parser = Sbapi(credentials["user"], credentials["passwd"])

# Create a cached session to store API responses
session = CachedSession(cache_name=mkdtemp(), backend="sqlite", expire_after=CACHED_CALLS_SECS)

class NoAuthWarning(UserWarning):
    """Warning raised when no user credentials are provided."""

    pass


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
        competition_dict = {
            "competition_name": comp["competition_name"],
            "country_name": comp["country_name"],
            "competition_id": comp["competition_id"],
            "season_name": comp["season_name"],
            "season_id": comp["season_id"]
        }
        competition_list.append(competition_dict)
    return competition_list

def matches(competition_id: int, season_id: int, match_week: int = None, team: str = None) -> list:
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
    url = f"https://data.statsbomb.com/api/v8/events/{match_id}"
    events_data = get_resource(url, {credentials["user"], credentials["passwd"]})
    return events_data



def main():
    # Call competitions function to fetch data
    competition_data = competitions()
    all_figures = []

    
    # Filter by country
    countries = sorted(set(comp["country_name"] for comp in competition_data))
    selected_country = st.sidebar.selectbox("Select Country", ["All"] + countries)

    # Filter by competition based on selected country
    competitions_filtered = [comp for comp in competition_data if selected_country == "All" or comp["country_name"] == selected_country]
    competition_names = sorted(set(comp["competition_name"] for comp in competitions_filtered))
    selected_competition = st.sidebar.selectbox("Select Competition", ["All"] + competition_names)

    # Filter by season based on selected competition
    seasons_filtered = [comp for comp in competitions_filtered if selected_competition == "All" or comp["competition_name"] == selected_competition]
    season_names = sorted(set(comp["season_name"] for comp in seasons_filtered))
    selected_season = st.sidebar.selectbox("Select Season", ["All"] + season_names)

    # Filter by match week
    selected_match_week = st.sidebar.selectbox("Select Match Week", ["All"] + list(range(1, 40)))

    # Filter by team
    teams = []
    if selected_season != "All":
        filtered_data = [comp for comp in seasons_filtered if comp["season_name"] == selected_season]


    # Display matches data
    if selected_season != "All":
        matches_data = matches(filtered_data[0]["competition_id"], filtered_data[0]["season_id"], selected_match_week)

        # Select specific match by combining home team and away team
        match_teams = [(match["home_team"], match["away_team"]) for match in matches_data]
        selected_match = st.sidebar.selectbox("Select Match", ["All"] + [f"{home_team} vs {away_team}" for home_team, away_team in match_teams])
        if selected_match != "All":
            selected_match_index = [f"{home_team} vs {away_team}" for home_team, away_team in match_teams].index(selected_match)
            selected_match_info = matches_data[selected_match_index]
            match_id= selected_match_info["match_id"]
                        # Fetch player match statistics
            player_match = sb.player_match_stats(match_id,"dict", {"user":credentials["user"] ,"passwd":credentials["passwd"] })
            
            # Convert player match statistics to DataFrame
            players_df = pd.DataFrame(player_match)
            # Drop unnecessary columns
            columns_to_drop = [
                'player_match_average_space_received_in',
                'player_match_average_fhalf_space_received_in',
                'player_match_average_f3_space_received_in',
                'player_match_ball_receipts_in_space_10',
                'player_match_ball_receipts_in_space_2',
                'player_match_ball_receipts_in_space_5',
                'player_match_fhalf_ball_receipts_in_space_10',
                'player_match_fhalf_ball_receipts_in_space_2',
                'player_match_fhalf_ball_receipts_in_space_5',
                'player_match_f3_ball_receipts_in_space_10',
                'player_match_f3_ball_receipts_in_space_2',
                'player_match_f3_ball_receipts_in_space_5',
                'player_match_lbp',
                'player_match_lbp_completed',
                'player_match_fhalf_lbp_completed',
                'player_match_f3_lbp_completed',
                'player_match_fhalf_lbp',
                'player_match_f3_lbp',
                'player_match_obv_lbp',
                'player_match_fhalf_obv_lbp',
                'player_match_f3_obv_lbp',
                'player_match_lbp_received',
                'player_match_fhalf_lbp_received',
                'player_match_f3_lbp_received',
                'player_match_average_lbp_to_space_distance',
                'player_match_fhalf_average_lbp_to_space_distance',
                'player_match_f3_average_lbp_to_space_distance',
                'player_match_lbp_to_space_10_received',
                'player_match_fhalf_lbp_to_space_10_received',
                'player_match_f3_lbp_to_space_10_received',
                'player_match_lbp_to_space_2_received',
                'player_match_fhalf_lbp_to_space_2_received',
                'player_match_f3_lbp_to_space_2_received',
                'player_match_lbp_to_space_5_received',
                'player_match_fhalf_lbp_to_space_5_received',
                'player_match_f3_lbp_to_space_5_received',
                'player_match_average_lbp_to_space_received_distance',
                'player_match_fhalf_average_lbp_to_space_received_distance',
                'player_match_f3_average_lbp_to_space_received_distance',
                'player_match_lbp_to_space_10',
                'player_match_fhalf_lbp_to_space_10',
                'player_match_f3_lbp_to_space_10',
                'player_match_lbp_to_space_2',
                'player_match_fhalf_lbp_to_space_2',
                'player_match_f3_lbp_to_space_2',
                'player_match_lbp_to_space_5',
                'player_match_fhalf_lbp_to_space_5',
                'player_match_f3_lbp_to_space_5',
                'player_match_passes_360',
                'player_match_obv_passes_360',
                'player_match_fhalf_passes_360',
                'player_match_fhalf_obv_passes_360',
                'player_match_f3_passes_360',
                'player_match_f3_obv_passes_360',
                'player_match_ball_receipts_360',
                'player_match_fhalf_ball_receipts_360',
                'player_match_f3_ball_receipts_360'
            ]
  # Define columns you want to drop
            players_df = players_df.drop(columns=columns_to_drop)
            players_df.columns = players_df.columns.str.replace('player_match_', '')
            

            # Load data (replace this with your data loading function)
            df_event, df_related, df_freeze, df_tactics = parser.event(match_id)
            df=df_event
            tactics =df_tactics
            
            # Split the data into two teams
        # Extract home team and away team from match details
            home_team_name = selected_match_info["home_team"]
            away_team_name = selected_match_info["away_team"]

            # Streamlit app
            st.title('Match Event Data')
            # Header displaying match details
            st.header(f"{selected_match_info['home_team']} ({selected_match_info['home_score']} - {selected_match_info['away_score']}) {selected_match_info['away_team']}")
            st.write(selected_match_info['match_date'])
            st.write(f"Stadium: {selected_match_info['stadium_name']}")
            

            # Filter data for the selected play type
            df_pass = df[(df['type_name'] == "Pass") & (~df['outcome_name'].isin(['Pass Offside', 'Out', 'Incomplete', 'Unknown']))]
            df_pass_pressure = df[((df['type_name'] == "Pass") | (df['type_name'] == "Carry")) & (df['under_pressure'] == 1)]
            press_df = df_event[df_event['type_name'] == 'Pressure']
            df_shots = df[df['type_name'] == "Shot"]
                      # Group by team name and show statistics
            # Display all statistics available in players_df

            # Group by team name
            team_grouped = players_df.groupby('team_name').sum().reset_index()
#########################################################################################################################################################################
        # Extract home and away team names from the match information
        home_team = selected_match_info["home_team"]
        away_team = selected_match_info["away_team"]

        total_passes_all_teams = team_grouped['passes'].sum()

        # Calculate possession percentage for each team
        team_grouped['possession_percent'] = (team_grouped['passes'] / total_passes_all_teams) * 100

        # Extract possession percentage with home team first
        possession_percent_team_1 = team_grouped[team_grouped['team_name'] == home_team]['possession_percent'].values[0]
        possession_percent_team_2 = team_grouped[team_grouped['team_name'] == away_team]['possession_percent'].values[0]

        # Extract total passes with home team first
        total_passes_team_1 = team_grouped[team_grouped['team_name'] == home_team]['passes'].values[0]
        total_passes_team_2 = team_grouped[team_grouped['team_name'] == away_team]['passes'].values[0]

        # Store stats in lists (home team first, away team second)
        possession_percent = [possession_percent_team_1, possession_percent_team_2]
        total_passes = [total_passes_team_1, total_passes_team_2]

        # Calculate total duels from df (considering all duels)
        total_duels = len(df[df['type_name'] == 'Duel'])
        # Extract aerial duels for both teams (already correctly ordered)
        aerials_team_1 = team_grouped[team_grouped['team_name'] == home_team]['aerials'].sum()
        aerials_team_2 = team_grouped[team_grouped['team_name'] == away_team]['aerials'].sum()

        # Extract successful aerial duels for both teams
        successful_aerials_team_1 = team_grouped[team_grouped['team_name'] == home_team]['successful_aerials'].sum()
        successful_aerials_team_2 = team_grouped[team_grouped['team_name'] == away_team]['successful_aerials'].sum()

        # Calculate the aerial success percentage for both teams
        aerial_success_team_1 = (successful_aerials_team_1 / aerials_team_1) * 100 if aerials_team_1 > 0 else 0
        aerial_success_team_2 = (successful_aerials_team_2 / aerials_team_2) * 100 if aerials_team_2 > 0 else 0


        # Calculate combined duels by adding total duels and aerials (avoiding double counting)
        combined_duels = total_duels + aerials_team_1  # Using only one team’s aerials to avoid overlap
        # Calculate average challenge ratio for each team
        # Calculate average challenge ratio for each team and convert to percentage

# Calculate the total challenge ratio for both teams
        total_challenge_ratio_team_1 = players_df[players_df['team_name'] == home_team]['challenge_ratio'].sum()
        total_challenge_ratio_team_2 = players_df[players_df['team_name'] == away_team]['challenge_ratio'].sum()


        # Calculate the normalized percentage share for each team
        total_challenge_ratio_sum = total_challenge_ratio_team_1 + total_challenge_ratio_team_2
        challenge_ratio_team_1_percent = (total_challenge_ratio_team_1 / total_challenge_ratio_sum) * 100
        challenge_ratio_team_2_percent = (total_challenge_ratio_team_2 / total_challenge_ratio_sum) * 100
                # Calculate total dribbles for both teams
        total_dribbles_team_1 = len(df[(df['team_name'] == home_team) & (df['type_name'] == 'Dribble')])
        total_dribbles_team_2 = len(df[(df['team_name'] == away_team) & (df['type_name'] == 'Dribble')])

        # Calculate successful dribbles for both teams
        successful_dribbles_team_1 = len(df[(df['team_name'] == home_team) & (df['type_name'] == 'Dribble') & (df['outcome_name'] == 'Complete')])
        successful_dribbles_team_2 = len(df[(df['team_name'] == away_team) & (df['type_name'] == 'Dribble') & (df['outcome_name'] == 'Complete')])

        # Add dribble success percentage
        dribble_success_team_1 = (successful_dribbles_team_1 / total_dribbles_team_1) * 100 if total_dribbles_team_1 > 0 else 0
        dribble_success_team_2 = (successful_dribbles_team_2 / total_dribbles_team_2) * 100 if total_dribbles_team_2 > 0 else 0
        # Extract fouls won from team_grouped
        fouls_won_team_1 = team_grouped[team_grouped['team_name'] == home_team]['fouls_won'].sum()
        fouls_won_team_2 = team_grouped[team_grouped['team_name'] == away_team]['fouls_won'].sum()

        # Count the number of Pass Offside from df
        pass_offside_team_1 = len(df[(df['team_name'] == home_team) & (df['type_name'] == 'Pass') & (df['outcome_name'] == 'Pass Offside')])
        pass_offside_team_2 = len(df[(df['team_name'] == away_team) & (df['type_name'] == 'Pass') & (df['outcome_name'] == 'Pass Offside')])

        # Count the number of corner passes from df
        corner_passes_team_1 = len(df[(df['team_name'] == home_team) & (df['type_name'] == 'Pass') & (df['sub_type_name'] == 'Corner')])
        corner_passes_team_2 = len(df[(df['team_name'] == away_team) & (df['type_name'] == 'Pass') & (df['sub_type_name'] == 'Corner')])



        # Define colors for each team
        # Define colors for each team dynamically
        team_colors = {
            home_team: "#215454", 
            away_team: "#05a5a6" 
        }

        # Ensure the home team always appears first in stats
        colors = [team_colors[home_team], team_colors[away_team]]
        # Create the figure and axis
 
        # Create the figure and axis with increased size
        fig, ax = plt.subplots(figsize=(24, 16),dpi=800)  # Larger figure size

        # Adjust spacing between bars
        spacing = 2  # Control vertical space between bars
        bar_height = 0.40  # Make bars thinner

        # Helper function to plot proportional bars with consistent order and dynamic colors
        def plot_bar(y_pos, value1, value2, title, percentage_format=False):
            """Plot a stat as a proportional bar with two segments."""
            total_value = value1 + value2  # Calculate total value

            # Calculate segment widths based on values
            width1 = (value1 / total_value) * 100 if total_value != 0 else 50
            width2 = (value2 / total_value) * 100 if total_value != 0 else 50

            # Plot home team segment
            ax.barh([y_pos], width1, color=colors[0], align='center', height=bar_height)
            # Plot away team segment
            ax.barh([y_pos], width2, left=width1, color=colors[1], align='center', height=bar_height)

            # Format labels as percentage or absolute values
            label1 = f"{value1:.1f}%" if percentage_format else f"{value1}"
            label2 = f"{value2:.1f}%" if percentage_format else f"{value2}"

            # Add labels outside the bars
            ax.text(-2, y_pos, label1, va='center', ha='right', fontsize=14,)
            ax.text(102, y_pos, label2, va='center', ha='left', fontsize=14,)

            # Add the title above the bar
            ax.text(50, y_pos + 0.5, title, va='center', ha='center', fontsize=14, color='grey')

        plot_bar(10 * spacing, possession_percent[0], possession_percent[1], "Possession (%)", percentage_format=True)
        plot_bar(9 * spacing, total_passes[0], total_passes[1], "Total Passes")
        plot_bar(8 * spacing, combined_duels, combined_duels, "Duels (including Aerials)")
        plot_bar(7 * spacing, challenge_ratio_team_1_percent, challenge_ratio_team_2_percent, "Duels Success Rate (%)", percentage_format=True)
        plot_bar(6 * spacing, aerials_team_1, aerials_team_2, "Aerial Duels")
        plot_bar(5 * spacing, aerial_success_team_1, aerial_success_team_2, "Aerial Success Rate (%)", percentage_format=True)
        plot_bar(4 * spacing, total_dribbles_team_1, total_dribbles_team_2, "Total Dribbles")
        plot_bar(3 * spacing, successful_dribbles_team_1, successful_dribbles_team_2, "Successful Dribbles")
        plot_bar(2 * spacing, fouls_won_team_1, fouls_won_team_2, "Fouls Won")
        plot_bar(1 * spacing, pass_offside_team_1, pass_offside_team_2, "Pass Offside")
        plot_bar(0 * spacing, corner_passes_team_1, corner_passes_team_2, "Corner Passes")

        # Adjust axis for a clean layout
        ax.set_xlim(0, 100)  # All bars span from 0 to 100%
        ax.set_ylim(-1, 11 * spacing)  # Adjust y-axis to fit all bars with spacing
        ax.set_yticks([])  # Remove y-axis ticks
        ax.set_xticks([])  # Remove x-axis ticks

        # Hide spines for minimal design
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add match information at the bottom
        fig.text(0.5, 0.04, f"{home_team} ({selected_match_info['home_score']} - {selected_match_info['away_score']}) {away_team}", 
                ha='center', fontsize=16, fontweight='bold')
        fig.text(0.5, 0.02, f"{selected_match_info['match_date']} | Stadium: {selected_match_info['stadium_name']}", 
                ha='center', fontsize=14)

        # Add a title for the figure
        fig.suptitle('Overview', fontsize=30, fontweight='bold', y=0.95)
        # Add home team name on the left
        fig.text(0.01, 0.95, home_team, fontsize=20, ha='left', va='center', fontweight='bold', color=colors[0])

        # Add away team name on the right
        fig.text(0.99, 0.95, away_team, fontsize=20, ha='right', va='center', fontweight='bold', color=colors[1])
        all_figures.append(fig)

        st.pyplot(fig)





#########################################################################################################################################################################

        # Goals from match info
        goals_team_1 = selected_match_info["home_score"]
        goals_team_2 = selected_match_info["away_score"]

        # Open Play xG from team_grouped
        xg_team_1 = team_grouped[team_grouped['team_name'] == home_team]['np_xg'].values[0]
        xg_team_2 = team_grouped[team_grouped['team_name'] == away_team]['np_xg'].values[0]

        # Total Shots from df
        shots_team_1 = len(df[(df['team_name'] == home_team) & (df['type_name'] == 'Shot')])
        shots_team_2 = len(df[(df['team_name'] == away_team) & (df['type_name'] == 'Shot')])

        # Shots on Target (Goals + Saves)
        shots_on_target_team_1 = len(df[(df['team_name'] == home_team) & 
                                        (df['type_name'] == 'Shot') & 
                                        (df['outcome_name'].isin(['Goal', 'Saved']))])
        shots_on_target_team_2 = len(df[(df['team_name'] == away_team) & 
                                        (df['type_name'] == 'Shot') & 
                                        (df['outcome_name'].isin(['Goal', 'Saved']))])

        # Shots Blocked
        shots_blocked_team_1 = len(df[(df['team_name'] == home_team) & 
                                    (df['type_name'] == 'Shot') & 
                                    (df['outcome_name'] == 'Blocked')])
        shots_blocked_team_2 = len(df[(df['team_name'] == away_team) & 
                                    (df['type_name'] == 'Shot') & 
                                    (df['outcome_name'] == 'Blocked')])

        # Headed Shots
        headed_shots_team_1 = len(df[(df['team_name'] == home_team) & 
                                    (df['type_name'] == 'Shot') & 
                                    (df['body_part_name'] == 'Head')])
        headed_shots_team_2 = len(df[(df['team_name'] == away_team) & 
                                    (df['type_name'] == 'Shot') & 
                                    (df['body_part_name'] == 'Head')])
        # Define the box coordinates
        box_x_min, box_y_min, box_y_max = 102, 18, 62

        # Calculate total shots inside the box for each team
        shots_inside_box_team_1 = len(df[(df['team_name'] == home_team) & 
                                        (df['type_name'] == 'Shot') & 
                                        (df['x'] >= box_x_min) & 
                                        (df['y'] >= box_y_min) & (df['y'] <= box_y_max)])

        shots_inside_box_team_2 = len(df[(df['team_name'] == away_team) & 
                                        (df['type_name'] == 'Shot') & 
                                        (df['x'] >= box_x_min) & 
                                        (df['y'] >= box_y_min) & (df['y'] <= box_y_max)])
        
                # Calculate total shots outside the box
        shots_outside_box_team_1 = shots_team_1 - shots_inside_box_team_1
        shots_outside_box_team_2 = shots_team_2 - shots_inside_box_team_2

                # Calculate Shot Accuracy
        shot_accuracy_team_1 = (shots_on_target_team_1 / shots_team_1) * 100 if shots_team_1 > 0 else 0
        shot_accuracy_team_2 = (shots_on_target_team_2 / shots_team_2) * 100 if shots_team_2 > 0 else 0

        # Calculate Shot Accuracy Excluding Blocked Shots
        shots_without_blocked_team_1 = shots_team_1 - shots_blocked_team_1
        shots_without_blocked_team_2 = shots_team_2 - shots_blocked_team_2

        shot_accuracy_no_block_team_1 = (shots_on_target_team_1 / shots_without_blocked_team_1) * 100 if shots_without_blocked_team_1 > 0 else 0
        shot_accuracy_no_block_team_2 = (shots_on_target_team_2 / shots_without_blocked_team_2) * 100 if shots_without_blocked_team_2 > 0 else 0



        # Create the figure and axis with increased size
        fig, ax = plt.subplots(figsize=(24, 16),dpi=800)  # Larger figure size

        # Adjust spacing between bars
        spacing = 2  # Control vertical space between bars
        bar_height = 0.40  # Make bars thinner

        # Helper function to plot proportional bars with consistent order and dynamic colors
        def plot_bar(y_pos, value1, value2, title, percentage_format=False,xg_format=False):
            """Plot a stat as a proportional bar with two segments."""
            total_value = value1 + value2  # Calculate total value

            # Calculate segment widths based on values
            width1 = (value1 / total_value) * 100 if total_value != 0 else 50
            width2 = (value2 / total_value) * 100 if total_value != 0 else 50

            # Plot home team segment
            ax.barh([y_pos], width1, color=colors[0], align='center', height=bar_height)
            # Plot away team segment
            ax.barh([y_pos], width2, left=width1, color=colors[1], align='center', height=bar_height)

            if percentage_format:
                label1 = f"{value1:.1f}%"
                label2 = f"{value2:.1f}%"
            elif xg_format:
                label1 = f"{value1:.2f}"
                label2 = f"{value2:.2f}"
            else:
                label1 = f"{value1}"
                label2 = f"{value2}"

            # Add labels outside the bars
            ax.text(-2, y_pos, label1, va='center', ha='right', fontsize=14,)
            ax.text(102, y_pos, label2, va='center', ha='left', fontsize=14,)

            # Add the title above the bar
            ax.text(50, y_pos + 0.5, title, va='center', ha='center', fontsize=14, color='grey',)

        plot_bar(10 * spacing, goals_team_1, goals_team_2, "Goals")
        plot_bar(9 * spacing, xg_team_1, xg_team_2, "Open Play xG", xg_format=True)
        plot_bar(8 * spacing, shots_team_1, shots_team_2, "Total Shots")
        plot_bar(7 * spacing, shots_on_target_team_1, shots_on_target_team_2, "Shots on Target")
        plot_bar(6 * spacing, shots_blocked_team_1, shots_blocked_team_2, "Shots Blocked")
        plot_bar(5 * spacing, headed_shots_team_1, headed_shots_team_2, "Headed Shots")
        plot_bar(4 * spacing, shot_accuracy_team_1, shot_accuracy_team_2, "Shot Accuracy (%)", percentage_format=True)
        plot_bar(3 * spacing, shot_accuracy_no_block_team_1, shot_accuracy_no_block_team_2, 
                "Shot Acc. (Excluding Blocked) (%)", percentage_format=True)
        plot_bar(2 * spacing, shots_outside_box_team_1, shots_outside_box_team_2, "Shots Outside Box")
        plot_bar(1 * spacing, shots_inside_box_team_1, shots_inside_box_team_2, "Shots Inside Box")

        # Adjust axis for a clean layout
        ax.set_xlim(0, 100)  # All bars span from 0 to 100%
        ax.set_ylim(-1, 11 * spacing)  # Adjust y-axis to fit all bars with spacing
        ax.set_yticks([])  # Remove y-axis ticks
        ax.set_xticks([])  # Remove x-axis ticks

        # Hide spines for minimal design
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add match information at the bottom
        fig.text(0.5, 0.04, f"{home_team} ({selected_match_info['home_score']} - {selected_match_info['away_score']}) {away_team}", 
                ha='center', fontsize=16, fontweight='bold')
        fig.text(0.5, 0.02, f"{selected_match_info['match_date']} | Stadium: {selected_match_info['stadium_name']}", 
                ha='center', fontsize=14)

        # Add a title for the figure
        fig.suptitle('Attacking', fontsize=30, fontweight='bold', y=0.95)
        # Add home team name on the left
        fig.text(0.01, 0.95, home_team, fontsize=20, ha='left', va='center', fontweight='bold', color=colors[0])

        # Add away team name on the right
        fig.text(0.99, 0.95, away_team, fontsize=20, ha='right', va='center', fontweight='bold', color=colors[1])


        all_figures.append(fig)

        # Display the figure in Streamlit
        st.pyplot(fig)


#########################################################################################################################################################################

        # Extract passing stats from team_grouped
        op_passes_team_1 = team_grouped[team_grouped['team_name'] == home_team]['op_passes'].values[0]
        op_passes_team_2 = team_grouped[team_grouped['team_name'] == away_team]['op_passes'].values[0]

        op_f3_passes_team_1 = team_grouped[team_grouped['team_name'] == home_team]['op_f3_passes'].values[0]
        op_f3_passes_team_2 = team_grouped[team_grouped['team_name'] == away_team]['op_f3_passes'].values[0]

        passes_into_box_team_1 = team_grouped[team_grouped['team_name'] == home_team]['passes_into_box'].values[0]
        passes_into_box_team_2 = team_grouped[team_grouped['team_name'] == away_team]['passes_into_box'].values[0]

        touches_inside_box_team_1 = team_grouped[team_grouped['team_name'] == home_team]['touches_inside_box'].values[0]
        touches_inside_box_team_2 = team_grouped[team_grouped['team_name'] == away_team]['touches_inside_box'].values[0]

        long_balls_team_1 = team_grouped[team_grouped['team_name'] == home_team]['long_balls'].values[0]
        long_balls_team_2 = team_grouped[team_grouped['team_name'] == away_team]['long_balls'].values[0]

        # Calculate long ball proportion
        long_ball_ratio_team_1 = (long_balls_team_1 / total_passes_team_1) * 100 if total_passes_team_1 > 0 else 0
        long_ball_ratio_team_2 = (long_balls_team_2 / total_passes_team_2) * 100 if total_passes_team_2 > 0 else 0

        through_balls_team_1 = team_grouped[team_grouped['team_name'] == home_team]['through_balls'].values[0]
        through_balls_team_2 = team_grouped[team_grouped['team_name'] == away_team]['through_balls'].values[0]

        crosses_team_1 = team_grouped[team_grouped['team_name'] == home_team]['crosses'].values[0]
        crosses_team_2 = team_grouped[team_grouped['team_name'] == away_team]['crosses'].values[0]

        successful_crosses_team_1 = team_grouped[team_grouped['team_name'] == home_team]['successful_crosses'].values[0]
        successful_crosses_team_2 = team_grouped[team_grouped['team_name'] == away_team]['successful_crosses'].values[0]

        # Calculate crossing ratio
        crossing_ratio_team_1 = (successful_crosses_team_1 / crosses_team_1) * 100 if crosses_team_1 > 0 else 0
        crossing_ratio_team_2 = (successful_crosses_team_2 / crosses_team_2) * 100 if crosses_team_2 > 0 else 0


        # Create the figure and axis with increased size
        fig, ax = plt.subplots(figsize=(24, 16),dpi=800)  # Larger figure size

        # Adjust spacing between bars
        spacing = 2  # Control vertical space between bars
        bar_height = 0.40  # Make bars thinner

        # Helper function to plot proportional bars with consistent order and dynamic colors
        def plot_bar(y_pos, value1, value2, title, percentage_format=False,xg_format=False):
            """Plot a stat as a proportional bar with two segments."""
            total_value = value1 + value2  # Calculate total value

            # Calculate segment widths based on values
            width1 = (value1 / total_value) * 100 if total_value != 0 else 50
            width2 = (value2 / total_value) * 100 if total_value != 0 else 50

            # Plot home team segment
            ax.barh([y_pos], width1, color=colors[0], align='center', height=bar_height)
            # Plot away team segment
            ax.barh([y_pos], width2, left=width1, color=colors[1], align='center', height=bar_height)

            if percentage_format:
                label1 = f"{value1:.1f}%"
                label2 = f"{value2:.1f}%"
            elif xg_format:
                label1 = f"{value1:.2f}"
                label2 = f"{value2:.2f}"
            else:
                label1 = f"{value1}"
                label2 = f"{value2}"

            # Add labels outside the bars
            ax.text(-2, y_pos, label1, va='center', ha='right', fontsize=14)
            ax.text(102, y_pos, label2, va='center', ha='left', fontsize=14,)

            # Add the title above the bar
            ax.text(50, y_pos + 0.5, title, va='center', ha='center', fontsize=14, color='grey')

        plot_bar(10 * spacing, op_passes_team_1, op_passes_team_2, "Open Play Passes")
        plot_bar(9 * spacing, op_f3_passes_team_1, op_f3_passes_team_2, "Final Third Passes")
        plot_bar(8 * spacing, passes_into_box_team_1, passes_into_box_team_2, "Passes Into the Box")
        plot_bar(7 * spacing, touches_inside_box_team_1, touches_inside_box_team_2, "Touches Inside Box")
        plot_bar(6 * spacing, long_balls_team_1, long_balls_team_2, "Long Balls")
        plot_bar(5 * spacing, long_ball_ratio_team_1, long_ball_ratio_team_2, "Long Ball Ratio (%)", percentage_format=True)
        plot_bar(4 * spacing, through_balls_team_1, through_balls_team_2, "Through Balls")
        plot_bar(3 * spacing, crosses_team_1, crosses_team_2, "Crosses")
        plot_bar(2 * spacing, successful_crosses_team_1, successful_crosses_team_2, "Successful Crosses")
        plot_bar(1 * spacing, crossing_ratio_team_1, crossing_ratio_team_2, "Crossing Ratio (%)", percentage_format=True)


        # Adjust axis for a clean layout
        ax.set_xlim(0, 100)  # All bars span from 0 to 100%
        ax.set_ylim(-1, 11 * spacing)  # Adjust y-axis to fit all bars with spacing
        ax.set_yticks([])  # Remove y-axis ticks
        ax.set_xticks([])  # Remove x-axis ticks

        # Hide spines for minimal design
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add match information at the bottom
        fig.text(0.5, 0.04, f"{home_team} ({selected_match_info['home_score']} - {selected_match_info['away_score']}) {away_team}", 
                ha='center', fontsize=16, fontweight='bold')
        fig.text(0.5, 0.02, f"{selected_match_info['match_date']} | Stadium: {selected_match_info['stadium_name']}", 
                ha='center', fontsize=14)

        # Add a title for the figure
        fig.suptitle('Distribution', fontsize=30, fontweight='bold', y=0.95)
        # Add home team name on the left
        fig.text(0.01, 0.95, home_team, fontsize=20, ha='left', va='center', fontweight='bold', color=colors[0])

        # Add away team name on the right
        fig.text(0.99, 0.95, away_team, fontsize=20, ha='right', va='center', fontweight='bold', color=colors[1])

        all_figures.append(fig)


        # Display the figure in Streamlit
        st.pyplot(fig)



#########################################################################################################################################################################

        # Extract defensive stats for both teams directly from team_grouped
        defensive_actions_team_1 = team_grouped[team_grouped['team_name'] == home_team]['defensive_actions'].values[0]
        defensive_actions_team_2 = team_grouped[team_grouped['team_name'] == away_team]['defensive_actions'].values[0]

        pressures_team_1 = team_grouped[team_grouped['team_name'] == home_team]['pressures'].values[0]
        pressures_team_2 = team_grouped[team_grouped['team_name'] == away_team]['pressures'].values[0]

        pressure_regains_team_1 = team_grouped[team_grouped['team_name'] == home_team]['pressure_regains'].values[0]
        pressure_regains_team_2 = team_grouped[team_grouped['team_name'] == away_team]['pressure_regains'].values[0]

        counterpressures_team_1 = team_grouped[team_grouped['team_name'] == home_team]['counterpressures'].values[0]
        counterpressures_team_2 = team_grouped[team_grouped['team_name'] == away_team]['counterpressures'].values[0]

        aggressive_actions_team_1 = team_grouped[team_grouped['team_name'] == home_team]['aggressive_actions'].values[0]
        aggressive_actions_team_2 = team_grouped[team_grouped['team_name'] == away_team]['aggressive_actions'].values[0]

        interceptions_team_1 = team_grouped[team_grouped['team_name'] == home_team]['interceptions'].values[0]
        interceptions_team_2 = team_grouped[team_grouped['team_name'] == away_team]['interceptions'].values[0]

        tackles_team_1 = team_grouped[team_grouped['team_name'] == home_team]['tackles'].values[0]
        tackles_team_2 = team_grouped[team_grouped['team_name'] == away_team]['tackles'].values[0]

        obv_defensive_actions_team_1 = team_grouped[team_grouped['team_name'] == home_team]['obv_defensive_action'].values[0]
        obv_defensive_actions_team_2 = team_grouped[team_grouped['team_name'] == away_team]['obv_defensive_action'].values[0]

                # Calculate ball recoveries in opponent's half for both teams
        ball_recoveries_opp_half_team_1 = len(df[(df['team_name'] == home_team) & 
                                                (df['type_name'] == 'Ball Recovery') & 
                                                (df['x'] < 60)])

        ball_recoveries_opp_half_team_2 = len(df[(df['team_name'] == away_team) & 
                                                (df['type_name'] == 'Ball Recovery') & 
                                                (df['x'] < 60)])



        # Create the figure and axis with increased size
        fig, ax = plt.subplots(figsize=(24, 16),dpi=800)  # Larger figure size

        # Adjust spacing between bars
        spacing = 2  # Control vertical space between bars
        bar_height = 0.40  # Make bars thinner

        # Helper function to plot proportional bars with consistent order and dynamic colors
        def plot_bar(y_pos, value1, value2, title, percentage_format=False,xg_format=False):
            """Plot a stat as a proportional bar with two segments."""
            total_value = value1 + value2  # Calculate total value

            # Calculate segment widths based on values
            width1 = (value1 / total_value) * 100 if total_value != 0 else 50
            width2 = (value2 / total_value) * 100 if total_value != 0 else 50

            # Plot home team segment
            ax.barh([y_pos], width1, color=colors[0], align='center', height=bar_height)
            # Plot away team segment
            ax.barh([y_pos], width2, left=width1, color=colors[1], align='center', height=bar_height)

            if percentage_format:
                label1 = f"{value1:.1f}%"
                label2 = f"{value2:.1f}%"
            elif xg_format:
                label1 = f"{value1:.2f}"
                label2 = f"{value2:.2f}"
            else:
                label1 = f"{value1}"
                label2 = f"{value2}"

            # Add labels outside the bars
            ax.text(-2, y_pos, label1, va='center', ha='right', fontsize=14)
            ax.text(102, y_pos, label2, va='center', ha='left', fontsize=14,)

            # Add the title above the bar
            ax.text(50, y_pos + 0.5, title, va='center', ha='center', fontsize=14, color='grey')

        plot_bar(8 * spacing, defensive_actions_team_1, defensive_actions_team_2, "Defensive Actions")
        plot_bar(7 * spacing, ball_recoveries_opp_half_team_1, ball_recoveries_opp_half_team_2, "Ball Recoveries in Opp Half")
        plot_bar(6 * spacing, pressures_team_1, pressures_team_2, "Pressures Applied")
        plot_bar(5 * spacing, pressure_regains_team_1, pressure_regains_team_2, "Pressure Regains")
        plot_bar(4 * spacing, counterpressures_team_1, counterpressures_team_2, "Counterpressures")
        plot_bar(3 * spacing, aggressive_actions_team_1, aggressive_actions_team_2, "Aggressive Actions")
        plot_bar(2 * spacing, interceptions_team_1, interceptions_team_2, "Interceptions")
        plot_bar(1 * spacing, tackles_team_1, tackles_team_2, "Tackles")
        plot_bar(0 * spacing, obv_defensive_actions_team_1, obv_defensive_actions_team_2, "OBV Defensive Actions",xg_format=True)


        # Adjust axis for a clean layout
        ax.set_xlim(0, 100)  # All bars span from 0 to 100%
        ax.set_ylim(-1, 9 * spacing)  # Adjust y-axis to fit all bars with spacing
        ax.set_yticks([])  # Remove y-axis ticks
        ax.set_xticks([])  # Remove x-axis ticks

        # Hide spines for minimal design
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add match information at the bottom
        fig.text(0.5, 0.04, f"{home_team} ({selected_match_info['home_score']} - {selected_match_info['away_score']}) {away_team}", 
                ha='center', fontsize=16, fontweight='bold')
        fig.text(0.5, 0.02, f"{selected_match_info['match_date']} | Stadium: {selected_match_info['stadium_name']}", 
                ha='center', fontsize=14)

        # Add a title for the figure
        fig.suptitle('Defensive', fontsize=30, fontweight='bold', y=0.95)
        # Add home team name on the left
        fig.text(0.01, 0.95, home_team, fontsize=20, ha='left', va='center', fontweight='bold', color=colors[0])

        # Add away team name on the right
        fig.text(0.99, 0.95, away_team, fontsize=20, ha='right', va='center', fontweight='bold', color=colors[1])
        all_figures.append(fig)

        # Display the figure in Streamlit
        st.pyplot(fig)



#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################

    # Initialize the vertical pitch
    pitch = VerticalPitch(pitch_type='statsbomb', line_zorder=2, pitch_color='white', line_color='white')

    # Create a figure with two side-by-side pitches
    fig1, (ax_home, ax_away) = plt.subplots(1, 2, figsize=(24, 16))
    fig1.set_facecolor('white')  # Set the background color to white

    # Draw the pitch on both axes
    pitch.draw(ax=ax_home)
    pitch.draw(ax=ax_away)

    # Filter data for home and away teams
    df_home = df_pass[df_pass['team_name'] == home_team_name]
    df_away = df_pass[df_pass['team_name'] == away_team_name]

    # Calculate the bin statistics for both teams
    bin_statistic_home = pitch.bin_statistic(df_home.x, df_home.y, statistic='count', bins=(20, 20))
    bin_statistic_away = pitch.bin_statistic(df_away.x, df_away.y, statistic='count', bins=(20, 20))

    # Apply Gaussian smoothing
    bin_statistic_home['statistic'] = gaussian_filter(bin_statistic_home['statistic'], 1)
    bin_statistic_away['statistic'] = gaussian_filter(bin_statistic_away['statistic'], 1)

    # Set consistent color scale (vmin and vmax) for both heatmaps
    vmin = min(bin_statistic_home['statistic'].min(), bin_statistic_away['statistic'].min())
    vmax = max(bin_statistic_home['statistic'].max(), bin_statistic_away['statistic'].max())

    # Create the home heatmap
    pcm_home = pitch.heatmap(
        bin_statistic_home, ax=ax_home, cmap='hot', edgecolors='black', vmin=vmin, vmax=vmax
    )

    # Create the away heatmap
    pcm_away = pitch.heatmap(
        bin_statistic_away, ax=ax_away, cmap='hot', edgecolors='black', vmin=vmin, vmax=vmax
    )

    # Add team names as titles above the pitches
    ax_home.set_title(home_team_name, fontsize=18, color='black', pad=20, fontweight='bold')
    ax_away.set_title(away_team_name, fontsize=18, color='black', pad=20, fontweight='bold')

    # Add individual colorbars for both heatmaps (on the right)
    cbar_home = fig1.colorbar(pcm_home, ax=ax_home, location='right', pad=0.05, shrink=0.8, aspect=30)
    cbar_home.outline.set_edgecolor('black')
    cbar_home.ax.yaxis.set_tick_params(color='black')
    plt.setp(cbar_home.ax.get_yticklabels(), color='black')

    cbar_away = fig1.colorbar(pcm_away, ax=ax_away, location='right', pad=0.05, shrink=0.8, aspect=30)
    cbar_away.outline.set_edgecolor('black')
    cbar_away.ax.yaxis.set_tick_params(color='black')
    plt.setp(cbar_away.ax.get_yticklabels(), color='black')

    # Set a title for the entire figure
    fig1.suptitle('Home and Away Team Heatmaps', fontsize=22, color='black', fontweight='bold')

    # Display the heatmaps using Streamlit
    st.pyplot(fig1)
    all_figures.append(fig1)


#########################################################################################################################################################################
#########################################################################################################################################################################

    home_color="#215454"
    away_color="#05a5a6"

    bgcolor="white"
    color1='#e21017' #red
    color2='#9a9a9a' #grey
    cmaplisth = [bgcolor,color2,home_color]
    cmaph = LinearSegmentedColormap.from_list("", cmaplisth)

    cmaplista = [bgcolor,color2,away_color]
    cmapa = LinearSegmentedColormap.from_list("", cmaplista)
    pitch = Pitch(pitch_type='statsbomb', line_zorder=2, pitch_color='white', line_color='black')


    def final_3rd_touches(ax, df):
        df_final_third = df[
            (df['type_name'].isin(['Pass', 'Carry', 'Shot', 'Dribble'])) &  # Only relevant types
            (df['x'] >= 80)  # Only final third
        ]

        # Separate data for home and away teams
        df_home = df_final_third[df_final_third['team_name'] == home_team]
        df_away = df_final_third[df_final_third['team_name'] == away_team]

        # Flip coordinates for the away team (to align with pitch direction)
        df_home['x'] = 120 - df_home['x']
        df_home['y'] = 80 - df_home['y']

        # Scatter plot for home and away teams
        pitch.scatter(df_home.x, df_home.y,
        s=300,
        edgecolors=color2,
        c=home_color, 
        marker='o',
        ax=ax,zorder=10)


        pitch.scatter(df_away.x, df_away.y,
        s=300,
        edgecolors=color2,
        c=away_color, 
        marker='o',
        ax=ax,zorder=10)

        # Create heatmaps for both teams
        bin_home = pitch.bin_statistic(df_home.x, df_home.y, statistic='count', bins=(12, 8))
        bin_home['statistic'] = gaussian_filter(bin_home['statistic'], 1)
        pitch.heatmap(bin_home, ax=ax, cmap=cmaph, edgecolors=bgcolor, zorder=1, alpha=0.75)

        bin_away = pitch.bin_statistic(df_away.x, df_away.y, statistic='count', bins=(12, 8))
        bin_away['statistic'] = gaussian_filter(bin_away['statistic'], 1)
        pitch.heatmap(bin_away, ax=ax, cmap=cmapa, edgecolors=bgcolor, zorder=1, alpha=0.5)

        # Add a dividing rectangle to separate halves
        
        rect=plt.Rectangle([40,0],40,80,color=bgcolor,zorder=1,alpha=1)
        ax.add_artist(rect)
        
        totalh =len(df_home)
        totala =len(df_away)
        total=len(df)
        percenth=round(totalh/total*100,1)
        percenta=round(totala/total*100,1)
        ax.text(87.5, -3, s=f"{away_team}:\n{totala} ({percenta}%)", fontsize=30,ha='center',weight="bold",color=away_color)
        ax.text(32.5, -3, s=f"{home_team}:\n{totalh} ({percenth}%)", fontsize=30,ha='center',weight="bold",color=home_color)
        ax.arrow (84, -2, 10, 0,width=0.35, color=away_color, head_length=2, length_includes_head=True)
        ax.arrow(37, -2, -10, 0, width=0.35, color=home_color, head_length=2, length_includes_head=True)
        

    fig = plt.figure(figsize=(24,16),constrained_layout=True)
    fig.set_facecolor('white')  # Set the background color to white
    gs = fig.add_gridspec(nrows=1,ncols=1)
    fig.patch.set_facecolor(bgcolor)

    ax1 = fig.add_subplot(gs[0])
    pitch.draw(ax=ax1)
    final_3rd_touches(ax1,df)
    plt.tight_layout()

    
    ax1.set_title('Open Play touches in final 3rd | pass, dribble, carry or shot.',fontsize=40,va='center',pad=100)
        # Display the plot
    all_figures.append(fig)

    st.pyplot(fig)


#########################################################################################################################################################################
#########################################################################################################################################################################


    # Initialize the pitch (half pitch view)
    pitch = VerticalPitch(
        pitch_type='statsbomb', pad_bottom=0.5, half=True, 
        goal_type='box', goal_alpha=0.8, pitch_color=bgcolor, line_color='black',line_zorder=100
    )
    

    # Define functions to plot the percentage circles and text
    def plot_circle(number, x, y, size, ax):
        circ = patches.Circle((x, y), size, facecolor=home_color, ec="black", lw=3, alpha=0.7, zorder=10)
        ax.add_patch(circ)
        ax.text(s=f"{number}", x=x, y=y, size=20, color=bgcolor, ha="center", va="center", 
                zorder=11, fontweight='bold')

    def plot_percentage(df_home, ax):
        total = len(df_home)
        # Calculate passes per zone
        left = len(df_home[df_home["end_y"] < 26.6])
        centre = len(df_home[(df_home["end_y"] >= 26.6) & (df_home["end_y"] < 53.3)])
        right = len(df_home[df_home["end_y"] >= 53.3])

        # Calculate percentages
        left_per = int((left / total) * 100)
        centre_per = int((centre / total) * 100)
        right_per = int((right / total) * 100)

        # Plot circles with percentages and counts
        plot_circle(f"{left_per}%", 12, 95, 5, ax)
        plot_circle(f"{centre_per}%", 40, 95, 5, ax)
        plot_circle(f"{right_per}%", 68, 95, 5, ax)
        plot_circle(left, 12, 85, 3, ax)
        plot_circle(centre, 40, 85, 3, ax)
        plot_circle(right, 68, 85, 3, ax)

    # Filter passes into the final third for the home team
    df_home = df[
        (df["team_name"] == home_team) & 
        (df["type_name"].isin(["Pass", "Carry"]))
    ]
    df_home = df_home[(df_home["x"] < 80) & (df_home["end_x"] >= 80)]
    dfl = df_home[df_home["end_y"] < 26.6]  # Left zone
    dfc = df_home[(df_home["end_y"] >= 26.6) & (df_home["end_y"] < 53.3)]  # Center zone
    dfr = df_home[df_home["end_y"] >= 53.3]  # Right zone

    # Create the figure and gridspec layout
    fig = plt.figure(figsize=(24, 16), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=3)  # 2 rows, 3 columns
    fig.patch.set_facecolor(bgcolor)

    # Create the main axis and draw the pitch once (top row)
    ax1 = fig.add_subplot(gs[0, 0:3])
    pitch.draw(ax=ax1)
    plot_percentage(df_home, ax1)
    ax1.set_title(f"{home_team} - Final 3rd Passing & Carry Entries by End Zone", fontsize=18)

    # Add divider lines to the main pitch
    for x in [26.6, 53.3]:
        ax1.axvline(x=x, ymax=0.942, color="black", linestyle='--', lw=1, zorder=1)
    ax1.axhline(y=80, color="black", linestyle='--', lw=3)

    # Create subplots for the three zones (bottom row)
    ax_left = fig.add_subplot(gs[1, 0])
    ax_center = fig.add_subplot(gs[1, 1])
    ax_right = fig.add_subplot(gs[1, 2])

    # Draw the pitch and plot passes for each zone
    pitch.draw(ax=ax_left)
    pitch.arrows(dfl["x"], dfl["y"], dfl["end_x"], dfl["end_y"], ax=ax_left, 
                color=home_color, width=2, headwidth=5, headlength=5, alpha=0.7)
    ax_left.set_title("Left Zone Passes", fontsize=16)

    pitch.draw(ax=ax_center)
    pitch.arrows(dfc["x"], dfc["y"], dfc["end_x"], dfc["end_y"], ax=ax_center, 
                color=home_color, width=2, headwidth=5, headlength=5, alpha=0.7)
    ax_center.set_title("Center Zone Passes", fontsize=16)

    pitch.draw(ax=ax_right)
    pitch.arrows(dfr["x"], dfr["y"], dfr["end_x"], dfr["end_y"], ax=ax_right, 
                color=home_color, width=2, headwidth=5, headlength=5, alpha=0.7)
    ax_right.set_title("Right Zone Passes", fontsize=16)
    all_figures.append(fig)

    # Display the plot using Streamlit
    st.pyplot(fig)



    # Initialize the pitch (half pitch view)
    pitch = VerticalPitch(
        pitch_type='statsbomb', pad_bottom=0.5, half=True, 
        goal_type='box', goal_alpha=0.8, pitch_color=bgcolor, line_color='black',line_zorder=100
    )
    

    # Define functions to plot the percentage circles and text
    def plot_circle(number, x, y, size, ax):
        circ = patches.Circle((x, y), size, facecolor=away_color, ec="black", lw=3, alpha=0.7, zorder=10)
        ax.add_patch(circ)
        ax.text(s=f"{number}", x=x, y=y, size=20, color=bgcolor, ha="center", va="center", 
                zorder=11, fontweight='bold')

    def plot_percentage(df_home, ax):
        total = len(df_home)
        # Calculate passes per zone
        left = len(df_home[df_home["end_y"] < 26.6])
        centre = len(df_home[(df_home["end_y"] >= 26.6) & (df_home["end_y"] < 53.3)])
        right = len(df_home[df_home["end_y"] >= 53.3])

        # Calculate percentages
        left_per = int((left / total) * 100)
        centre_per = int((centre / total) * 100)
        right_per = int((right / total) * 100)

        # Plot circles with percentages and counts
        plot_circle(f"{left_per}%", 12, 95, 5, ax)
        plot_circle(f"{centre_per}%", 40, 95, 5, ax)
        plot_circle(f"{right_per}%", 68, 95, 5, ax)
        plot_circle(left, 12, 85, 3, ax)
        plot_circle(centre, 40, 85, 3, ax)
        plot_circle(right, 68, 85, 3, ax)

    # Filter passes into the final third for the home team
    df_away = df[
        (df["team_name"] == away_team) & 
        (df["type_name"].isin(["Pass", "Carry"]))
    ]
    df_away = df_away[(df_away["x"] < 80) & (df_away["end_x"] >= 80)]
    dfl = df_away[df_away["end_y"] < 26.6]  # Left zone
    dfc = df_away[(df_away["end_y"] >= 26.6) & (df_away["end_y"] < 53.3)]  # Center zone
    dfr = df_away[df_away["end_y"] >= 53.3]  # Right zone

    # Create the figure and gridspec layout
    fig = plt.figure(figsize=(24, 16), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=3)  # 2 rows, 3 columns
    fig.patch.set_facecolor(bgcolor)

    # Create the main axis and draw the pitch once (top row)
    ax1 = fig.add_subplot(gs[0, 0:3])
    pitch.draw(ax=ax1)
    plot_percentage(df_away, ax1)
    ax1.set_title(f"{away_team} - Final 3rd Passing & Carry Entries by End Zone", fontsize=18)

    # Add divider lines to the main pitch
    for x in [26.6, 53.3]:
        ax1.axvline(x=x, ymax=0.942, color="black", linestyle='--', lw=1, zorder=1)
    ax1.axhline(y=80, color="black", linestyle='--', lw=3)

    # Create subplots for the three zones (bottom row)
    ax_left = fig.add_subplot(gs[1, 0])
    ax_center = fig.add_subplot(gs[1, 1])
    ax_right = fig.add_subplot(gs[1, 2])

    # Draw the pitch and plot passes for each zone
    pitch.draw(ax=ax_left)
    pitch.arrows(dfl["x"], dfl["y"], dfl["end_x"], dfl["end_y"], ax=ax_left, 
                color=home_color, width=2, headwidth=5, headlength=5, alpha=0.7)
    ax_left.set_title("Left Zone Passes", fontsize=16)

    pitch.draw(ax=ax_center)
    pitch.arrows(dfc["x"], dfc["y"], dfc["end_x"], dfc["end_y"], ax=ax_center, 
                color=home_color, width=2, headwidth=5, headlength=5, alpha=0.7)
    ax_center.set_title("Center Zone Passes", fontsize=16)

    pitch.draw(ax=ax_right)
    pitch.arrows(dfr["x"], dfr["y"], dfr["end_x"], dfr["end_y"], ax=ax_right, 
                color=home_color, width=2, headwidth=5, headlength=5, alpha=0.7)
    ax_right.set_title("Right Zone Passes", fontsize=16)
    all_figures.append(fig)

    # Display the plot using Streamlit
    st.pyplot(fig)
    
#########################################################################################################################################################################
#########################################################################################################################################################################


    # Filter passes ending inside the box (x >= 102, y within goal width)
    df_home_box_passes = df_event[
        (df_event['team_name'] == home_team) &
        (df_event['type_name'] == 'Pass') &
        (df_event['end_x'] >= 102) &  # Passes into the box
        (df_event['end_y'] >= 18) & (df_event['end_y'] <= 62)
    ]

    df_away_box_passes = df_event[
        (df_event['team_name'] == away_team) &
        (df_event['type_name'] == 'Pass') &
        (df_event['end_x'] >= 102) &  # Passes into the box
        (df_event['end_y'] >= 18) & (df_event['end_y'] <= 62)
    ]
    # Separate completed and incomplete passes for both teams
    df_home_completed = df_home_box_passes[df_home_box_passes['outcome_name'].isna()]
    df_home_incomplete = df_home_box_passes[~df_home_box_passes['outcome_name'].isna()]

    df_away_completed = df_away_box_passes[df_away_box_passes['outcome_name'].isna()]
    df_away_incomplete = df_away_box_passes[~df_away_box_passes['outcome_name'].isna()]

    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='white', line_color='black', line_zorder=2)

    # Create the figure with side-by-side pitches
    fig, axs = plt.subplots(1, 2, figsize=(24, 16))

    ### HOME TEAM PASS MAP ###
    ax_home = axs[0]
    pitch.draw(ax=ax_home)

    # Plot heatmap for home team
    bin_statistic_home = pitch.bin_statistic(df_home_box_passes.x, df_home_box_passes.y, statistic='count', bins=(12, 10))
    pitch.heatmap(bin_statistic_home, ax=ax_home, edgecolor='grey',alpha=0.5, cmap='Blues')

    # Plot completed and incomplete passes for home team
    pitch.arrows(
        df_home_completed.x, df_home_completed.y, df_home_completed.end_x, df_home_completed.end_y,
        width=3, headwidth=8, headlength=5, color='blue', ax=ax_home, zorder=2, label='Completed'
    )
    pitch.arrows(
        df_home_incomplete.x, df_home_incomplete.y, df_home_incomplete.end_x, df_home_incomplete.end_y,
        width=3, headwidth=8,alpha=0.3, headlength=5, color='black', ax=ax_home, zorder=2, label='Incomplete'
    )
    home_completed_count = len(df_home_completed)
    home_incomplete_count = len(df_home_incomplete)
    ax_home.set_title(
        f"{home_team} Passes into the Box\n{home_completed_count} Successful | {home_incomplete_count} Unsuccessful",
        fontsize=20, color='black')
    ax_home.legend(facecolor='white', edgecolor='None', fontsize=14, loc='lower left')

    ### AWAY TEAM PASS MAP ###
    ax_away = axs[1]
    pitch.draw(ax=ax_away)

    # Plot heatmap for away team
    bin_statistic_away = pitch.bin_statistic(df_away_box_passes.x, df_away_box_passes.y, statistic='count', bins=(12, 10))
    pitch.heatmap(bin_statistic_away, ax=ax_away, edgecolor='grey',alpha=0.5, cmap='Reds')

    # Plot completed and incomplete passes for away team
    pitch.arrows(
        df_away_completed.x, df_away_completed.y, df_away_completed.end_x, df_away_completed.end_y,
        width=3, headwidth=8, headlength=5, color='red', ax=ax_away, zorder=2, label='Completed'
    )
    pitch.arrows(
        df_away_incomplete.x, df_away_incomplete.y, df_away_incomplete.end_x, df_away_incomplete.end_y,
        width=3, headwidth=8,alpha=0.3, headlength=5, color='black', ax=ax_away, zorder=2, label='Incomplete',
    )

    # Update title with counts of successful and unsuccessful passes
    away_completed_count = len(df_away_completed)
    away_incomplete_count = len(df_away_incomplete)
    ax_away.set_title(
        f"{away_team} Passes into the Box\n{away_completed_count} Successful | {away_incomplete_count} Unsuccessful",
        fontsize=20, color='black'
    )
    ax_away.legend(facecolor='white', edgecolor='None', fontsize=14, loc='lower left')

    # Adjust layout and display the plot in Streamlit
    plt.tight_layout()
    all_figures.append(fig)

    st.pyplot(fig)


#########################################################################################################################################################################
#########################################################################################################################################################################


    # Initialize the pitch (half-pitch view)
    pitch = VerticalPitch(
        pad_bottom=0.5, pitch_type='statsbomb', half=True,
        goal_type='box', goal_alpha=0.8
    )

    # Filter shot data for home and away teams
    df_home_shots = df[(df['team_name'] == home_team) & (df['type_name'] == "Shot")]
    df_away_shots = df[(df['team_name'] == away_team) & (df['type_name'] == "Shot")]

    # Categorize shots for home and away teams
    df_home_goals = df_home_shots[df_home_shots['outcome_name'] == "Goal"]
    df_home_ontarget = df_home_shots[df_home_shots['outcome_name'] == "Saved"]
    df_home_other_shots = df_home_shots[~df_home_shots['outcome_name'].isin(['Goal', 'Saved'])]

    df_away_goals = df_away_shots[df_away_shots['outcome_name'] == "Goal"]
    df_away_ontarget = df_away_shots[df_away_shots['outcome_name'] == "Saved"]
    df_away_other_shots = df_away_shots[~df_away_shots['outcome_name'].isin(['Goal', 'Saved'])]

    # Create the figure and subplots with correct figsize
    fig, axs = plt.subplots(1, 2, figsize=(24, 16), gridspec_kw={'width_ratios': [1, 1]})

    # 1. Home Team Shot Map (Left Subplot)
    pitch.draw(ax=axs[0])
    pitch.scatter(df_home_goals.x, df_home_goals.y, s=(df_home_goals.shot_statsbomb_xg * 900) + 100,
                c=home_color, edgecolors='#383838', marker='*', label='Goals', ax=axs[0], zorder=2)
    pitch.scatter(df_home_ontarget.x, df_home_ontarget.y, s=(df_home_ontarget.shot_statsbomb_xg * 900) + 100,
                c=home_color, edgecolors='white', marker='o', label='On Target', ax=axs[0], alpha=0.8)
    pitch.scatter(df_home_other_shots.x, df_home_other_shots.y, s=(df_home_other_shots.shot_statsbomb_xg * 900) + 100,
                c='grey', edgecolors='white', marker='X', label='Misses', ax=axs[0], alpha=0.3)

    # Add title and summary text to the home team pitch
    axs[0].set_title(f"{home_team} Shots", fontsize=16)
    home_goal_count = len(df_home_goals)
    home_total_shots = len(df_home_shots)
    home_total_xg = df_home_shots['shot_statsbomb_xg'].sum()
    axs[0].text(0.5, 0.12, f"Goals: {home_goal_count}\nShots: {home_total_shots}\nxG: {home_total_xg:.2f}",
                transform=axs[0].transAxes, fontsize=14, ha='center', color='black')

    # 2. Away Team Shot Map (Right Subplot)
    pitch.draw(ax=axs[1])
    pitch.scatter(df_away_goals.x, df_away_goals.y, s=(df_away_goals.shot_statsbomb_xg * 900) + 100,
                c=away_color, edgecolors='#383838', marker='*', label='Goals', ax=axs[1], zorder=2)
    pitch.scatter(df_away_ontarget.x, df_away_ontarget.y, s=(df_away_ontarget.shot_statsbomb_xg * 900) + 100,
                c=away_color, edgecolors='white', marker='o', label='On Target', ax=axs[1], alpha=0.8)
    pitch.scatter(df_away_other_shots.x, df_away_other_shots.y, s=(df_away_other_shots.shot_statsbomb_xg * 900) + 100,
                c='grey', edgecolors='white', marker='X', label='Misses', ax=axs[1], alpha=0.3)

    # Add title and summary text to the away team pitch
    axs[1].set_title(f"{away_team} Shots", fontsize=16)
    away_goal_count = len(df_away_goals)
    away_total_shots = len(df_away_shots)
    away_total_xg = df_away_shots['shot_statsbomb_xg'].sum()
    axs[1].text(0.5, 0.12, f"Goals: {away_goal_count}\nShots: {away_total_shots}\nxG: {away_total_xg:.2f}",
                transform=axs[1].transAxes, fontsize=14, ha='center', color='black')
    all_figures.append(fig)

    # Adjust layout and display the figure in Streamlit
    st.pyplot(fig)


#########################################################################################################################################################################
#########################################################################################################################################################################

    # Filter event data for both home and away teams
    press_df_home = df_event[(df_event['team_name'] == home_team) & (df_event['type_name'] == 'Pressure')]
    press_df_away = df_event[(df_event['team_name'] == away_team) & (df_event['type_name'] == 'Pressure')]

    path_eff2 = [path_effects.Stroke(linewidth=1.5, foreground='black'),
    path_effects.Normal()]


    # Create the figure with two vertical pitches side by side
    fig, axs = plt.subplots(1, 2, figsize=(24, 16))

    # Customize the figure background and layout
    fig.set_facecolor('white')

    # Initialize the vertical pitch
    pitch = VerticalPitch(
        pitch_type='statsbomb', pitch_color='white', line_color='black', line_zorder=2
    )

    # Plot the pressure map for the home team (left side)
    ax_home = axs[0]
    pitch.draw(ax=ax_home)

    bin_statistic_home = pitch.bin_statistic(
        press_df_home.x, press_df_home.y, statistic='count', bins=(8, 5), normalize=False
    )

    pitch.heatmap(
        bin_statistic_home, edgecolor='#323b49', ax=ax_home, alpha=0.55,
        cmap=LinearSegmentedColormap.from_list("custom_cmap", ["white", 'blue'], N=100)
    )

    pitch.label_heatmap(
        bin_statistic_home, color='white', fontsize=16, ax=ax_home, ha='center', va='center',
        fontweight='bold', family='monospace', path_effects=path_eff2,
        
    )

    home_event_count = len(press_df_home)

    ax_home.set_title(
      f"{home_team} Pressure Applied \n Pressure Events: {home_event_count}", color='black', size=20, fontweight="bold", family="monospace"
    )


    # Plot the pressure map for the away team (right side)
    ax_away = axs[1]
    pitch.draw(ax=ax_away)

    bin_statistic_away = pitch.bin_statistic(
        press_df_away.x, press_df_away.y, statistic='count', bins=(8, 5), normalize=False
    )

    pitch.heatmap(
        bin_statistic_away, edgecolor='#323b49', ax=ax_away, alpha=0.55,
        cmap=LinearSegmentedColormap.from_list("custom_cmap", ["white", 'red'], N=100)  # Different color for away team
    )

    pitch.label_heatmap(
        bin_statistic_away, color='white', fontsize=16, ax=ax_away, ha='center', va='center',
        fontweight='bold', family='monospace', path_effects=path_eff2,
    )
# Annotate the number of pressure events for the away team
    away_event_count = len(press_df_away)
    ax_away.set_title(
    f"{away_team} Pressure Applied \n Pressure Events: {away_event_count}", color='black', size=20, fontweight="bold", family="monospace"
)


    # Adjust layout and display the figure in Streamlit
    plt.tight_layout()
    all_figures.append(fig)
    st.pyplot(fig)


#########################################################################################################################################################################
#########################################################################################################################################################################
        # Filter defensive actions for home and away teams
    df_home_defense = df[
        (df['team_name'] == home_team) & 
        (df['type_name'].isin(['Interception', 'Tackle', 'Clearance', 'Ball Recovery']))
    ]

    df_away_defense = df[
        (df['team_name'] == away_team) & 
        (df['type_name'].isin(['Interception', 'Tackle', 'Clearance', 'Ball Recovery']))
    ]
    
    df_home_engage = df[
        (df['team_name'] == home_team) & 
        (df['type_name'].isin(['Ball Recovery', 'Interception']))
    ]

    df_away_engage = df[
        (df['team_name'] == away_team) & 
        (df['type_name'].isin(['Ball Recovery', 'Interception']))
    ]

    # Calculate the average y-coordinate (distance from own goal)
    home_avg_y = df_home_engage['x'].mean()
    away_avg_y = df_away_engage['x'].mean()
    # Initialize the vertical pitch
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='white', line_color='black', line_zorder=2)

    # Create the figure with two vertical pitches side by side
    fig, axs = plt.subplots(1, 2, figsize=(24, 16))
    fig.set_facecolor('white')

    ### HOME TEAM DEFENSIVE ACTIONS ###
    ax_home = axs[0]
    pitch.draw(ax=ax_home)

    # Plot interceptions for home team
    home_interceptions = df_home_defense[df_home_defense['type_name'] == 'Interception']
    pitch.scatter(
        home_interceptions.x, home_interceptions.y, ax=ax_home, color='red', label='Interceptions',
        alpha=0.7, s=200, edgecolor='black'
    )

    # Plot ball recoveries for home team
    home_recoveries = df_home_defense[df_home_defense['type_name'] == 'Ball Recovery']
    pitch.scatter(
        home_recoveries.x, home_recoveries.y, ax=ax_home, color='yellow', label='Ball Recovery',
        alpha=0.7, s=200, edgecolor='black'
    )

    # Plot tackles for home team
    home_tackles = df_home_defense[df_home_defense['type_name'] == 'Tackle']
    pitch.scatter(
        home_tackles.x, home_tackles.y, ax=ax_home, color='green', label='Tackles',
        alpha=0.7, s=200, edgecolor='black'
    )

    # Plot clearances for home team
    home_clearances = df_home_defense[df_home_defense['type_name'] == 'Clearance']
    pitch.scatter(
        home_clearances.x, home_clearances.y, ax=ax_home, color='blue', label='Clearances',
        alpha=0.7, s=200, edgecolor='black'
    )
    ax_home.axhline(y=home_avg_y, color='black', linestyle='--', linewidth=2,)
    ax_home.text(
    5, home_avg_y + 1, f"{home_avg_y:.1f}m", color='black', fontsize=18, ha='right', va='center',
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4')
)

    ax_home.set_title(
        f"{home_team} Defensive Actions\nLine indicates the average position for Ball Recoveries and Interceptions",
        fontsize=18, color='black')
    ax_home.legend(loc='lower right', fontsize=12)

    ### AWAY TEAM DEFENSIVE ACTIONS ###
    ax_away = axs[1]
    pitch.draw(ax=ax_away)

    # Plot interceptions for away team
    away_interceptions = df_away_defense[df_away_defense['type_name'] == 'Interception']
    pitch.scatter(
        away_interceptions.x, away_interceptions.y, ax=ax_away, color='red', label='Interceptions',
        alpha=0.7, s=200, edgecolor='black'
    )

    # Plot ball recoveries for away team
    away_recoveries = df_away_defense[df_away_defense['type_name'] == 'Ball Recovery']
    pitch.scatter(
        away_recoveries.x, away_recoveries.y, ax=ax_away, color='yellow', label='Ball Recovery',
        alpha=0.7, s=200, edgecolor='black'
    )

    # Plot tackles for away team
    away_tackles = df_away_defense[df_away_defense['type_name'] == 'Tackle']
    pitch.scatter(
        away_tackles.x, away_tackles.y, ax=ax_away, color='green', label='Tackles',
        alpha=0.7, s=200, edgecolor='black'
    )

    # Plot clearances for away team
    away_clearances = df_away_defense[df_away_defense['type_name'] == 'Clearance']
    pitch.scatter(
        away_clearances.x, away_clearances.y, ax=ax_away, color='blue', label='Clearances',
        alpha=0.7, s=200, edgecolor='black'
    )
    ax_away.axhline(y=away_avg_y, color='black', linestyle='--', linewidth=2,)
    ax_away.text(
    5, away_avg_y + 1, f"{away_avg_y:.1f}m", color='black', fontsize=18, ha='right', va='center',
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4')
)

    ax_away.set_title(
    f"{away_team} Defensive Actions\nLine indicates the average position for Ball Recoveries and Interceptions",
    fontsize=18, color='black'
)
    ax_away.legend(loc='lower right', fontsize=12)

    # Adjust layout and display the plot in Streamlit
    plt.tight_layout()
    st.pyplot(fig)
    all_figures.append(fig)



                # Set the title
    players_df = players_df [['player_name','team_name','minutes','touches','passes','passing_ratio','goals','np_xg','assists','xa','dribbles']].round(2)
    players_df = players_df[(players_df['team_name'] == home_team)]
    players_df.drop(columns=['team_name'], inplace=True)
    cmap_color = LinearSegmentedColormap.from_list(
name="bugw", colors=["#ffffff", "#ffffff",'#F4fbf1', "#93d3ab", "#35b0ab"], N=256
)
    
    
    bg_color = "#FFFFFF" # I usually just like to do a white background
    text_color = "#000000" # With black text

    plt.rcParams["text.color"] = text_color
    plt.rcParams["font.family"] = "monospace"

    col_defs = [
                ColumnDefinition(
        name="player_name",
        textprops={"ha": "left", "weight": "bold"},
        width=2.5,),

                ColumnDefinition(
        name="minutes",
        group="General",
        textprops={"ha": "center", "weight": "bold"},
        width=0.9,
    ),
    ColumnDefinition(
        name="touches",
        group="General",
        cmap=normed_cmap(players_df["touches"], cmap=cmap_color, num_stds=2),
        textprops={"ha": "center", "weight": "bold"},
    ),
    ColumnDefinition(
        name="passes",
        group="General",
        cmap=normed_cmap(players_df["passes"], cmap=cmap_color, num_stds=2),
        textprops={"ha": "center", "weight": "bold"},
        width=1.5,

    ),
    ColumnDefinition(
        name="passing_ratio",
        group="General",
        formatter=lambda x: f"{x*100:.0f}%",  # Convert passing_ratio to percentage
        cmap=normed_cmap(players_df["passing_ratio"], cmap=cmap_color, num_stds=2),
        textprops={"ha": "center"},
        
    ),
    ColumnDefinition(
        name="goals",
        group="General",
        cmap=normed_cmap(players_df["goals"], cmap=cmap_color, num_stds=2),
        textprops={"ha": "center", "weight": "bold"},
        width=1.5,
    ),
    ColumnDefinition(
        name="np_xg",
        group="General",
        cmap=normed_cmap(players_df["np_xg"], cmap=cmap_color, num_stds=2),
        textprops={"ha": "center"},
        width=1.5,
    ),
    ColumnDefinition(
        name="assists",
        group="General",
        cmap=normed_cmap(players_df["assists"], cmap=cmap_color, num_stds=2),
        textprops={"ha": "center"},
        width=1.5,
    ),
    ColumnDefinition(
        name="xa",
        group="General",
        cmap=normed_cmap(players_df["xa"], cmap=cmap_color, num_stds=2.5),
        width=1.5,
    ),
    ColumnDefinition(
        name="dribbles",
        group="General",
        cmap=normed_cmap(players_df["dribbles"], cmap=cmap_color, num_stds=2.5),
        width=1.5,
    ),
]
    players_df_sorted = players_df.sort_values(by=["minutes"],ascending=[False])


# Ok lets actually create the table
    fig, ax = plt.subplots(figsize=(20, 15),dpi=500)
    fig.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    table = Table(
        players_df_sorted,
        column_definitions=col_defs,
        index_col="player_name",
        row_dividers=True,
        row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
        footer_divider=True,
        textprops={"fontsize": 12},
        col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
        column_border_kw={"linewidth": 1, "linestyle": "-"},
        ax=ax)
    table.cells[10, 3].textprops["color"] = "#8ACB88"
    ax.set_title(f"{home_team} Players Statstic", fontsize=20, color="black",pad=-90)


    st.pyplot(fig)


    pdf_buffer = BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        for fig in all_figures:
            pdf.savefig(fig)  # Save each figure into the PDF
    pdf_buffer.seek(0)  # Reset buffer position to the start
    pdf_data = pdf_buffer.getvalue()

    # Provide a download button in Streamlit
    st.download_button(label="Download PDF", data=pdf_data, file_name="Players Season Analysis.pdf", mime="application/pdf")


if __name__ == "__main__":
    main()

#####################################################################################################################################################################


# players_df = players_df [['player_name','team_name','minutes','touches','passes','passing_ratio','goals','np_xg','np_shots','xa','assists','key_passes','passes_into_box','dribbles','defensive_actions','ball_recoveries','counterpressures','pressures','pressure_regains']]

if __name__ == "__main__":
    main()
