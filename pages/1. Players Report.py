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
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import FancyArrowPatch
from mplsoccer.pitch import VerticalPitch
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
import matplotlib.image as mpimg
from matplotlib import patches
import numpy as np


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = 'Palatino Linotype'
# Set your login credentials
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


def stats(competition_id: int, season_id: int) -> list:
    url = f"https://data.statsbomb.com/api/v4/competitions/{competition_id}/seasons/{season_id}/player-stats"
    stats = get_resource(url, {credentials["user"], credentials["passwd"]})
    return stats



def main():
    # Call competitions function to fetch data
    
    competition_data = competitions()
    
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

    # Sort the DataFrame by team name and then by minutes played
    sorted_players_df = players_df.sort_values(by=['team_name', 'minutes'], ascending=[True, False])
    
    # Generate reports for each player
    for player_id, player_data in sorted_players_df.groupby('player_id'):
        player_info = player_data.iloc[0]  # Get player info
        player_name = player_info['player_name']
        team_name = player_info['team_name']
        minutes = player_info['minutes']

        # Create a multi-plot layout
        fig, axes = plt.subplots(3, 3, figsize=(24, 18), dpi=200)
        fig.patch.set_facecolor('black')
        fig.suptitle(f"{player_name} - {team_name}", fontsize=35, color='white')

        # Plot 0,0: Shot map
        df_shots = df[(df['player_id'] == player_id) & (df['type_name'] == 'Shot')]
        df_goals = df_shots[df_shots['outcome_name'] == "Goal"]
        df_ont = df_shots[df_shots['outcome_name'] == "Saved"]
        df_other_shots = df_shots[~df_shots['outcome_name'].isin(['Goal', 'Saved'])]

        pitch = VerticalPitch(pad_bottom=0.5, pitch_type='statsbomb', half=True, goal_type='box', goal_alpha=0.8, pitch_color='black', line_color='white')
        pitch.draw(ax=axes[0, 0])

        pitch.scatter(df_goals.x, df_goals.y,
                      s=(df_goals.shot_statsbomb_xg * 900) + 100,
                      c='white', edgecolors='#383838', marker='*',
                      label='Goals', ax=axes[0, 0], zorder=2)

        pitch.scatter(df_ont.x, df_ont.y,
                      s=(df_ont.shot_statsbomb_xg * 900) + 100,
                      c='blue', edgecolors='white', marker='o',
                      label='On Target', alpha=0.8, ax=axes[0, 0])

        pitch.scatter(df_other_shots.x, df_other_shots.y,
                      s=(df_other_shots.shot_statsbomb_xg * 900) + 100,
                      c='white', edgecolors='white', marker='X',
                      label='Misses', ax=axes[0, 0], alpha=0.3)

        goal_count = len(df_goals)
        total_shots = len(df_shots)
        shots_on_target = len(df_ont)
        total_xg = df_shots['shot_statsbomb_xg'].sum()

        goal_count_text = f"Goals: {goal_count}"
        shots_text = f"Shots: {total_shots}"
        xg_text = f"xG: {total_xg:.2f}"

        axes[0, 0].text(0.5, 0.40, f"{goal_count_text}\n{shots_text}\n{xg_text}",
                        transform=axes[0, 0].transAxes, fontsize=16, ha='center', va='top', color='white')
        axes[0, 0].set_title('Shot Map', fontsize=16, color='white')

        # Plot 0,1: Stats table as text
        stats = {
            'Minutes': player_info['minutes'],
            'Goals': player_info['goals'],
            'xG': player_info['np_xg'],
            'Shots': player_info['np_shots'],
            'Passes': player_info['passes'],
            'xA': player_info['xa'],
            'Key Passes': player_info['key_passes'],
            'Passes into Box': player_info['op_passes_into_box'],
            'Dribbles': player_info['dribbles'],
            'OBV': player_info['obv'],
            'Defensive Actions': player_info['defensive_actions'],
            'Pressure Regains': player_info['pressure_regains']
        }

        # Format stats to display as XX.X
        formatted_stats = {key: f"{value:.1f}" for key, value in stats.items()}
        # Add the text to the subplot
        axes[0, 1].axis('off')
        stats_text = '\n'.join([f"{key}: {value}" for key, value in formatted_stats.items()])
        axes[0, 1].text(0.5, 0.5, stats_text, fontsize=16, color='white', ha='center', va='center')
        axes[0, 1].set_title('Player Stats', fontsize=20, color='white')




        # Plot 0,2: Pass map
        pitch = Pitch(pitch_type='statsbomb', pitch_color='black', line_color='white')
        pitch.draw(ax=axes[0, 2])
        axes[0, 2].set_facecolor('black')
        # Filter and plot completed passes
        df_passes = df[(df['player_id'] == player_id) & (df['type_name'] == 'Pass')]
        completed_passes = df_passes[~df_passes['outcome_name'].isin(['Incomplete', 'Pass_offside', 'Unknown', 'Out'])]
        for _, row in completed_passes.iterrows():
            arrow = FancyArrowPatch((row['x'], row['y']), (row['end_x'], row['end_y']),
                                    arrowstyle="->,head_length=10,head_width=8", color='blue', linewidth=2)
            axes[0, 2].add_patch(arrow)

        # Filter and plot carries with a different arrow style
        carries = df[(df['player_id'] == player_id) & (df['type_name'] == 'Carry')]
        for _, row in carries.iterrows():
            arrow = FancyArrowPatch((row['x'], row['y']), (row['end_x'], row['end_y']),
                                    arrowstyle="->,head_length=10,head_width=8", linestyle=':', color='yellow', linewidth=1)
            axes[0, 2].add_patch(arrow)

        # Filter and plot uncompleted passes
        uncompleted_passes = df_passes[df_passes['outcome_name'].isin(['Incomplete', 'Pass_offside', 'Unknown', 'Out'])]
        for _, row in uncompleted_passes.iterrows():
            arrow = FancyArrowPatch((row['x'], row['y']), (row['end_x'], row['end_y']),
                                    arrowstyle="->,head_length=10,head_width=8", color='grey', linewidth=2, alpha=0.5)
            axes[0, 2].add_patch(arrow)

        # Add colored text annotations as subtitle
        fig.text(0.75, 0.66, 'Completed Passes', ha='center', fontsize=10, color='blue', weight='bold')
        fig.text(0.82, 0.66, 'Uncompleted Passes', ha='center', fontsize=10, color='grey', weight='bold')
        fig.text(0.87, 0.66, 'Carries', ha='center', fontsize=10, color='yellow', weight='bold')
        axes[0, 2].set_title('Passes and Carries Map', color='white')

        # Plot 1,0: Dribble map
        df_dribbles = df[(df['player_id'] == player_id) & (df['type_name'] == 'Dribble')]
        pitch.draw(ax=axes[1, 0])
        pitch.scatter(df_dribbles['x'], df_dribbles['y'], ax=axes[1, 0], color='green',s=120)
        axes[1, 0].set_title('Dribble Map', fontsize=16, color='white')

        # Plot 1,1: Heatmap
        df_actions = df[df['player_id'] == player_id]
        pitch_gaussian = Pitch(pitch_type='statsbomb', line_zorder=2, pitch_color='black', line_color='white')
        pitch_gaussian.draw(ax=axes[1, 1])
        axes[1, 1].set_facecolor('black')

        bin_statistic_gaussian = pitch_gaussian.bin_statistic(df_actions.x, df_actions.y, statistic='count', bins=(20, 20))
        bin_statistic_gaussian['statistic'] = gaussian_filter(bin_statistic_gaussian['statistic'], 1)
        pcm_gaussian = pitch_gaussian.heatmap(bin_statistic_gaussian, ax=axes[1, 1], cmap='hot', edgecolors='black')

        axes[1, 1].set_title('Heatmap', fontsize=16, color='white')

        # Plot 1,2: Defensive actions map
        df_defensive_actions = df[(df['player_id'] == player_id) & (df['type_name'].isin(['Interception', 'Tackle', 'Clearance','Ball Recovery']))]
        pitch.draw(ax=axes[1, 2])
        axes[1, 2].set_facecolor('black')

        # Plot interceptions
        interceptions = df_defensive_actions[df_defensive_actions['type_name'] == 'Interception']
        pitch.scatter(interceptions.x, interceptions.y, ax=axes[1, 2], color='red', label='Interceptions', alpha=0.7, s=120, edgecolor='black')

        recovrey = df_defensive_actions[df_defensive_actions['type_name'] == 'Ball Recovery']
        pitch.scatter(recovrey.x, recovrey.y, ax=axes[1, 2], color='yellow', label='Ball Recovery', alpha=0.7, s=120, edgecolor='black')

        # Plot tackles
        tackles = df_defensive_actions[df_defensive_actions['type_name'] == 'Tackle']
        pitch.scatter(tackles.x, tackles.y, ax=axes[1, 2], color='green', label='Tackles', alpha=0.7, s=120, edgecolor='black')

        # Plot clearances
        clearances = df_defensive_actions[df_defensive_actions['type_name'] == 'Clearance']
        pitch.scatter(clearances.x, clearances.y, ax=axes[1, 2], color='blue', label='Clearances', alpha=0.7, s=120, edgecolor='black')

        axes[1, 2].set_title('Defensive Actions', fontsize=16, color='white')
        axes[1, 2].legend(loc='lower right', fontsize=5)

        # Plot 2,0: Enter into final third (You need to provide the code for this)
        # Example code (replace with your actual code)
        # Filtering progressive passes
        df_progrssive = df[(df['player_id'] == player_id) & 
                        (df['type_name'] == "Pass") & 
                        (~df['outcome_name'].isin(['Pass Offside', 'Out', 'Incomplete', 'Unknown', 'Injury Clearance']))]

        # Calculate the beginning and end distances
        df_progrssive['beginning'] = np.sqrt(np.square(120 - df_progrssive['x']) + np.square(40 - df_progrssive['y']))
        df_progrssive['end'] = np.sqrt(np.square(120 - df_progrssive['end_x']) + np.square(40 - df_progrssive['end_y']))

        # Determine if the pass is progressive
        df_progrssive['progressive'] = (df_progrssive['beginning'] - df_progrssive['end']) >= 0.3 * df_progrssive['beginning']

        # Plotting
        pitch.draw(ax=axes[2, 0])
        progressive_passes = df_progrssive[df_progrssive['progressive']]
        pitch.lines(progressive_passes['x'], progressive_passes['y'], 
                    progressive_passes['end_x'], progressive_passes['end_y'], 
                    comet=True, ax=axes[2, 0])

        axes[2, 0].set_title('Progressive Passes',fontsize = 16, color='white')



        # Plot 2,1: Comparing player stats to seasonal stats (Provide the actual code)
        # Example code (replace with your actual code)

        bins = list(range(0, 100, 5))
        df['interval'] = pd.cut(df['minute'], bins=bins, right=False)
        # Count the number of actions in each interval
        
        # Count the number of actions in each interval
        interval_counts = df[df['player_id'] == player_id]['interval'].value_counts().sort_index()
        interval_counts_team = df[df['possession_team_name'] == team_name]['interval'].value_counts().sort_index()

        # Calculate the average number of actions per interval for the team
        num_players_in_team = df[df['possession_team_name'] == team_name]['player_id'].nunique()
        average_counts_team = interval_counts_team / 11

        # Plot the line chart to create a wave effect
        axes[2, 1].plot(interval_counts.index.astype(str), interval_counts.values, color='white', marker='o', linestyle='--', linewidth=2, markersize=8, label='Player Actions')
        axes[2, 1].plot(average_counts_team.index.astype(str), average_counts_team.values, color='blue', marker='o', linestyle='-', linewidth=2, markersize=8, alpha=0.5, label='Team Average Actions')
        axes[2, 1].set_title('Player Involvement in Game (Actions per 5 Minutes)', fontsize=16, color='white', pad=20)
        axes[2, 1].set_xlabel('Minutes', fontsize=14, color='white')
        axes[2, 1].set_ylabel('Number of Actions', fontsize=14, color='white')
        axes[2, 1].tick_params(colors='white', labelsize=10)  # Adjust the font size of the ticks
        axes[2, 1].grid(axis='y', linestyle='--', alpha=0.7)

        # Set the tick positions and labels
        tick_positions = list(range(len(interval_counts)))
        tick_labels = [f'{5 * i}' for i in tick_positions]
        axes[2, 1].set_xticks(tick_positions)
        axes[2, 1].set_ylim(0, max(interval_counts.max(), average_counts_team.max()) + 10)
        axes[2, 1].set_xticklabels(tick_labels, rotation=45)
        axes[2, 1].set_facecolor('black')  # Set your desired color here

        # Add legend
        axes[2, 1].legend(loc='upper right', fontsize=10, facecolor='white', framealpha=0.7)



        # Plot 2,2: Pass receiving heat map (Provide the actual code)
        # Example code (replace with your actual code)
        path_eff2 = [path_effects.Stroke(linewidth=0.2, foreground='black'),
        path_effects.Normal()]

        press_df = df[(df['player_id'] == player_id) & (df['type_name'] == 'Pressure')]
        pitch.draw(ax=axes[2, 2])
        bin_statistic = pitch.bin_statistic(press_df.x, press_df.y, statistic='count', bins=(8, 5), normalize=False)
        pitch.heatmap(bin_statistic, edgecolor='#323b49', ax=axes[2, 2], alpha=0.55,
                cmap=LinearSegmentedColormap.from_list("custom_cmap", ["#f3f9ff", 'blue'], N=100))

        pitch.label_heatmap(bin_statistic, color='white', fontsize=12, ax=axes[2, 2], ha='center', va='center',
                            fontweight='bold', family='monospace',path_effects=path_eff2)
        
        pitch.scatter(press_df.x, press_df.y, s=50, color='white', ax=axes[2, 2],alpha=0.3)
            
        axes[2, 2].set_title('Pressure Applying Map', fontsize=16, color='white')

        # Save and display the figure
        st.pyplot(fig)


        # Define the directory and filename
#         directory = "/Users/galfishman/Desktop/33"
#         filename = f"{player_info['team_name']} - {player_info['player_name']} - .png"
#         full_path = f"{directory}/{filename}"
# # Save the figure
#         plt.savefig(full_path, bbox_inches='tight', facecolor=fig.get_facecolor())

if __name__ == "__main__":
    main()
