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
from matplotlib import patches



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
    selected_match_week = st.sidebar.selectbox("Select Match Week", ["All"] + list(range(1, 36)))

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

            # Sidebar for team selection
            TeamPick = st.sidebar.selectbox('Select Team', df['team_name'].unique())

            # Filter data for the selected team and play type
            df_pass = df[(df['team_name'] == TeamPick) & (df['type_name'] == "Pass") & (~df['outcome_name'].isin(['Pass Offside', 'Out', 'Incomplete','Unknown']))]
            df_pass_pressure = df[(df['team_name'] == TeamPick) & 
                                        ((df['type_name'] == "Pass") | (df['type_name'] == "Carry")) & 
                                        (df['under_pressure'] == 1)]
            press_df_team = df_event[df_event['team_name'] == TeamPick]
            press_df = press_df_team[(press_df_team['type_name'] == 'Pressure')]
            df_shots = df[(df['team_name'] == TeamPick) & 
                            (df['type_name'] == "Shot")]    


            # Get unique passers for the selected team
            passers = df_pass['player_name'].unique().tolist()

            # Add "All" option
            passers.insert(0, "All")

            # Sidebar for passer selection
            PasserPick = st.sidebar.selectbox('Select Player', passers)

            # Filter df_pass by selected passer or show all if "All" is selected
            if PasserPick != "All":
                df_pass = df_pass[df_pass['player_name'] == PasserPick]
                df_pass_pressure = df_pass_pressure [df_pass_pressure['player_name'] == PasserPick]
                press_df = press_df[press_df['player_name']==PasserPick ]

               


            End_Zone_Display = st.sidebar.checkbox('Display End Zone')
            if End_Zone_Display:
                st.write('Plots Shows now the Passes End Zones!')

            #####################################################################################################################################################################

            # Create pitch object for Gaussian smoothed heatmap
            pitch_gaussian = Pitch(pitch_type='statsbomb', line_zorder=2, pitch_color='black', line_color='white')

            # Plot Gaussian smoothed heatmap
            fig_gaussian, ax_gaussian = pitch_gaussian.draw(figsize=(8, 16))
            fig_gaussian.set_facecolor('black')

            # Filter data for the plot
            df_pressure = df_pass

            # Calculate Gaussian smoothed heatmap
            # Check if the checkbox is checked
            if End_Zone_Display:
                bin_statistic_gaussian = pitch_gaussian.bin_statistic(df_pressure.end_x, df_pressure.end_y, statistic='count', bins=(20, 20))
            else:
                # Calculate Gaussian smoothed heatmap using pass origins
                bin_statistic_gaussian = pitch_gaussian.bin_statistic(df_pressure.x, df_pressure.y, statistic='count', bins=(20, 20))
            bin_statistic_gaussian['statistic'] = gaussian_filter(bin_statistic_gaussian['statistic'], 1)
            pcm_gaussian = pitch_gaussian.heatmap(bin_statistic_gaussian, ax=ax_gaussian, cmap='hot', edgecolors='black')

            # Add the colorbar and format off-white
            cbar_gaussian = fig_gaussian.colorbar(pcm_gaussian, ax=ax_gaussian, shrink=0.25)
            cbar_gaussian.outline.set_edgecolor('#efefef')
            cbar_gaussian.ax.yaxis.set_tick_params(color='#efefef')
            ticks_gaussian = plt.setp(plt.getp(cbar_gaussian.ax.axes, 'yticklabels'), color='#efefef')

            # Set title for Gaussian smoothed heatmap
            ax_title_gaussian = ax_gaussian.set_title(f"{PasserPick} Heat Map" if PasserPick != "All" else f"{TeamPick} Heat Map", fontsize=16, color='#efefef')

            # Display the Gaussian smoothed heatmap using Streamlit
            st.pyplot(fig_gaussian)
            


#####################################################################################################################################################################
            path_eff = [path_effects.Stroke(linewidth=2, foreground='black'),
                        path_effects.Normal()]
            # setup pitch
            pitch = Pitch(pitch_type='statsbomb', line_zorder=2,
                                pitch_color='black', line_color='white')
            # Plot the pass flow map
            fig_pass, ax_pass = pitch.draw(figsize=(8, 16))
            fig_pass.set_facecolor('black')

            # Check if the checkbox is checked
            if End_Zone_Display:
                # Create a positional bin statistic using pass endpoints
                bin_statistic_pass = pitch.bin_statistic_positional(df_pass.end_x, df_pass.end_y, statistic='count',
                                                                    positional='full', normalize=True)
            else:
                # Create a positional bin statistic using pass origins
                bin_statistic_pass = pitch.bin_statistic_positional(df_pass.x, df_pass.y, statistic='count',
                                                                    positional='full', normalize=True)

            pitch.heatmap_positional(bin_statistic_pass, ax=ax_pass, cmap='coolwarm', edgecolors='#22312b')

            pitch.scatter(df_pass.x, df_pass.y, c='white', s=2, ax=ax_pass, alpha = 0.2)

            labels = pitch.label_heatmap(bin_statistic_pass, color='white', fontsize=11,
                                        ax=ax_pass, ha='center', va='center',
                                        str_format='{:0.0%}', path_effects=path_eff)

            # Display the pass flow map with custom colormap using Streamlit
            ax_title = ax_pass.set_title(f"{PasserPick} Passes zones" if PasserPick != "All" else f"{TeamPick} Passes zones", fontsize=20, pad=10,color='white')

            st.pyplot(fig_pass)


#####################################################################################################################################################################


            # Create pitch object
            pitch = Pitch(pitch_type='statsbomb', line_zorder=2, line_color='black', pitch_color='black',linewidth=4)
            bins = (6, 4)

            # Plot pass flow map
            fig, ax = pitch.draw(figsize=(10, 20), constrained_layout=True, tight_layout=False)
            fig.set_facecolor('black')

            # plot the heatmap - darker colors = more passes originating from that square
            bs_heatmap = pitch.bin_statistic(df_pass.x, df_pass.y, statistic='count', bins=bins)
            hm = pitch.heatmap(bs_heatmap, ax=ax, cmap='Blues')
            # plot the pass flow map with a single color ('black') and length of the arrow (5)
            fm = pitch.flow(df_pass.x, df_pass.y, df_pass.end_x, df_pass.end_y,color='grey',
                            arrow_type='average', arrow_length=15, bins=bins, ax=ax)

            ax_title = ax.set_title(f"{PasserPick} Pass flow map" if PasserPick != "All" else f"{TeamPick} Pass flow map", fontsize=20, pad=-20,color ='white')



            # Display the plot using Streamlit
            st.pyplot(fig)


#####################################################################################################################################################################
            df_pass = df[(df['team_name'] == TeamPick) & ((df['type_name'] == "Pass") | (df['type_name'] == "Carry")) & (~df['outcome_name'].isin(['Pass Offside', 'Out', 'Incomplete', 'Unknown']))]
            if PasserPick != "All":
                df_pass = df[(df['player_name'] == PasserPick) & ((df['type_name'] == "Pass") | (df['type_name'] == "Carry")) & (~df['outcome_name'].isin(['Pass Offside', 'Out', 'Incomplete', 'Unknown']))]


            # Set focus team and colors
            focus_team = "TeamPick"
            bgcolor = "white"
            color1 = 'grey'  # red
            color2 = '#9a9a9a'  # grey

            def plot_circle(number, x_co_ord, y_co_ord, size, ax):
                circ = patches.Circle((x_co_ord, y_co_ord), size, facecolor=color1, ec="black", lw=3, alpha=0.7, zorder=10)
                ax.add_patch(circ)
                ax.text(s=f"{number}", x=x_co_ord, y=y_co_ord, size=20, color=bgcolor, ha="center", va="center", zorder=11, fontweight='bold')

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

                plot_circle(left, 12, 85, 3, ax)
                plot_circle(centre, 40, 85, 3, ax)
                plot_circle(right, 68, 85, 3, ax)


                pitch = VerticalPitch(pad_bottom=0.5, pitch_type='statsbomb', half=True, goal_type='box', goal_alpha=0.8,pitch_color='black')
                pitch.draw(ax=ax)

                for x in [26.6, 53.3]:
                    ax.axvline(x=x, ymax=0.85, color="white", linestyle='--', lw=1)
                
                ax.axhline(y=80, color="white", linestyle='--', lw=3)

            # Filtering and clustering the data
            df3 = df_pass[(df_pass["x"] < 80) & (df_pass["end_x"] >= 80)]

            dfl = df3[df3["end_y"] < 26.6]
            dfc = df3[(df3["end_y"] >= 26.6) & (df3["end_y"] < 53.3)]
            dfr = df3[df3["end_y"] >= 53.3]

            # Plotting the data
            pitch = VerticalPitch(pad_bottom=0.5, pitch_type='statsbomb', half=True, goal_type='box', goal_alpha=0.8,pitch_color='black')
            pitch.draw(ax=ax)

            fig = plt.figure(figsize=(18, 14), constrained_layout=True)
            gs = fig.add_gridspec(nrows=2, ncols=3)
            fig.patch.set_facecolor(bgcolor)

            ax1 = fig.add_subplot(gs[0, 0:3])
            plot_percentage(df3, ax1)
            ax1.set_title(label=f"Final 3rd Passing & Carry entries by end zone", fontsize=18)


            fig.text(s=f"How {PasserPick} attack each side of the pitch?" if PasserPick != "All" else f"How {TeamPick} attack each side of the pitch?",  x=0.27, y=1.02,fontsize=20, fontweight="bold",color ='black')
            # Display the plot in Streamlit
            st.pyplot(fig)



#####################################################################################################################################################################            
           # Set up the pitch
            pitch = Pitch(pitch_type='statsbomb', pitch_color='black', line_color='white')
            fig, ax = pitch.draw(figsize=(16, 10), constrained_layout=True, tight_layout=False)
            fig.set_facecolor('black')

            # Plot the completed passes in green

            # Filter and plot completed passes
            completed_passes = df_pass_pressure[(~df_pass_pressure['outcome_name'].isin(['Incomplete', 'Pass_offside', 'Unknown', 'Out'])) & 
                                                (df_pass_pressure['type_name'] == 'Pass')]
            for _, row in completed_passes.iterrows():
                arrow = FancyArrowPatch((row['x'], row['y']), (row['end_x'], row['end_y']),
                                        arrowstyle="->,head_length=10,head_width=8", color='blue', linewidth=2)
                ax.add_patch(arrow)

            # Filter and plot carries with a different arrow style
            carries = df_pass_pressure[df_pass_pressure['type_name'] == 'Carry']
            for _, row in carries.iterrows():
                arrow = FancyArrowPatch((row['x'], row['y']), (row['end_x'], row['end_y']),
                                        arrowstyle="->,head_length=10,head_width=8", linestyle=':', color='yellow', linewidth=1)
                ax.add_patch(arrow)

            # Filter and plot uncompleted passes
            uncompleted_passes = df_pass_pressure[df_pass_pressure['outcome_name'].isin(['Incomplete', 'Pass_offside', 'Unknown', 'Out'])]
            for _, row in uncompleted_passes.iterrows():
                arrow = FancyArrowPatch((row['x'], row['y']), (row['end_x'], row['end_y']),
                                        arrowstyle="->,head_length=10,head_width=8", color='grey', linewidth=2, alpha=0.5)
                ax.add_patch(arrow)

                
            all_passes = df_pass_pressure[(df_pass_pressure['outcome_name'].isin(['Incomplee', 'Passoffe', 'Unnow', 'ut'])) & 
                               (df_pass_pressure['type_name'] == 'Pass')]
            pitch.arrows(all_passes['x'], all_passes['y'],
                        all_passes['end_x'], all_passes['end_y'], 
                        width=3, headwidth=7, headlength=7, color='blue', ax=ax, label='Completed passes')

            all_carry = df_pass_pressure[df_pass_pressure['type_name'] == 'Crry']
            pitch.arrows(all_carry['x'], all_carry['y'],
                        all_carry['end_x'], all_carry['end_y'], 
                        width=3, headwidth=7, headlength=7, color='yellow', ax=ax, label='Carries')
            

            # Plot uncompleted passes with reduced opacity in grey
            uncompleted_passes = df_pass_pressure[df_pass_pressure['outcome_name'].isin(['Incmplte', 'Pasffside', 'Uknwn', 'ut'])]
            pitch.arrows(uncompleted_passes['x'], uncompleted_passes['y'],
                        uncompleted_passes['end_x'], uncompleted_passes['end_y'], 
                        width=3, headwidth=7, headlength=7, color='grey', alpha=0.3, ax=ax, label='Uncompleted passes')

            # Set up the legend
            ax.legend(facecolor='white', handlelength=3, edgecolor='None', fontsize=10, loc='lower left')
            ax_title = ax.set_title(f"{PasserPick} Under Pressure Events" if PasserPick != "All" else f"{TeamPick} Under Pressure Events", fontsize=25, pad=1,color='white')


            st.pyplot(fig)


#####################################################################################################################################################################

            path_eff2 = [path_effects.Stroke(linewidth=0.2, foreground='black'),
            path_effects.Normal()]


            fig ,ax = plt.subplots(figsize=(13, 8),constrained_layout=False, tight_layout=True)
            fig.set_facecolor('#0e1117')
            ax.patch.set_facecolor('#0e1117')
            pitch = Pitch(pitch_type='statsbomb', pitch_color='#0e1117', line_color='white', line_zorder=2)
            pitch.draw(ax=ax)


            bin_statistic = pitch.bin_statistic(press_df.x, press_df.y, statistic='count', bins=(8, 5), normalize=False)
            pitch.heatmap(bin_statistic, edgecolor='#323b49', ax=ax, alpha=0.55,
                    cmap=LinearSegmentedColormap.from_list("custom_cmap", ["#f3f9ff", 'blue'], N=100))

            pitch.label_heatmap(bin_statistic, color='white', fontsize=12, ax=ax, ha='center', va='center',
                                fontweight='bold', family='monospace',path_effects=path_eff2)
            
            pitch.scatter(press_df.x, press_df.y, s=50, color='white', ax=ax,alpha=0.3)

            
            pitch.annotate(text='The direction of play  ', xytext=(45, 82), xy=(85, 82), ha='center', va='center', ax=ax,
                            arrowprops=dict(facecolor='white'), fontsize=12, color='white', fontweight="bold", family="monospace")

            plt.title(f'{PasserPick} : Pressure Applied Map' if PasserPick != "All" else f"{TeamPick} : Pressure Applied Map",
                    color='white', size=20,  fontweight="bold", family="monospace")
            
            st.pyplot(fig)
#####################################################################################################################################################################
            if PasserPick != "All":
                df_shots = df_shots[df_shots['player_name'] == PasserPick]

            df_goals = df_shots[(df_shots['outcome_name'] == "Goal")]
            df_ont = df_shots[(df_shots['outcome_name'] == "Saved")]
            df_other_shots = df_shots[~df_shots['outcome_name'].isin(['Goal', 'Saved'])]
            pitch = VerticalPitch(pad_bottom=0.5,  # pitch extends slightly below halfway line
                                pitch_type='statsbomb',
                                half=True,  # half of a pitch
                                goal_type='box',
                                goal_alpha=0.8)  # control the goal transparency


            shot_fig, shot_ax = pitch.draw(figsize=(10, 8))
            sc_goals = pitch.scatter(df_goals.x, df_goals.y,
                                    s=(df_goals.shot_statsbomb_xg * 900) + 100,
                                    c='white',
                                    edgecolors='#383838',
                                    marker='*',  # You can choose a different marker for goals
                                    label='Goals',
                                    ax=shot_ax,
                                    zorder = 2)

            # Scatter plot for "Shot Saved" and "Post"
            sc_ontarget = pitch.scatter(df_ont.x, df_ont.y,
                                        s=(df_ont.shot_statsbomb_xg * 900) + 100,
                                        c='blue',
                                        edgecolors='white',
                                        marker='o',  # You can choose a different marker for on-target shots
                                        label='On Target',
                                        alpha = 0.8,
                                        ax=shot_ax)

            # Scatter plot for "Miss"
            sc_miss = pitch.scatter(df_other_shots.x, df_other_shots.y,
                                    s=(df_other_shots.shot_statsbomb_xg * 900) + 100,
                                    c='grey',
                                    edgecolors='white',
                                    marker='X',
                                    label='Misses',
                                    ax=shot_ax,
                                    alpha = 0.3)

            title_text = f"{PasserPick if PasserPick != 'All' else TeamPick} Shots"
            txt = shot_ax.text(0.5, 1, title_text, transform=shot_ax.transAxes, fontsize=20, ha='center', color='Black')

            # Text for goal count at the bottom of the plot

            goal_count = len(df_goals)  # Count the number of goals
            goal_count_text = f"Total Goals: {goal_count}"
            # Adjust the y-position (e.g., y=-0.1) to move the text below the pitch. Adjust 'ha' and 'va' for alignment.
            # Calculate counts and xG sums
            total_shots = len(df_shots)
            shots_on_target = len(df_ont)
            total_xg = df_shots['shot_statsbomb_xg'].sum()

            # Adjust the goal count text to include these metrics
            goal_count_text = f"Goals: {goal_count}"
            shots_text = f"Shots: {total_shots} "
            xg_text = f"xG: {total_xg:.2f}"  # formatted to 2 decimal places

            # Position and display the metrics on the plot
            txt_goal_count = shot_ax.text(0.5, 0.12, f"{goal_count_text}\n{shots_text}\n{xg_text}", transform=shot_ax.transAxes, fontsize=16, ha='center', va='top', color='black')

            # Now, display the plot in Streamlit
            st.pyplot(shot_fig)


#####################################################################################################################################################################

                # Set the title
            players_df = players_df [['player_name','team_name','minutes','touches','passes','passing_ratio','goals','np_xg','assists','xa','dribbles']].round(2)
            players_df = players_df[(players_df['team_name'] == TeamPick)]
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
            ax.set_title(f"{TeamPick} Players Statstic", fontsize=20, color="black",pad=-90)


            st.pyplot(fig)





# players_df = players_df [['player_name','team_name','minutes','touches','passes','passing_ratio','goals','np_xg','np_shots','xa','assists','key_passes','passes_into_box','dribbles','defensive_actions','ball_recoveries','counterpressures','pressures','pressure_regains']]




if __name__ == "__main__":
    main()
