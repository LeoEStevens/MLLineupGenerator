"""
game.py
CSC 492 - 01 Senior Design
Author: Leo Stevens

This class is used to hold the information for each game. 
The data is passed in from player_list.py and stored in each class instance.
There are two types of games, completed and incomplete. 
Incomplete games use the default param of game = None and completed games use the game from nflgame.
The class has one data frame for general game information and one for each team that holds ranks and win information.

"""
import pandas as pd
import datetime
import nflgame
from get_weather_data import get_weather_data
from team import get_id
#from ranks import Ranks
class Game:
    """
    This class holds the information for a single game.
    """
    #The columns in the game data frame
    GAME_HEADER = \
            "season, week, game_time, game_day, time,\
            stadium, surface, conditions, temp, wind,\
            humidity, visibility, barometric_pressure, dew_point"
    #The columns in the game data frame for each team
    GAME_HEADER_TEAM = \
            "def_conf, def_div, def_conf_div,\
            def_win_ratio, def_rush_rank, def_pass_rank, def_pts_allowed,\
            opp_win_ratio, opp_streak, opp_rushing_rank, opp_passing_rank, opp_scoring_rank"
    def __init__(self, home, away, ranks, schedule, teams, season, week, game = None):
        """
        Constructor
        Processes all information into the data frame
        """

        #Get home and away team
        self.home = teams[get_id(nflgame.standard_team(home))]
        self.away = teams[get_id(nflgame.standard_team(away))]

        #Get stadium and surface information
        self.stadium = self.home.stadium
        self.surface = self.home.surface

        #Check if game has completed
        if not game is None:
            self.score_home = game.score_home
            self.score_away = game.score_away
        else:
            self.score_home = 0
            self.score_away = 0
        #Get information from nflgame schedule
        id_string = str(self.home.id) + "-" + str(self.away.id)
        schedule_info = schedule[season][week][id_string]
        if 'meridiem' in schedule_info and schedule_info['meridiem'] is not None:
            meridiem = schedule_info['meridiem']
        else:
            meridiem = 'PM'
        self.time_string = str(schedule_info['month']) + " " + str(schedule_info['day']) + " " + str(season) + " " + str(schedule_info['time']) + str(meridiem)
        self.game_time = schedule_info['time'].strip(":")
        self.game_day = schedule_info['wday']
        self.time = datetime.datetime.strptime(self.time_string, '%m %d %Y %I:%M%p')
        
        #Set game weather information
        if 'Fixed' not in self.stadium:
            self.weather = get_weather_data(season, week, self.away.name, self.home.name)
        else:
            self.weather = {'conditions':'Clear', 'temp':72, 'wind':0,
                 'humidity':40, 'visibility':10, 'barometric_pressure':1013,
                 'dew_point':55}
        #Build data frame
        self.data = pd.DataFrame(columns=Game.GAME_HEADER.replace(' ', '').split(','))
        self.data_home = pd.DataFrame(columns=Game.GAME_HEADER_TEAM.replace(' ', '').split(','))
        self.data_away = pd.DataFrame(columns=Game.GAME_HEADER_TEAM.replace(' ', '').split(','))
        game_info = self.weather
        rank_info_home = ranks.get_ranks(nflgame.standard_team(self.away.abbr))
        rank_info_home['season'] = season
        rank_info_home['week'] = week
        rank_info_home['opp_conf'] = self.away.conference
        rank_info_home['opp_div'] = self.away.division
        rank_info_home['opp_div_conf'] = self.away.conf_div
        rank_info_home['opp_win_ratio'] = self.away.win_ratio
        rank_info_home['opp_streak'] = self.away.streak
        rank_info_home['win_ratio'] = self.home.win_ratio
        rank_info_away = ranks.get_ranks(nflgame.standard_team(self.home.abbr))
        rank_info_away['season'] = season
        rank_info_away['week'] = week
        rank_info_away['opp_conf'] = self.home.conference
        rank_info_away['opp_div'] = self.home.division
        rank_info_away['opp_div_conf'] = self.home.conf_div
        rank_info_away['opp_win_ratio'] = self.home.win_ratio
        rank_info_away['opp_streak'] = self.home.streak
        rank_info_away['win_ratio'] = self.away.win_ratio
        game_info['stadium'] = self.stadium
        game_info['surface'] = self.surface
        game_info['season'] = season
        game_info['week'] = week
        game_info['game_time'] = self.game_time
        game_info['game_day'] = self.game_day
        game_info['time'] = self.time
        self.data = self.data.append(game_info, ignore_index=True)
        self.data_team = {}
        self.data_team[get_id(nflgame.standard_team(self.home.abbr))] = self.data_home.append(rank_info_home, ignore_index=True)
        self.data_team[get_id(nflgame.standard_team(self.away.abbr))] = self.data_away.append(rank_info_away, ignore_index=True)
