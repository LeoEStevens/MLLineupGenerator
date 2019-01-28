"""
player_list.py
CSC 492 - 01 Senior Design
Author: Leo Stevens

This class holds builds and holds all of the data for the program. The sequence for updating/creating the data is:
    update (starts build_data at the last week and season the program was run) 
        -> build_data (builds the data, cleans the data, saves the data) 
            -> build_season (loops through every week in the season and calls build_week) 
                -> build_week (build and process data for the week)
        -> build_next_week (gets projections, salaries, game information)
Update is used for both initial data creation and data updating. 
This file has most of the data generation for the program, it gets stats, injury status, projections, salaries, and game information.
That information is then passed into dictionaries of classes. 
Those dictionaries are then iterated over and combined.
"""
import nflgame
import csv
import pandas as pd
import numpy as np
import os
import sys
import requests
from player import Player
from defense import Defense
from game import Game
from team import Team
from team import get_id
from ranks import Ranks

class player_list:
    """
    This class stores all of the player, defense, team, game, 
    and rank information and has methods to update the data.
    The data is saved using pickle in the data directory.
    """
    #Dictionary to convert team names from RotoGrinders to nflgame
    PROJ_TEAM_CONV = {
            'NOS':'NO',
            'KCC':'KC',
            'GBP':'GB',
            'TBB':'TB',
            'NEP':'NE'}


    def __init__(self):
        """
        Constructor
        Initializes class variables and creates directories as needed.
        All variables saved by pickle need to be initialized here.
        init does not run when the class is loaded from pickle.
        """
        self.current_year, self.current_week = nflgame.live.current_year_and_week()
        self.schedule = self.build_games()
        self.players = {}
        self.defense = {}
        self.teams = {}
        self.game_list = {}
        self.ranks = Ranks()
        #Default starting week and year
        self.last_run = [1, 2009]
        #Create directories if they do not exist
        if not os.path.isdir('data'):
            os.mkdir('data')
        if not os.path.isdir('data/csv'):
            os.mkdir('data/csv')
        
    def build_games(self):
        """
        This method builds the schedule of all nfl games.
        The games are put into dictionaries based on year, each of those
        holds a dictionary of weeks.
        """
        #Get schedule from nflgame
        temp_schedule = nflgame.sched.games
        temp_games_list = {}
        for key in temp_schedule:
            game = temp_schedule[key]
            if not game['year'] in temp_games_list:
                temp_games_list[game['year']] = {}
            if not game['week'] in temp_games_list[game['year']]:
                temp_games_list[game['year']][game['week']] = {}
            if game['season_type'] == 'REG':
                id_string = str(get_id(game['home'])) + "-" + str(get_id(game['away']))
                temp_games_list[game['year']][game['week']][id_string] = game
        return temp_games_list


    def build_week(self, week, season):
        """
        This method is called by build season to build information for the week passed in as a parameter.
        """
        #Get game information from nflgames
        games = nflgame.games(season, week)
        #Build game and team information
        self.build_week_games(week, season, games)
        #Build player information
        self.build_week_players(week, season, games)
    
    def build_week_games(self, week, season, games):
        """
        This method processes data into game and team information.
        """
        #Dictionary to hold the games for the week
        self.game_list[season][week] = {}
        print_string = "\rBuilding [S: " + str(season) + "-W: " + str(week) + "]"
        sys.stdout.write(print_string)
        sys.stdout.flush()
        #Process all of the games
        for game in games:
            #Get ID for each team to be used as dictionary key
            away_team = get_id(game.away)
            home_team = get_id(game.home)
        
            #Create dictionary entries if they do not exist, else update team info
            if not away_team in self.teams:
                temp_team = Team(game.away)
                self.teams[away_team] = temp_team
            else:
                self.teams[away_team].update_info(game.away)
            if not home_team in self.teams:
                temp_team = Team(game.home)
                self.teams[home_team] = temp_team
            else:
                self.teams[home_team].update_info(game.home)
            
            #Create new game and add it to the list and teams
            temp_game = Game(game.home, game.away, self.ranks, self.schedule, self.teams, season, week, game)
            self.game_list[season][week][home_team] = temp_game
            self.game_list[season][week][away_team] = temp_game
            self.teams[away_team].add_game(game.home, game.away, week, season, game)
            self.teams[home_team].add_game(game.home, game.away, week, season, game)

    def build_week_players(self, week, season, games):
        """
        This method gets all of the information for each player.
        """
        #Dictionary to hold the defense information for the week
        defense_week = {}
        #Get player stats from nflgame
        player_stats = nflgame.combine_play_stats(games)
        #Process every player
        for player in player_stats:
            self.process_player(season, week, player, defense_week)
        #Copy final defense values into defense dictionary
        for team in defense_week:
            if team in self.defense:
                if self.teams[team].home:
                    defense_week[team].df['points_allowed'] = self.game_list[season][week][team].score_away
                else:
                    defense_week[team].df['points_allowed'] = self.game_list[season][week][team].score_home
                self.defense[team].update(defense_week[team].df, 1, 1)
            else:
                temp_def = Defense(self.teams[team])
                if self.teams[team].home:
                    defense_week[team].df['points_allowed'] = self.game_list[season][week][team].score_away
                else:
                    defense_week[team].df['points_allowed'] = self.game_list[season][week][team].score_home
                temp_def.update(defense_week[team].df, 1, 1)
                self.defense[team] = temp_def
        #Update ranks
        for game in games:
            self.ranks.add_game(game)
            
    def process_player(self, season, week, player, defense_week):
        """
        This method gets all of the information for each player.
        """
        #Wrap stats in a dataframe
        stats = pd.DataFrame([player.stats])
        #Set season and week
        stats['season'] =  season
        stats['week'] = week
        #Process based on offense or defense
        if ('QB' in player.guess_position or \
                'RB' in player.guess_position or \
                'WR' in player.guess_position or \
                'TE' in player.guess_position):
            #Check if player class is set in Player, class is needed for the number of years played
            if player.player is None:
                stats['years_played'] = 1
            else:
                stats['years_played'] = player.player.years_pro - (self.current_year - season) + 1
            #If the player exists in the dictionary update their info, else create new player
            if player.playerid in self.players:
                self.players[player.playerid].setInfo(player, self.current_year, season, self.teams[get_id(player.team)])
                self.players[player.playerid].update(stats, 1, 1)
            else:
                temp_player = Player(player, self.teams[get_id(player.team)])
                temp_player.setInfo(player, self.current_year, season, self.teams[get_id(player.team)])
                temp_player.update(stats, 1, 1)
                self.players[player.playerid] = temp_player
        #Process defense player
        elif ('LB' in player.guess_position or \
                'DE' in player.guess_position or \
                'CB' in player.guess_position or \
                'FS' in player.guess_position or \
                'SS' in player.guess_position or \
                'DT' in player.guess_position or \
                'NT' in player.guess_position or \
                'DEF' in player.guess_position or \
                'SAF' in player.guess_position or \
                'DB' in player.guess_position):
            #Get unique ID of the players team
            team = get_id(player.team)
            #If the player's team exists update the stats, otherwise create new defense
            if team in defense_week:
                defense_week[team].build_week(stats)
                defense_week[team].setInfo(player.team)
            else:
                temp_def = Defense(self.teams[get_id(player.team)])
                temp_def.setInfo(player.team)
                temp_def.build_week(stats)
                defense_week[team] = temp_def

    def build_next_week(self):
        """
        Method to build the next weeks projections and salaries.
        """
        print '\nBuilding next week'
        #Create a new entry in the game list
        self.game_list[self.current_year][self.current_week] = {}
        #Get information from nflgame's schedule
        for id_info in self.schedule[self.current_year][self.current_week]:
            season = self.current_year
            week = self.current_week
            schedule_game = self.schedule[season][week][id_info]
            game = Game(schedule_game['home'], schedule_game['away'], self.ranks, self.schedule, self.teams, season, week)
            away_team = get_id(schedule_game['home'])
            home_team = get_id(schedule_game['away'])
            self.game_list[season][week][home_team] = game
            self.game_list[season][week][away_team] = game
            self.teams[home_team].add_game(schedule_game['home'], schedule_game['away'], week, season)
            self.teams[away_team].add_game(schedule_game['home'], schedule_game['away'], week, season)
        print 'Downloading projections from rotogrinders'
        #Download projections
        projections = self.download_projections()
        #Update player information
        for player in self.players:
            last_name = self.players[player].name.split(".")[1]
            key = last_name + "-" + self.players[player].team.abbr
            key_fd = key + "-" + "fanduel"
            key_dk = key + "-" + "draftkings"
            if key_fd in projections:
                fd_info = projections[key_fd]
            else:
                fd_info = (0, 0, 0, 0)
            if key_dk in projections:
                dk_info = projections[key_dk]
            else:
                dk_info = (0, 0, 0, 0)
            series = {}
            series['week'] = int(self.current_week)
            series['season'] = int(self.current_year)
            series['dk_salary'] = float(dk_info[0])
            series['projection_pro_dk'] = round(float(dk_info[1]), 2)
            series['projection_pro_dk_floor'] = round(float(dk_info[2]), 2)
            series['projection_pro_dk_ceiling'] = round(float(dk_info[3]), 2)
            series['fd_salary'] = float(fd_info[0])
            series['projection_pro_fd'] = round(float(fd_info[1]), 2)
            series['projection_pro_fd_floor'] = round(float(fd_info[2]), 2)
            series['projection_pro_fd_ceiling'] = round(float(fd_info[3]), 2)
            self.players[player].update(pd.DataFrame([series]), 0, 1)
        #Update defense information
        for defense in self.defense:
            key = self.defense[defense].team.abbr
            key_fd = key + "-" + "fanduel"
            key_dk = key + "-" + "draftkings"
            if key_fd in projections:
                fd_info = projections[key_fd]
            else:
                fd_info = (0, 0, 0, 0)
            if key_dk in projections:
                dk_info = projections[key_dk]
            else:
                dk_info = (0, 0, 0, 0)
            series = {}
            series['week'] = int(self.current_week)
            series['season'] = int(self.current_year)
            series['dk_salary'] = float(dk_info[0])
            series['projection_pro_dk'] = round(float(dk_info[1]), 2)
            series['projection_pro_dk_floor'] = round(float(dk_info[2]), 2)
            series['projection_pro_dk_ceiling'] = round(float(dk_info[3]), 2)
            series['fd_salary'] = float(fd_info[0])
            series['projection_pro_fd'] = round(float(fd_info[1]), 2)
            series['projection_pro_fd_floor'] = round(float(fd_info[2]), 2)
            series['projection_pro_fd_ceiling'] = round(float(fd_info[3]), 2)
            self.defense[defense].update(pd.DataFrame([series]), 0, 1)
        #Download injury status
        print 'Downloading injury status from pro-football-reference'
        injury_status = self.download_injury_status()
        #Update player information with injury status
        for player in self.players:
            last_name = self.players[player].name.split(".")[1]
            key = last_name + "-" + self.players[player].team.abbr
            if key in injury_status:
                injury_stat = injury_status[key]['status']
                injury_det = injury_status[key]['details']
            else:
                injury_stat = 0
                injury_det = 0
            series = {}
            series['week'] = self.current_week
            series['season'] = self.current_year
            series['status'] = injury_stat
            series['details'] = injury_det
            self.players[player].update(pd.DataFrame([series]), 0, 1)

    def download_projections(self):
        """
        Method to download all projections from rotogrinders.
        """
        projections = {}
        #The sites and positions to use in rotogrinders url
        sites = ('fanduel', 'draftkings')
        positions = ('qb', 'rb', 'wr', 'te', 'defense')
        all_proj = (sites, positions)
        for site in sites:
            for pos in positions:
                projections.update(self.download_projections_site(site, pos))
        return projections

    def download_projections_site(self, site, pos):
        """
        Method to download an individual site and position projection csv.
        """
        projections = {}
        #Set URL
        url = "https://rotogrinders.com/projected-stats/nfl-" + str(pos) + ".csv?site=" + str(site)
        #Use requests to download csv
        with requests.Session() as session:
            download = session.get(url)
            temp_list = download.content.decode('utf-8')
            #Process csv
            csv_reader = csv.reader(temp_list.splitlines(), delimiter=',')
            proj_list = list(csv_reader)
            #Process defense information
            if 'defense' in pos:
                for row in proj_list:
                    team = nflgame.standard_team(row[2])
                    if team is None:
                        team = self.PROJ_TEAM_CONV[row[2]]
                    key = str(team) + "-" + str(site)
                    salary = row[1]
                    proj = row[7]
                    floor = row[6]
                    ceiling = row[5]
                    projections[key] = (salary, proj, floor, ceiling)
            #Process offense information
            else:
                for row in proj_list:
                    name = row[0].split(" ")[1]
                    team = nflgame.standard_team(row[2])
                    if team is None:
                        team = self.PROJ_TEAM_CONV[row[2]]
                    key = str(name) + "-" + str(team) + "-" + str(site)
                    salary = row[1]
                    proj = row[7]
                    floor = row[6]
                    ceiling = row[5]
                    projections[key] = (salary, proj, floor, ceiling)
        return projections

    def download_injury_status(self):
        """
        This method downloads injury status from pro football reference using pandas.
        """
        url = 'https://www.pro-football-reference.com/players/injuries.htm'
        html = requests.get(url).content
        df_list = pd.read_html(html)
        df = df_list[-1]
        injury_dict = {}
        team_list = []
        for index, row in df.iterrows():
            name = row["Player"]
            name = name.split(" ", 1)[1]
            team = nflgame.standard_team(row["Tm"])
            key = name + "-" + team
            injury_dict[key] = {'status':row["Class"], 'details':row["Details"]}
        print("Injury status download complete")

        return injury_dict

                    
                
    def build_season(self, season, week = 1):
        """
        This method builds information for the season passed as a parameter.
        """
        self.ranks = Ranks()
        if not season in self.game_list:
            self.game_list[season] = {}
        if season < self.current_year:
            for i in range(1, 18):
                self.build_week(i, season)
        elif season == self.current_year:
            for i in range(week, self.current_week):
                self.build_week(i, season)

    def build_data(self, season, week = 1):
        """
        This method updates data from the time the program last ran to the current week and year.
        """
        if season == self.current_year and not week == 1:
            num_weeks = self.current_week - week + 1
        else:
            num_weeks = 0
        while season <= self.current_year:
            self.build_season(season, week)
            season += 1
        self.build_next_week()
        self.clean_list()
        self.build_player_data(num_weeks)
        self.build_defense_data(num_weeks)
        self.save_list()
        self.last_run = self.current_week, self.current_year

    def build_player_data(self, num_weeks):
        """
        This method adds the game and team data into each players dataframe.
        """
        print 'Building player data'
        #Total number of players and current count
        total = len(self.players)
        count = 1
        #Process each player
        for player in self.players:
            print_line = "\r[" + str(count) + "/" + str(total) + "] Building data for " + self.players[player].name
            sys.stdout.write('\x1b[2K')
            sys.stdout.write(print_line)
            sys.stdout.flush()
            team = self.players[player].team.id
            #If we are updating the current season and the number of weeks since last update is more than 1
            if num_weeks > 1:
                #Week for the start of the loop
                start_week = self.current_week - num_weeks
                #Update to the current week
                while start_week <= self.current_week:
                    try:
                        game_data = self.game_list[self.current_year][start_week][team].data
                        team_data = self.game_list[self.current_year][start_week][team].data_team[team]
                        if start_week == self.current_week:
                            self.players[player].update(game_data, 0, 1)
                            self.players[player].update(team_data, 0, 1)
                        else:
                            self.players[player].update(game_data)
                            self.players[player].update(team_data)
                    except KeyError:
                        pass
                    start_week += 1
            #If were building data for a previous season
            else:
                for season in self.game_list:
                    for week in self.game_list[season]:
                        try:
                            game_data = self.game_list[season][week][team].data
                            team_data = self.game_list[season][week][team].data_team[team]
                            if week == self.current_week and season == self.current_year:
                                self.players[player].update(game_data, 0, 1)
                                self.players[player].update(team_data, 0, 1)
                            else:
                                self.players[player].update(game_data)
                                self.players[player].update(team_data)
                        except KeyError:
                            pass
            #Only get data for the number of weeks we are updating
            if num_weeks > 1:
                team_data = self.teams[team].data.tail(num_weeks)
            else:
                team_data = self.teams[team].data
            self.players[player].update(team_data)
            count += 1

    def build_defense_data(self, num_weeks):
        """
        This method update the defense dataframes with information for each game and team. Works the same as build_player_data.
        """
        print '\nBuilding defense data'
        total = len(self.defense)
        count = 1
        #Process each team
        for team in self.defense:
            print_line = "\r[" + str(count) + "/" + str(total) + "] Building data for " + self.defense[team].team.name
            sys.stdout.write('\x1b[2K')
            sys.stdout.write(print_line)
            sys.stdout.flush()
            team = self.defense[team].team.id
            if num_weeks > 1:
                start_week = self.current_week - num_weeks
                while start_week <= self.current_week:
                    try:
                        game_data = self.game_list[self.current_year][start_week][team].data
                        team_data = self.game_list[self.current_year][start_week][team].data_team[team]
                        if start_week == self.current_week:
                            self.defense[team].update(game_data, 0, 0)
                            self.defense[team].update(team_data, 0, 0)
                        else:
                            self.defense[team].update(game_data, 0, 0)
                            self.defense[team].update(team_data, 0, 0)
                    except KeyError:
                        pass
                    start_week += 1
            else:
                for season in self.game_list:
                    for week in self.game_list[season]:
                        try:
                            if week == self.current_week and season == self.current_year:
                                self.defense[team].update(self.game_list[season][week][team].data, 0, 0)
                                self.defense[team].update(self.game_list[season][week][team].data_team[team], 0, 0)
                            else:
                                self.defense[team].update(self.game_list[season][week][team].data, 0, 0)
                                self.defense[team].update(self.game_list[season][week][team].data_team[team], 0, 0)
                        except KeyError:
                            pass
            if num_weeks > 1:
                team_data = self.teams[team].data.tail(num_weeks)
            else:
                team_data = self.teams[team].data
            self.defense[team].update(team_data)
            count += 1

    def clean_list(self):
        """
        This method removes any players from the list that do not have stats for the current year.
        """
        self.players = {key: value for key, value in self.players.items() if value.has_current_stats}

    def update(self):
        """
        This method updates or builds the player list.
        """
        season = self.last_run[1]
        if self.last_run[0] > 1:
            week = int(self.last_run[0]) - 1
        else:
            week = int(self.last_run[0])
        self.build_data(season, int(week))
        
    def save_list(self):
        """
        This method saves the player and defense information into csv files.
        """
        print('\n')
        print('Saving data')
        for player in self.players:
            self.players[player].write_csv()
        for team in self.defense:
            self.defense[team].write_csv()
