"""
team.py
CSC 492 - 01 Senior Design
Author: Leo Stevens

This class holds the information for each team. The data is passed in from player_list and held in a data frame.
The class holds a running total for win/loss/tie information which is reset each season.
The class also has a method to return a unique ID for each team to deal with teams moving locations.

"""
import nflgame
import pandas as pd
from ranks import Ranks

class Team:
    """
    This class holds information for each team.
    """
    #Dictionary of team names
    TEAMS = {
        'ARI':['Arizona', 'Cardinals', 'Arizona Cardinals'],
        'ATL':['Atlanta', 'Falcons', 'Atlanta Falcons'],
        'BAL':['Baltimore', 'Ravens', 'Baltimore Ravens'],
        'BUF':['Buffalo', 'Bills', 'Buffalo Bills'],
        'CAR':['Carolina', 'Panthers', 'Carolina Panthers'],
        'CHI':['Chicago', 'Bears', 'Chicago Bears'],
        'CIN':['Cincinnati', 'Bengals', 'Cincinnati Bengals'],
        'CLE':['Cleveland', 'Browns', 'Cleveland Browns'],
        'DAL':['Dallas', 'Cowboys', 'Dallas Cowboys'],
        'DEN':['Denver', 'Broncos', 'Denver Broncos'],
        'DET':['Detroit', 'Lions', 'Detroit Lions'],
        'GB':['Green Bay', 'Packers', 'Green Bay Packers', 'G.B.', 'GNB'],
        'HOU':['Houston', 'Texans', 'Houston Texans'],
        'IND':['Indianapolis', 'Colts', 'Indianapolis Colts'],
        'JAC':['Jacksonville', 'Jaguars', 'Jacksonville Jaguars', 'JAX'],
        'JAX':['Jacksonville', 'Jaguars', 'Jacksonville Jaguars', 'JAX'],
        'KC':['Kansas City', 'Chiefs', 'Kansas City Chiefs', 'K.C.', 'KAN'],
        'LA':['Los Angeles', 'Rams', 'Los Angeles Rams', 'L.A.'],
        'MIA':['Miami', 'Dolphins', 'Miami Dolphins'],
        'MIN':['Minnesota', 'Vikings', 'Minnesota Vikings'],
        'NE':['New England', 'Patriots', 'New England Patriots', 'N.E.', 'NWE'],
        'NO':['New Orleans', 'Saints', 'New Orleans Saints', 'N.O.', 'NOR'],
        'NYG':['Giants', 'Giants', 'N.Y.G.'],
        'NYJ':['Jets', 'Jets', 'N.Y.J.'],
        'OAK':['Oakland', 'Raiders', 'Oakland Raiders'],
        'PHI':['Philadelphia', 'Eagles', 'Philadelphia Eagles'],
        'PIT':['Pittsburgh', 'Steelers', 'Pittsburgh Steelers'],
        'SD':['San Diego', 'Chargers', 'San Diego Chargers', 'S.D.', 'SDG'],
        'LAC':['San Diego', 'Chargers', 'San Diego Chargers', 'S.D.', 'SDG'],
        'SEA':['Seattle', 'Seahawks', 'Seattle Seahawks'],
        'SF':['San Francisco', '49ers', 'San Francisco 49ers', 'S.F.', 'SFO'],
        'STL':['St. Louis', 'Rams', 'St. Louis Rams', 'S.T.L.'],
        'TB':['Tampa Bay', 'Buccaneers', 'Tampa Bay Buccaneers', 'T.B.', 'TAM'],
        'TEN':['Tennessee', 'Titans', 'Tennessee Titans'],
        'WAS':['Washington', 'Redskins', 'Washington Redskins', 'WSH'],
    }
    #Dictionary of unique IDs
    TEAMS_ID = {'ARI':0, 'ATL':1, 'BAL':2, 'BUF':3, 'CAR':4, 'CHI':5,
         'CIN':6, 'CLE':7, 'DAL':8, 'DEN':9, 'DET':10, 'GB':11, 'HOU' :12,
         'IND' :13, 'JAC' :14, 'JAX' :14, 'KC' : 15, 'MIA' :16, 'MIN' :17,
         'NE' :18, 'NO' :19, 'NYG' :20, 'NYJ' :21, 'OAK' :22, 'PHI':23,
         'PIT' :24, 'SEA' :25, 'SF' :26, 'TB' :27, 'TEN' :28, 'WAS' : 29,
         'LAC' : 30, 'SD' : 30, 'LA' :31, 'STL' :31}

    #Dictionary of team information
    TEAM_INFO = {
        'ARI': ('NFC', 'West', 'Retractable', 'Grass'),
        'ATL': ('NFC', 'South', 'Retractable', 'Turf'),
        'BAL': ('AFC', 'North', 'Open', 'Grass'),
        'BUF': ('AFC', 'East', 'Open', 'Turf'),
        'CAR': ('NFC', 'South', 'Open', 'Grass'),
        'CHI': ('NFC', 'North', 'Open', 'Grass'),
        'CIN': ('AFC', 'North', 'Open', 'Turf'),
        'CLE': ('AFC', 'North', 'Open', 'Grass'),
        'DAL': ('NFC', 'East', 'Retractable', 'Turf'),
        'DEN': ('AFC', 'West', 'Open', 'Grass'),
        'DET': ('NFC', 'North', 'Fixed', 'Turf'),
        'GB': ('NFC', 'North', 'Open', 'Hybrid'),
        'HOU' : ('AFC', 'South', 'Retractable', 'Turf'),
        'IND' : ('AFC', 'South', 'Retractable', 'Turf'),
        'JAC' : ('AFC', 'South', 'Open', 'Grass'),
        'JAX' : ('AFC', 'South', 'Open', 'Grass'),
        'KC' : ('AFC', 'West', 'Open', 'Grass'),
        'MIA' : ('AFC', 'East', 'Open', 'Grass'),
        'MIN' : ('NFC', 'North', 'Fixed', 'Turf'),
        'NE' : ('AFC', 'East', 'Open', 'Turf'),
        'NO' : ('NFC', 'South', 'Fixed', 'Turf'),
        'NYG' : ('NFC', 'East', 'Open', 'Turf'),
        'NYJ' : ('AFC', 'East', 'Open', 'Turf'),
        'OAK' : ('AFC', 'West', 'Open', 'Grass'),
        'PHI': ('NFC', 'East', 'Open', 'Hybrid'),
        'PIT' : ('AFC', 'North', 'Open', 'Grass'),
        'SEA' : ('NFC', 'West', 'Open', 'Turf'),
        'SF' : ('NFC', 'West', 'Open', 'Grass'),
        'TB' : ('NFC', 'South', 'Open', 'Grass'),
        'TEN' : ('AFC', 'South', 'Open', 'Grass'),
        'WAS' : ('NFC', 'East', 'Open', 'Grass'),
        'LAC' : ('AFC', 'West', 'Open', 'Grass'),
        'SD' : ('AFC', 'West', 'Open', 'Grass'),
        'LA' : ('NFC', 'West', 'Open', 'Grass'),
        'STL' : ('NFC', 'West', 'Open', 'Grass'),
    }
    #Columns for team data frame
    TEAM_HEADER = \
            "week, season, opp, home, wins, losses, ties, win_ratio, streak"

    def __init__(self, team):
        """
        Constructor
        Sets initial information
        """
        self.abbr = nflgame.standard_team(team)
        self.id = Team.TEAMS_ID[self.abbr]
        self.name = Team.TEAMS[self.abbr][1]
        self.conference = Team.TEAM_INFO[self.abbr][0]
        self.division = Team.TEAM_INFO[self.abbr][1]
        self.conf_div = self.conference + " " + self.division
        self.current_season = 0
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.streak = 0
        self.win_ratio = 1.0
        self.home = False
        self.opp = ''
        self.stadium = Team.TEAM_INFO[self.abbr][2]
        self.surface = Team.TEAM_INFO[self.abbr][3]

        self.data = pd.DataFrame(columns=Team.TEAM_HEADER.replace(' ', '').split(','))

    def update_info(self, team):
        """
        This method updates the teams information
        """
        self.abbr = nflgame.standard_team(team)
        self.name = Team.TEAMS[self.abbr][1]

    def add_game(self, home, away, week, season, game = None):
        """
        Add a game to the teams information
        """
        rank_info = pd.Series()
        rank_info['season'] = season
        #If we are starting a new season, reset information
        if not season == self.current_season:
            self.current_season = season
            self.wins = 0
            self.losses = 0
            self.ties = 0
            self.streak = 0
            self.win_ratio = 1.0
        rank_info['week'] = week
        if self.current_season != season:
            self.wins = 0
            self.losses = 0
            self.ties = 0
            self.streak = 0
            self.win_ratio = 1.0
            self.current_season = season
        if home in self.abbr:
            self.home = 'True'
            self.opp = away
        else:
            self.home = 'False'
            self.opp = home
        rank_info['wins'] = self.wins
        rank_info['losses'] = self.losses
        rank_info['ties'] = self.ties
        #rank_info['win_ratio'] = self.win_ratio
        rank_info['opp'] = self.opp
        rank_info['home'] = self.home
        rank_info['streak'] = self.streak
        self.data = self.data.append(rank_info, ignore_index=True)
        #Update win/loss/tie/streak information
        if not game is None:
            if '/' in game.winner:
                self.ties += 1
                self.streak = 0
            elif self.abbr in nflgame.standard_team(game.winner):
                self.wins += 1
                if self.streak >= 0:
                    self.streak += 1
                else:
                    self.streak = 1
            elif self.abbr in nflgame.standard_team(game.loser):
                self.losses += 1
                if self.streak >= 0:
                    self.streak = -1
                else:
                    self.streak -= 1
            games = self.wins + self.losses + self.ties
            if games != 0:
                self.win_ratio = round(float(self.wins) / games, 2)
            else:
                self.win_ratio = 1.0
            #print(nflgame.standard_team(game.winner), nflgame.standard_team(game.loser), self.abbr, self.win_ratio)
def get_id(team):
    """
    Method to get id from team name
    """
    return Team.TEAMS_ID[nflgame.standard_team(team)]

