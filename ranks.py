"""
ranks.py
CSC 492 - 01 Senior Design
Author: Leo Stevens

This class holds rank information for each team for a season.
The information is held in dictionaries and OrderedDict is used to sort the dictionaries.
Each season the rank information is reset.
"""
from collections import OrderedDict
import pandas as pd
import numpy as np
import nflgame

class Ranks:
    """
    This class hold the rankings for each team.
    """
    rushing_o = {}
    passing_o = {}
    scoring_o = {}
      
    rushing_d = {}
    passing_d = {}
    scoring_d = {}

    def add_game(self, game): 
        """
        This method processes a game and updates the rankings.
        """
        #If the team exists in the dictionary update its values, else create new entry 
        if nflgame.standard_team(game.home) in self.rushing_o:
            self.rushing_o[nflgame.standard_team(game.home)] += game.stats_home[3]
            self.passing_o[nflgame.standard_team(game.home)] += game.stats_home[2]
            self.scoring_o[nflgame.standard_team(game.home)] += game.score_home              
               
            self.rushing_d[nflgame.standard_team(game.home)] += game.stats_away[3]
            self.passing_d[nflgame.standard_team(game.home)] += game.stats_away[2]
            self.scoring_d[nflgame.standard_team(game.home)] += game.score_away
        else:
            self.rushing_o[nflgame.standard_team(game.home)] = game.stats_home[3]
            self.passing_o[nflgame.standard_team(game.home)] = game.stats_home[2]
            self.scoring_o[nflgame.standard_team(game.home)] = game.score_home        

            self.rushing_d[nflgame.standard_team(game.home)] = game.stats_away[3]
            self.passing_d[nflgame.standard_team(game.home)] = game.stats_away[2]
            self.scoring_d[nflgame.standard_team(game.home)] = game.score_away

        if nflgame.standard_team(game.away) in self.rushing_o:
            self.rushing_o[nflgame.standard_team(game.away)] += game.stats_away[3]
            self.passing_o[nflgame.standard_team(game.away)] += game.stats_away[2]
            self.scoring_o[nflgame.standard_team(game.away)] += game.score_away              
              
            self.rushing_d[nflgame.standard_team(game.away)] += game.stats_home[3]
            self.passing_d[nflgame.standard_team(game.away)] += game.stats_home[2]
            self.scoring_d[nflgame.standard_team(game.away)] += game.score_home
        else:
            self.rushing_o[nflgame.standard_team(game.away)] = game.stats_away[3]
            self.passing_o[nflgame.standard_team(game.away)] = game.stats_away[2]
            self.scoring_o[nflgame.standard_team(game.away)] = game.score_away
                
            self.rushing_d[nflgame.standard_team(game.away)] = game.stats_home[3]
            self.passing_d[nflgame.standard_team(game.away)] = game.stats_home[2]
            self.scoring_d[nflgame.standard_team(game.away)] = game.score_home
        
        #Wrap dictionaries in OrderedDict to sort them
        self.rushing_o = OrderedDict(sorted(self.rushing_o.items(), 
            key=lambda x: x[1], reverse=True))
        self.passing_o = OrderedDict(sorted(self.passing_o.items(), 
            key=lambda x: x[1], reverse=True))
        self.scoring_o = OrderedDict(sorted(self.scoring_o.items(), 
            key=lambda x: x[1], reverse=True))
        self.rushing_d = OrderedDict(sorted(self.rushing_d.items(), 
            key=lambda x: x[1]))
        self.passing_d = OrderedDict(sorted(self.passing_d.items(), 
            key=lambda x: x[1]))
        self.scoring_d = OrderedDict(sorted(self.scoring_d.items(), 
            key=lambda x: x[1]))
        
    def get_ranks(self, team):
        """
        Method to return the current rankings for a team.
        """
        if team in self.passing_o:
            series = pd.Series(np.array([self.rushing_o.keys().index(team) + 1,
               self.passing_o.keys().index(team) + 1,
               self.scoring_o.keys().index(team) + 1,
               self.rushing_d.keys().index(team) + 1,
               self.passing_d.keys().index(team) + 1,
               self.scoring_d.keys().index(team) + 1]), 
               index=['opp_rushing_rank', 'opp_passing_rank', 'opp_scoring_rank',
                   'def_rush_rank', 'def_pass_rank', 'def_pts_allowed'])
            return series
        else:
            series = pd.Series(np.array([0,0,0,0,0,0]),
               index=['opp_rushing_rank', 'opp_passing_rank', 'opp_scoring_rank',
                   'def_rush_rank', 'def_pass_rank', 'def_pts_allowed'])
            return series

