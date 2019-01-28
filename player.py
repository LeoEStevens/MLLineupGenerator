"""
player.py
CSC 492 - 01 Senior Design
Author: Leo Stevens

This class holds all of the player information in a data frame. The data is passed in from player_list.
The data for the team and game come from the team and game classes and the stats come from nflgame.
"""
import pandas as pd
import nflgame
import math

class Player:
    """
    This class holds offensive player information.
    """

    #These are all of the data fields that are tracked for offensive players
    OFFENSE_HEADER = \
        "name,position,team,gsis_id,season,week,years_played,status,details,dk_salary,fd_salary,\
        projection_pro_dk,projection_pro_dk_floor,projection_pro_dk_ceiling,\
        projection_pro_fd,projection_pro_fd_floor,projection_pro_fd_ceiling,dk_score,fd_score,\
        opp,home,wins,losses,ties,streak,win_ratio,passing_att,passing_cmp,passing_yds,passing_tds,passing_twoptm,passing_sk,rushing_att,rushing_yds,rushing_tds,\
        rushing_twoptm,receiving_tar,receiving_rec,receiving_yds,receiving_tds,receiving_twoptm,kickret_tds,fumbles_lost,\
        puntret_tds,fumbles_rec_tds,penalty,penalty_yds,passing_int,opp_win_ratio,opp_streak,\
        def_pass_rank,def_rush_rank,\
        def_pts_allowed,opp_conf,opp_div,opp_div_conf,stadium,surface,game_time,game_day,time,\
        conditions,temp,wind,humidity,visibility,barometric_pressure,dew_point,\
        randforest_pro_dk,randforest_pro_dk_floor,randforest_pro_dk_ceiling,\
        randforest_pro_fd,randforest_pro_fd_floor,randforest_pro_fd_ceiling,\
        ridge_pro_dk,ridge_pro_dk_floor,ridge_pro_dk_ceiling,\
        ridge_pro_fd,ridge_pro_fd_floor,ridge_pro_fd_ceiling,\
        lasso_pro_fd,lasso_pro_dk_floor,lasso_pro_dk_ceiling,\
        lasso_pro_dk,lasso_pro_fd_floor,lasso_pro_fd_ceiling,\
        svr_pro_dk,svr_pro_dk_floor,svr_pro_dk_ceiling,\
        svr_pro_fd,svr_pro_fd_floor,svr_pro_fd_ceiling,\
        nn_pro_dk,nn_pro_dk_floor,nn_pro_dk_ceiling,\
        nn_pro_fd,nn_pro_fd_floor,nn_pro_fd_ceiling,\
        gb_pro_dk,gb_pro_dk_floor,gb_pro_dk_ceiling,\
        gb_pro_fd,gb_pro_fd_floor,gb_pro_fd_ceiling,\
        knn_pro_dk,knn_pro_dk_floor,knn_pro_dk_ceiling,\
        knn_pro_fd,knn_pro_fd_floor,knn_pro_fd_ceiling,\
        enet_pro_dk,enet_pro_dk_floor,enet_pro_dk_ceiling,\
        enet_pro_fd,enet_pro_fd_floor,enet_pro_fd_ceiling"


    #These are the columns we dont want to update
    UPDATE_IGNORE_COLUMNS = ('randforest_pro_dk','randforest_pro_fd','ridge_pro_dk',
            'ridge_pro_fd','lasso_pro_fd','lasso_pro_dk','svr_pro_dk','svr_pro_fd',
            'nn_pro_dk','nn_pro_fd','gb_pro_dk','gb_pro_fd','knn_pro_dk','knn_pro_fd',
            'enet_pro_dk','enet_pro_fd')

    def __init__(self, player, team):
        """
        Constructor
        Only used first time player is created
        """
        self.name = player.name
        self.id = player.playerid
        self.position = player.guess_position
        self.team = team
        self.current_year, self.current_week = nflgame.live.current_year_and_week()
        self.has_current_stats = 0
        self.df = pd.DataFrame(columns=Player.OFFENSE_HEADER.replace(' ', '').split(','))
        self.years_played = 1

    def update(self, updatedDataframe, score = 0, write = 0):
        """
        Method to update the players data frame with another data frame.
        """
        #Iterate over data frame
        for index, row in updatedDataframe.iterrows():
            #Get position of row in players data frame
            index_pos = self.df[(self.df['season'] == row['season']) & (self.df['week'] == row['week'])].index.tolist()
            #If season and week are not in players data frame
            if not index_pos and write:
                #Set initial values
                row['years_played'] = self.years_played
                row['name'] = self.name
                row['position'] = self.position
                row['gsis_id'] = self.id
                row['team'] = self.team.abbr
                #Add row
                self.df.loc[len(self.df)] = row
            #If season and week are in players data frame
            elif index_pos:
                #Convert row to series
                series = pd.Series(row).to_frame().T
                #For each value in updated data frame, update players data frame
                for value in series:
                    if series[value].item() is not None \
                            and value not in Player.UPDATE_IGNORE_COLUMNS \
                            and not series[value].item() == 0 \
                            and series[value].item() == series[value].item() \
                            and value in self.df.columns.values:
                        try:
                            self.df.iloc[index_pos[0], self.df.columns.get_loc(value)] = series[value].item()
                        except ValueError as e:
                            print(e, series[value])
                            pass
        #If we want to score the week
        if score:
            last_season = self.df.iloc[-1, self.df.columns.get_loc('season')]
            if last_season == self.current_year:
                self.has_current_stats = 1
            else:
                self.has_current_stats = 0 
            fd_score, dk_score = score_offense_data(self.df.tail(1))
            self.df.iloc[-1, self.df.columns.get_loc('fd_score')] = fd_score[0]
            self.df.iloc[-1, self.df.columns.get_loc('dk_score')] = dk_score[0]

    def setInfo(self, p, year, season, team):
        """
        Method to update the players personal information
        """
        self.name = p.name
        self.id = p.playerid
        self.position = p.guess_position
        if p.player is None:
            self.years_played = 1
        else:
            self.years_played = p.player.years_pro - (year - season) + 1
        self.team = team
    def write_csv(self):
        """
        Method to write players data to csv file.
        """
        filename = "data/csv/" + self.name + "-" + self.team.abbr + ".csv"
        self.df.to_csv(filename, index=False)

def score_offense_data(week):
    """
    Method to score the player.
    """
    fd_score = 0
    fd_floor = 0
    fd_ceiling = 0
    
    dk_score = 0
    dk_floor = 0
    dk_ceiling = 0

    week = week.fillna(0)

    fd_score += week.iloc[0]['passing_tds'] * 4
    fd_floor += math.floor(week.iloc[0]['passing_tds']) * 4
    fd_ceiling += math.ceil(week.iloc[0]['passing_tds']) * 4

    dk_score += week.iloc[0]['passing_tds'] * 4
    dk_floor += math.floor(week.iloc[0]['passing_tds']) * 4
    dk_ceiling += math.ceil(week.iloc[0]['passing_tds']) * 4

    fd_score += week.iloc[0]['passing_yds'] * 0.04
    fd_floor += math.floor(week.iloc[0]['passing_yds']) * 0.04
    fd_ceiling += math.ceil(week.iloc[0]['passing_yds']) * 0.04
    
    dk_score += week.iloc[0]['passing_yds'] * 0.04
    dk_floor += math.floor(week.iloc[0]['passing_yds']) * 0.04
    dk_ceiling += math.ceil(week.iloc[0]['passing_yds']) * 0.04
    
    if(week.iloc[0]['passing_yds'] >= 300):
        dk_score += 3
    if(math.floor(week.iloc[0]['passing_yds']) >= 300):
        dk_floor += 3
    if(math.ceil(week.iloc[0]['passing_yds']) >= 300):
        dk_ceiling += 3
    
    fd_score -= week.iloc[0]['passing_int']
    fd_floor -= math.ceil(week.iloc[0]['passing_int'])
    fd_ceiling -= math.floor(week.iloc[0]['passing_int'])
    
    dk_score -= week.iloc[0]['passing_int']
    dk_floor -= math.ceil(week.iloc[0]['passing_int'])
    dk_ceiling -= math.floor(week.iloc[0]['passing_int'])
    
    fd_score += week.iloc[0]['rushing_yds'] * 0.1
    fd_floor += math.floor(week.iloc[0]['rushing_yds']) * 0.1
    fd_ceiling += math.ceil(week.iloc[0]['rushing_yds']) * 0.1
    
    dk_score += week.iloc[0]['rushing_yds'] * 0.1
    dk_floor += math.floor(week.iloc[0]['rushing_yds']) * 0.1
    dk_ceiling += math.ceil(week.iloc[0]['rushing_yds']) * 0.1

    if(week.iloc[0]['rushing_yds'] >= 100):
        dk_score += 3
    if(math.floor(week.iloc[0]['rushing_yds']) >= 100):
        dk_floor += 3
    if(math.ceil(week.iloc[0]['rushing_yds']) >= 100):
        dk_ceiling += 3

    fd_score += week.iloc[0]['rushing_tds'] * 6
    fd_floor += math.floor(week.iloc[0]['rushing_tds']) * 6
    fd_ceiling += math.ceil(week.iloc[0]['rushing_tds']) * 6

    dk_score += week.iloc[0]['rushing_tds'] * 6
    dk_floor += math.floor(week.iloc[0]['rushing_tds']) * 6
    dk_ceiling += math.ceil(week.iloc[0]['rushing_tds']) * 6
    
    fd_score += week.iloc[0]['receiving_yds'] * 0.1
    fd_floor += math.floor(week.iloc[0]['receiving_yds']) * 0.1
    fd_ceiling += math.ceil(week.iloc[0]['receiving_yds']) * 0.1

    dk_score += week.iloc[0]['receiving_yds'] * 0.1
    dk_floor += math.floor(week.iloc[0]['receiving_yds']) * 0.1
    dk_ceiling += math.ceil(week.iloc[0]['receiving_yds']) * 0.1
    
    if(week.iloc[0]['receiving_yds'] >= 100):
        dk_score += 3
    if(math.floor(week.iloc[0]['receiving_yds']) >= 100):
        dk_floor += 3
    if(math.ceil(week.iloc[0]['receiving_yds']) >= 100):
        dk_ceiling += 3

    fd_score += week.iloc[0]['receiving_tds'] * 6
    fd_floor += math.floor(week.iloc[0]['receiving_tds']) * 6
    fd_ceiling += math.ceil(week.iloc[0]['receiving_tds']) * 6

    dk_score += week.iloc[0]['receiving_tds'] * 6
    dk_floor += math.floor(week.iloc[0]['receiving_tds']) * 6
    dk_ceiling += math.ceil(week.iloc[0]['receiving_tds']) * 6

    fd_score += week.iloc[0]['receiving_rec'] * 0.5
    fd_floor += math.floor(week.iloc[0]['receiving_rec']) * 0.5
    fd_ceiling += math.ceil(week.iloc[0]['receiving_rec']) * 0.5

    dk_score += week.iloc[0]['receiving_rec']
    dk_floor += math.floor(week.iloc[0]['receiving_rec'])
    dk_ceiling += math.ceil(week.iloc[0]['receiving_rec'])

    fd_score += week.iloc[0]['kickret_tds'] * 6
    fd_floor += math.floor(week.iloc[0]['kickret_tds']) * 6
    fd_ceiling += math.ceil(week.iloc[0]['kickret_tds']) * 6

    dk_score += week.iloc[0]['kickret_tds'] * 6
    dk_floor += math.floor(week.iloc[0]['kickret_tds']) * 6
    dk_ceiling += math.ceil(week.iloc[0]['kickret_tds']) * 6

    fd_score += week.iloc[0]['puntret_tds'] * 6
    fd_floor += math.floor(week.iloc[0]['puntret_tds']) * 6
    fd_ceiling += math.ceil(week.iloc[0]['puntret_tds']) * 6

    dk_score += week.iloc[0]['puntret_tds'] * 6
    dk_floor += math.floor(week.iloc[0]['puntret_tds']) * 6
    dk_ceiling += math.ceil(week.iloc[0]['puntret_tds']) * 6

    fd_score -= week.iloc[0]['fumbles_lost'] * 2
    fd_floor -= math.ceil(week.iloc[0]['fumbles_lost']) * 2
    fd_ceiling -= math.floor(week.iloc[0]['fumbles_lost']) * 2

    dk_score -= week.iloc[0]['fumbles_lost']
    dk_floor -= math.ceil(week.iloc[0]['fumbles_lost'])
    dk_ceiling -= math.floor(week.iloc[0]['fumbles_lost'])

    fd_score += week.iloc[0]['receiving_twoptm'] * 2
    fd_floor += math.floor(week.iloc[0]['receiving_twoptm']) * 2
    fd_ceiling += math.ceil(week.iloc[0]['receiving_twoptm']) * 2
    
    dk_score += week.iloc[0]['receiving_twoptm'] * 2
    dk_floor += math.floor(week.iloc[0]['receiving_twoptm']) * 2
    dk_ceiling += math.ceil(week.iloc[0]['receiving_twoptm']) * 2

    fd_score += week.iloc[0]['rushing_twoptm'] * 2
    fd_floor += math.floor(week.iloc[0]['rushing_twoptm']) * 2
    fd_ceiling += math.ceil(week.iloc[0]['rushing_twoptm']) * 2

    dk_score += week.iloc[0]['rushing_twoptm'] * 2
    dk_floor += math.floor(week.iloc[0]['rushing_twoptm']) * 2
    dk_ceiling += math.ceil(week.iloc[0]['rushing_twoptm']) * 2

    fd_score += week.iloc[0]['passing_twoptm'] * 2
    fd_floor += math.floor(week.iloc[0]['passing_twoptm']) * 2
    fd_ceiling += math.ceil(week.iloc[0]['passing_twoptm']) * 2

    dk_score += week.iloc[0]['passing_twoptm'] * 2
    dk_floor += math.floor(week.iloc[0]['passing_twoptm']) * 2
    dk_ceiling += math.ceil(week.iloc[0]['passing_twoptm']) * 2

    dk_score += week.iloc[0]['fumbles_rec_tds'] * 6
    dk_floor += math.floor(week.iloc[0]['fumbles_rec_tds']) * 6
    dk_ceiling += math.ceil(week.iloc[0]['fumbles_rec_tds']) * 6
    return (round(fd_score, 2), round(fd_floor, 2), round(fd_ceiling, 2)), (round(dk_score, 2), round(dk_floor, 2), round(dk_ceiling, 2))

