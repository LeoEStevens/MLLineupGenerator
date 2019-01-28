"""
defense.py
CSC 492 - 01 Senior Design
Author: Leo Stevens

This class holds all defense information. The data is passed in from player_list and held in data frames.
The file also has the method to score defensive data.
"""
import pandas as pd
import nflgame
import math

class Defense:
    """
    This class holds the information for defense.
    """
    #These are the columns in the data frame
    DEFENSE_HEADER = \
        "team,season,week,dk_salary,fd_salary,projection_pro_dk,projection_pro_dk_floor,projection_pro_dk_ceiling,\
        projection_pro_fd,projection_pro_fd_floor,projection_pro_fd_ceiling,dk_score,fd_score,opp,home,wins,losses,ties,streak,win_ratio,points_allowed,defense_sk,\
        defense_sk_yds,defense_ffum,defense_frec,defense_frec_yds,defense_frec_tds,defense_misc_tds,\
        defense_safe,defense_fgblk,defense_puntblk,defense_xpblk,defense_tds,\
        defense_int,defense_int_tds,defense_int_yds,defense_pass_def,defense_qbhit,defense_tkl,defense_tkl_loss,penalty,penalty_yds,opp_win_ratio,opp_streak,\
        opp_passing_rank,opp_rushing_rank,opp_scoring_rank,opp_conf,opp_div,opp_div_conf,stadium,surface,game_time,game_day,time,\
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


    #These are the columns we do not want to update
    UPDATE_IGNORE_COLUMNS = ('season', 'week', 
            'randforest_pro_dk','randforest_pro_fd','ridge_pro_dk',
            'ridge_pro_fd','lasso_pro_fd','lasso_pro_dk','svr_pro_dk','svr_pro_fd',
            'nn_pro_dk','nn_pro_fd','gb_pro_dk','gb_pro_fd','knn_pro_dk','knn_pro_fd',
            'enet_pro_dk','enet_pro_fd')

    def __init__(self, team):
        """
        Constructor
        """
        self.team = team
        self.df = pd.DataFrame(columns=Defense.DEFENSE_HEADER.replace(' ', '').split(','))

    def setInfo(self, team):
        """
        Update team name
        """
        self.abbr = nflgame.standard_team(team)

    def build_week(self, updatedDataframe):
        """
        This method builds the week for the temporary defenses.
        Update and build_week are seperate so information isnt doubled when updating.
        """
        for index, row in updatedDataframe.iterrows():
            row['team'] = str(self.team.abbr)
            index_pos = self.df[(self.df['season'] == row['season']) & (self.df['week'] == row['week'])].index.tolist()
            if not index_pos:
                self.df.loc[len(self.df)] = row
            else:
                series = pd.Series(row).to_frame().T
                for value in series:
                    if series[value].item() is not None and \
                            value not in Defense.UPDATE_IGNORE_COLUMNS \
                            and not series[value].item() == 0 \
                            and series[value].item() == series[value].item() \
                            and value in self.df.columns.values:
                        try:
                            if self.df.iloc[index_pos[0], self.df.columns.get_loc(value)] == self.df.iloc[index_pos[0], self.df.columns.get_loc(value)]:
                                self.df.iloc[index_pos[0], self.df.columns.get_loc(value)] += series[value].item()
                            else:
                                self.df.iloc[index_pos[0], self.df.columns.get_loc(value)] = series[value].item()
                        except ValueError:
                            print(series[value])
                            pass

    def update(self, updatedDataframe, score = 0, write = 0):
        """
        This method updates the defense data frame.
        """
        for index, row in updatedDataframe.iterrows():
            row['team'] = str(self.team.abbr)
            index_pos = self.df[(self.df['season'] == row['season']) & (self.df['week'] == row['week'])].index.tolist()
            if not index_pos and write:
                self.df.loc[len(self.df)] = row
            elif index_pos:
                series = pd.Series(row).to_frame().T
                for value in series:
                    if series[value].item() is not None and \
                            value not in Defense.UPDATE_IGNORE_COLUMNS \
                            and not series[value].item() == 0 \
                            and series[value].item() == series[value].item() \
                            and value in self.df.columns.values:
                        try:
                            self.df.iloc[index_pos[0], self.df.columns.get_loc(value)] = series[value].item()
                        except ValueError:
                            print(series[value])
                            pass
        if score:
            fd_score, dk_score = score_defense_data(self.df.tail(1))
            self.df.iloc[-1, self.df.columns.get_loc('fd_score')] = fd_score[0]
            self.df.iloc[-1, self.df.columns.get_loc('dk_score')] = dk_score[0]

    def write_csv(self):
        """
        This method writes the defense data into csv files.
        """
        filename = "data/csv/DEF_" + self.team.abbr + ".csv"
        self.df.to_csv(filename, index=False)

def score_defense_data(week):
    """
    This method scores the defensive data
    """
    fd_score = 0
    fd_floor = 0
    fd_ceiling = 0

    dk_score = 0
    dk_floor = 0
    dk_ceiling = 0

    week = week.fillna(0)

    fd_score += week.iloc[0]['defense_sk']
    fd_floor += math.floor(week.iloc[0]['defense_sk'])
    fd_ceiling += math.ceil(week.iloc[0]['defense_sk'])
    
    dk_score += week.iloc[0]['defense_sk']
    dk_floor += math.floor(week.iloc[0]['defense_sk'])
    dk_ceiling += math.ceil(week.iloc[0]['defense_sk'])
    
    fd_score += week.iloc[0]['defense_int'] * 2
    fd_floor += math.floor(week.iloc[0]['defense_int']) * 2
    fd_ceiling += math.ceil(week.iloc[0]['defense_int']) * 2
    
    dk_score += week.iloc[0]['defense_int'] * 2
    dk_floor += math.floor(week.iloc[0]['defense_int']) * 2
    dk_ceiling += math.ceil(week.iloc[0]['defense_int']) * 2
    
    fd_score += week.iloc[0]['defense_frec'] * 2
    fd_floor += math.floor(week.iloc[0]['defense_frec']) * 2
    fd_ceiling += math.ceil(week.iloc[0]['defense_frec']) * 2
    
    dk_score += week.iloc[0]['defense_frec'] * 2
    dk_floor += math.floor(week.iloc[0]['defense_frec']) * 2
    dk_ceiling += math.ceil(week.iloc[0]['defense_frec']) * 2
    
    fd_score += week.iloc[0]['defense_tds'] * 6
    fd_floor += math.floor(week.iloc[0]['defense_tds']) * 6
    fd_ceiling += math.ceil(week.iloc[0]['defense_tds']) * 6
    
    dk_score += week.iloc[0]['defense_tds'] * 6
    dk_floor += math.floor(week.iloc[0]['defense_tds']) * 6
    dk_ceiling += math.ceil(week.iloc[0]['defense_tds']) * 6
    
    fd_score += week.iloc[0]['defense_safe'] * 2
    fd_floor += math.floor(week.iloc[0]['defense_safe']) * 2
    fd_ceiling += math.ceil(week.iloc[0]['defense_safe']) * 2
    
    dk_score += week.iloc[0]['defense_safe'] * 2
    dk_floor += math.floor(week.iloc[0]['defense_safe']) * 2
    dk_ceiling += math.ceil(week.iloc[0]['defense_safe']) * 2
    
    fd_score += week.iloc[0]['defense_fgblk'] * 2
    fd_floor += math.floor(week.iloc[0]['defense_fgblk']) * 2
    fd_ceiling += math.ceil(week.iloc[0]['defense_fgblk']) * 2
    
    dk_score += week.iloc[0]['defense_fgblk'] * 2
    dk_floor += math.floor(week.iloc[0]['defense_fgblk']) * 2
    dk_ceiling += math.ceil(week.iloc[0]['defense_fgblk']) * 2
    
    points_allowed = week.iloc[0]['points_allowed']
    if points_allowed == 0:
        fd_score += 10
        dk_score += 10
    elif points_allowed <= 6:
        fd_score += 7
        dk_score += 7
    elif points_allowed <= 13:
        fd_score += 4
        dk_score += 4
    elif points_allowed <= 20:
        fd_score += 1
        dk_score += 1
    elif points_allowed <= 27:
        fd_score += 0
        dk_score += 0
    elif points_allowed <= 34:
        fd_score -= 1
        dk_score -= 1
    else:
        fd_score -= 4
        dk_score -= 4

    points_allowed = math.floor(week.iloc[0]['points_allowed'])
    if points_allowed == 0:
        fd_floor += 10
        dk_floor += 10
    elif points_allowed <= 6:
        fd_floor += 7
        dk_floor += 7
    elif points_allowed <= 13:
        fd_floor += 4
        dk_floor += 4
    elif points_allowed <= 20:
        fd_floor += 1
        dk_floor += 1
    elif points_allowed <= 27:
        fd_floor += 0
        dk_floor += 0
    elif points_allowed <= 34:
        fd_floor -= 1
        dk_floor -= 1
    else:
        fd_floor -= 4
        dk_floor -= 4

    points_allowed = math.ceil(week.iloc[0]['points_allowed'])
    if points_allowed == 0:
        fd_ceiling += 10
        dk_ceiling += 10
    elif points_allowed <= 6:
        fd_ceiling += 7
        dk_ceiling += 7
    elif points_allowed <= 13:
        fd_ceiling += 4
        dk_ceiling += 4
    elif points_allowed <= 20:
        fd_ceiling += 1
        dk_ceiling += 1
    elif points_allowed <= 27:
        fd_ceiling += 0
        dk_ceiling += 0
    elif points_allowed <= 34:
        fd_ceiling -= 1
        dk_ceiling -= 1
    else:
        fd_ceiling -= 4
        dk_ceiling -= 4

    return (round(fd_score, 2), round(fd_floor, 2), round(fd_ceiling, 2)), (round(dk_score, 2), round(dk_floor, 2), round(dk_ceiling, 2))

