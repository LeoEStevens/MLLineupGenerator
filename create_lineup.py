"""
create_lineup.py
CSC 492 - 01 Senior Design
Author: Leo Stevens

This class uses pulp to create lineups. The data and constrain information is passed as a parameter.
The other constraints are defined according to draft kings and fan duels rules. 
The lineups are created using pulp and then put into a data frame to be displayed.
PULP needs information in very specific ways (LpAffineExpressions, LpVariables, etc) so those classes are built and passed into the problem.
A LpVariable holds information about the name, index, upper and lower bounds. 
LpAffineExpression is a linear combination of LpVariables. 
"""
import os
import pandas as pd
import math
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpAffineExpression


def create_lineups(players_list, defense_list, site, professional = 0, game_type = 0, slate = 0, stacking = 0, lineup_num = 1, duplicate_players = 0, max_times_used = 0):
    """
    This class creates and displays lineups using PULP.
    """
    #Define caps
    DK_SALARY_CAP = 50000
    FD_SALARY_CAP = 60000
    #Lists for days and times
    DAYS = []
    TIMES = []
    #List for players that have been used (No Duplicates)
    used_players = []
    #List of players in all lineups (Duplicates)
    all_lineups = []

    #Process parameters
    if site == 0:
        site_name = "FanDuel"
    else:
        site_name = "DraftKings"
    if professional == 1:
        lineup_type = "professional"
    else:
        lineup_type = "machine learning"
    if game_type == 0:
        game_name = "floor"
    elif game_type == 1:
        game_name = "projection"
    else:
        game_name = "ceiling"

    print "Creating", lineup_num, game_name, lineup_type, "lineups for", site_name 
    if slate is 0:
        DAYS.append('All')
        TIMES.append('All')
    elif slate is 1:
        DAYS.append('Sun')
        DAYS.append('Mon')
        TIMES.append('All')
    elif slate is 2:
        DAYS.append('Sun')
        TIMES.append('1:00')
        TIMES.append('4:05')
        TIMES.append('4:25')
    elif slate is 3:
        DAYS.append('Sun')
        TIMES.append('9:30')
        TIMES.append('1:00')
    elif slate is 4:
        DAYS.append('Sun')
        TIMES.append('4:05')
        TIMES.append('4:25')
    elif slate is 5:
        DAYS.append('Sun')
        TIMES.append('4:25')
    elif slate is 6:
        DAYS.append('Sun')
        DAYS.append('Mon')
        TIMES.append('8:25')
        TIMES.append('8:30')
        TIMES.append('8:40')
    #Create data frame for the lineups
    players = pd.DataFrame(columns=('name', 'position', 'team', 'projection', 'floor', 'ceiling', 'salary', 'opp', 'day', 'time', 'status'))
    #Set names of columns to use from data frame, random forest is currently the most accurate ML algo
    if site is 0:
        salary = 'fd_salary'
        if professional:
            proj = 'projection_pro_fd'
            floor = 'projection_pro_fd_floor'
            ceiling = 'projection_pro_fd_ceiling'
        else:
            proj = 'randforest_pro_fd'
            floor = 'randforest_pro_fd_floor'
            ceiling = 'randforest_pro_fd_ceiling'
        SALARY_CAP = FD_SALARY_CAP
    else:
        salary = 'dk_salary'
        if professional:
            proj = 'projection_pro_dk'
            floor = 'projection_pro_dk_floor'
            ceiling = 'projection_pro_dk_ceiling'
        else:
            proj = 'randforest_pro_dk'
            floor = 'randforest_pro_dk_floor'
            ceiling = 'randforest_pro_dk_ceiling'
        SALARY_CAP = DK_SALARY_CAP
    #Dictionaries for projections and salaries
    projections = {'QB':{}, 'RB':{}, 'WR':{}, 'TE':{}, 'Flex':{}, 'DST':{}}
    salaries = {'QB':{}, 'RB':{}, 'WR':{}, 'TE':{}, 'Flex':{}, 'DST':{}}
#    unused_players = []
#    unused_teams = []
#    unused_rows = []

    #Dictionary of all players
    all_players = {}
    all_players.update(players_list)
    all_players.update(defense_list)
    #Process all players
    for player in all_players:
        dataFrame = all_players[player].df
        last_row = dataFrame.tail(1)
        #Check if player has all the values we need, if they do then add them to the list, else ignore them
        if math.isnan(last_row.iloc[-1, last_row.columns.get_loc(proj)]) is False \
                and int(last_row.iloc[-1, last_row.columns.get_loc(salary)]) != 0 \
                and int(last_row.iloc[-1, last_row.columns.get_loc(proj)]) != 0 \
                and (last_row.iloc[-1, last_row.columns.get_loc('game_time')] in TIMES or 'All' in TIMES) \
                and (last_row.iloc[-1, last_row.columns.get_loc('game_day')] in DAYS or 'All' in DAYS) \
                and (math.isnan(last_row.iloc[-1, last_row.columns.get_loc('fd_score')]) is True \
                    or int(last_row.iloc[-1, last_row.columns.get_loc('fd_score')]) == 0): 
            #Check if its an instance of the player class, if not then grab info from defense dataframe, else get info from offense dataframe
            if not 'player' in str(all_players[player].__class__):
                pos = 'DST'
                name = last_row.iloc[-1, last_row.columns.get_loc('team')]
                team = name
                status = " "
                opp = last_row.iloc[-1, last_row.columns.get_loc('opp')]
                players = players.append({'name':name, 'position':pos, 'team':team,
                    'projection':last_row.iloc[-1, last_row.columns.get_loc(proj)],
                    'salary':last_row.iloc[-1, last_row.columns.get_loc(salary)],
                    'opp':opp,
                    'day':last_row.iloc[-1, last_row.columns.get_loc('game_day')],
                    'time':last_row.iloc[-1, last_row.columns.get_loc('game_time')],
                    'floor':last_row.iloc[-1, last_row.columns.get_loc(floor)],
                    'ceiling':last_row.iloc[-1, last_row.columns.get_loc(ceiling)],
                    'status':status}, ignore_index=True)

            #Dont use injured players
            elif 'questionable' not in str(last_row.iloc[-1, last_row.columns.get_loc('status')]) \
                    and 'out' not in str(last_row.iloc[-1, last_row.columns.get_loc('status')]) \
                    and 'doubtful' not in  str(last_row.iloc[-1, last_row.columns.get_loc('status')]) \
                    and 'I-R' not in str(last_row.iloc[-1, last_row.columns.get_loc('status')]):
                pos = last_row.iloc[-1, last_row.columns.get_loc('position')]
                name = last_row.iloc[-1, last_row.columns.get_loc('name')]
                team = last_row.iloc[-1, last_row.columns.get_loc('team')]
                status = last_row.iloc[-1, last_row.columns.get_loc('status')]
                details = last_row.iloc[-1, last_row.columns.get_loc('details')]
                if not isinstance(status, str):
                    status = " "
                opp = last_row.iloc[-1, last_row.columns.get_loc('opp')]
                players = players.append({'name':name, 'position':pos, 'team':team,
                    'projection':last_row.iloc[-1, last_row.columns.get_loc(proj)],
                    'salary':last_row.iloc[-1, last_row.columns.get_loc(salary)],
                    'opp':opp,
                    'day':last_row.iloc[-1, last_row.columns.get_loc('game_day')],
                    'time':last_row.iloc[-1, last_row.columns.get_loc('game_time')],
                    'floor':last_row.iloc[-1, last_row.columns.get_loc(floor)],
                    'ceiling':last_row.iloc[-1, last_row.columns.get_loc(ceiling)],
                    'status':status}, ignore_index=True)

    #Copy the dataframe before it gets processed
    original_players = players.copy()
    lineup_count = 0
    #Create user inputted number of lineups
    while lineup_count < lineup_num:
        players = original_players.copy()
        #Set problem
        problem = LpProblem("Daily Fantasy", LpMaximize)
        #Create columns for times used(int) and in_lineup (bool)
        players['times_used'] = [all_lineups.count(var) for var in players['name']]
        players['in_lineup'] = [used_players.count(var) for var in players['name']]
        #Create new dictionary for pulp to use
        _vars =  [LpVariable('x_{0:04d}'.format(index), cat='Binary') for index in players.index]

        #Declare constraints for position, times_used, and in_lineup
        problem += LpAffineExpression(zip(_vars, 1 * (players['position'] == 'QB'))) == 1
        problem += LpAffineExpression(zip(_vars, 1 * (players['position'] == 'RB'))) >= 2
        problem += LpAffineExpression(zip(_vars, 1 * (players['position'] == 'WR'))) >= 3
        problem += LpAffineExpression(zip(_vars, 1 * (players['position'] == 'TE'))) >= 1
        problem += LpAffineExpression(zip(_vars, 1 * (players['position'] == 'RB'))) <= 3
        problem += LpAffineExpression(zip(_vars, 1 * (players['position'] == 'WR'))) <= 4
        problem += LpAffineExpression(zip(_vars, 1 * (players['position'] == 'TE'))) <= 2
        problem += LpAffineExpression(zip(_vars, 1 * (players['position'] == 'DST'))) == 1
        problem += LpAffineExpression(zip(_vars, 1 * (players['position'] != 0))) == 9
        problem += LpAffineExpression(zip(_vars, 1 * (players['times_used'] < max_times_used))) == 9
        problem += LpAffineExpression(zip(_vars, 1 * (players['in_lineup'] == 1))) <= duplicate_players
        #Dictionaries for stacking constraints
        stacking_dict = {}
        using_dict = {}

        #Process team information for stacking and max number of players on a team constraint.
        for team in players['team']:
            problem += LpAffineExpression(zip(_vars, 1 * (players['team'] == team))) <= 4
            stacking_dict[team] = {}
            stacking_dict[team]['QB'] = 0
            stacking_dict[team]['WR'] = 0

        #Process stacking info
        for index, row in players.iterrows():
            if stacking:
                if 'QB' in row['position']:
                    stacking_dict[row['team']]['QB'] += _vars[index]
                elif 'WR' in row['position']:
                    stacking_dict[row['team']]['WR'] += _vars[index]
                for team in stacking_dict:
                    problem += (stacking_dict[team]['QB'] <= stacking_dict[team]['WR'])
        #Declare problem type based on user input
        if game_type == 0:
            problem += LpAffineExpression(zip(_vars, players['floor']))
        elif game_type == 1:
            problem += LpAffineExpression(zip(_vars, players['projection']))
        else:
            problem += LpAffineExpression(zip(_vars, players['ceiling']))
        #Salary constraint
        problem += LpAffineExpression(zip(_vars, players['salary'])) <= SALARY_CAP
        #Solve the problem
        problem.solve()
        #Variables to hold totals
        projection_total = 0
        floor_total = 0
        ceiling_total = 0
        salary_total = 0

        #Get players that are in the lineup and drop unwanted columns
        lineup = [var.varValue for var in problem.variables()]
        players['lineup'] = lineup
        lineup = players[players['lineup'] == 1].drop(['lineup', 'times_used', 'in_lineup'], 1).fillna(" ").sort_values(by=['position']).reset_index(drop=True)

        #If we got a good solution process it, else exit
        if problem.status is 1:
            #Print lineup information and totals
            for index, row in lineup.iterrows():
                all_lineups.append(row['name'])
                if not row['name'] in used_players:
                    used_players.append(row['name'])
            print "Lineup: " + str(lineup_count + 1)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', 15, 'display.width', 500):
                print lineup
           
            projection_total += lineup['projection'].sum()
            floor_total += lineup['floor'].sum()
            ceiling_total += lineup['ceiling'].sum()
            salary_total += lineup['salary'].sum()


            print "Projection:", round(projection_total, 2) 
            print "Floor:", round(floor_total, 2) 
            print "Ceiling:", round(ceiling_total, 2) 
            print "Salary:", salary_total 
            lineup_count += 1
        else:
            if len(used_players) is 0:
                print("No lineups found")
            else:
                print("No additional lineups match the criteria")
            break
