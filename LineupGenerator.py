"""
LineupGenerator.py
CSC 492 - 01 Senior Design
Author: Leo Stevens

This is the driver class for the LineupGenerator.
The program is started using the command: python2.7 LineupGenerator.py
When the program is run it checks if a data file exists, if it does then the data is loaded, if not then a new player_list is created.
If the data is corrupted the user is notified and the program exits.
"""
import os
import pickle
import sys
from player_list import player_list
from create_projections import build_projections
from create_lineup import create_lineups

DATA_FILE = 'data/data.pkl'

def save(plist):
    """
    Method to save the player list
    """
    with open(DATA_FILE, 'wb') as output_file:
        pickle.dump(plist, output_file, pickle.HIGHEST_PROTOCOL)
        output_file.close()

def main():
    """
    This method displays a basic menu for the lineup generator.
    It uses a simple while loop for error checking currently.
    It handles user input and calls the appropriate functions based on that input.
    """
    print 'Starting Daily NFL Lineup Generator'
    #Check if the data file exists, if it does load it, else create new player_list
    if os.path.isfile(DATA_FILE):
        print 'Loading data...'
        with open(DATA_FILE, 'rb') as file_input:
            try:
                plist = pickle.load(file_input)
                print 'Loading complete'
            except:
                print 'Data is corrupted, please delete data file and try again'
                sys.exit()
    else:
        print 'Data file not found'
        print 'Data must be updated before projections or lineups can be created'
        plist = player_list()
    exit = 0
    while exit == 0:
        print("----Daily Fantasy Football Lineup Generator-----")
        print("1) Update Data")
        print("2) Create Projections")
        print("3) Create Lineups")
        print("Q/q) Exit")
        print("------------------------------------------------")
        choice = raw_input("Input> ")
        if choice == 'Q' or choice == 'q':
            break
            continue
        try:
            choice = int(choice)
        except ValueError:
            print "Invalid Input"
        if choice == 1:
            print("Updating Data")
            plist.update()
            save(plist)
        elif choice == 2:
            print("Creating Projections (This may take a while)")
            build_projections(plist.players, plist.defense)
            save(plist)
        elif choice == 3:
            error = 1
            back = 0
            while error == 1:
                print("----Daily Fantasy Football Lineup Generator-----")
                print("------------------Choose Site-------------------")
                print("1) FanDuel")
                print("2) DraftKings")
                print("Q/q) Back")
                print("------------------------------------------------")
                site = raw_input("Input> ")
                if site == 'Q' or site == 'q':
                    back = 1
                    break
                try:
                    site = int(site)
                except ValueError:
                    print "Invalid Input"
                    continue
                if not site == 1 and not site == 2:
                    print "Invalid Input"
                    continue
                site = int(site) - 1
                error = 0
            if back == 1:
                back = 0
                continue
            error = 1
            while error == 1:
                print("----Daily Fantasy Football Lineup Generator-----")
                print("------------------Choose Type-------------------")
                print("1) Machine Learning")
                print("2) Professional")
                print("Q/q) Back")
                print("------------------------------------------------")
                professional = raw_input("Input> ")
                if professional == 'Q' or professional == 'q':
                    back = 1
                    break
                try:
                    professional = int(professional)
                except ValueError:
                    print "Invalid Input"
                    continue
                if not professional == 1 and not professional == 2:
                    print "Invalid Input"
                    continue
                professional = int(professional) - 1
                error = 0
            if back == 1:
                back = 0
                continue
            error = 1
            while error == 1:
                print("----Daily Fantasy Football Lineup Generator-----")
                print("-----------Choose Optimzation Type--------------")
                print("1) Floor")
                print("2) Projection")
                print("3) Ceiling")
                print("Q/q) Back")
                print("------------------------------------------------")
                game_type = raw_input("Input> ")
                if game_type == 'Q' or game_type == 'q':
                    back = 1
                    break
                try:
                    game_type = int(game_type)
                except ValueError:
                    print "Invalid Input"
                    continue
                if not game_type >= 1 and not game_type <= 3:
                    print "Invalid Input"
                    continue
                game_type = int(game_type) - 1
                error = 0
            if back == 1:
                back = 0
                continue
            error = 1
            while error == 1:
                print("----Daily Fantasy Football Lineup Generator-----")
                print("-----------------Choose Slate-------------------")
                print("1) All")
                print("2) Sunday + Monday")
                print("3) Sunday Early")
                print("4) Sunday Morning")
                print("5) Sunday Afternoon (4:05 EST and 4:25 EST")
                print("6) Sunday Afternoon (4:25 EST")
                print("7) Sunday + Monday Night")
                print("Q/q) Back")
                print("------------------------------------------------")
                slate = raw_input("Input> ")
                if slate == 'Q' or slate == 'q':
                    back = 1
                    break
                try:
                    slate = int(slate)
                except ValueError:
                    print "Invalid Input"
                    continue
                if slate < 1 or slate > 7:
                    print "Invalid Input"
                    continue
                slate = int(slate) - 1
                error = 0
            if back == 1:
                back = 0
                continue
            error = 1
            while error == 1:
                print("----Daily Fantasy Football Lineup Generator-----")
                print("-------------------Stacking---------------------")
                print("1) No Stacking")
                print("2) Stacking")
                print("Q/q) Back")
                print("------------------------------------------------")
                stacking = raw_input("Input> ")
                if stacking == 'Q' or stacking == 'q':
                    back = 1
                    break
                try:
                    stacking = int(stacking)
                except ValueError:
                    print "Invalid Input"
                    continue
                if not stacking == 1 and not stacking == 2:
                    print "Invalid Input"
                    continue
                stacking = int(stacking) - 1
                error = 0
            if back == 1:
                back = 0
                continue
            error = 1
            while error == 1:
                print("----Daily Fantasy Football Lineup Generator-----")
                print("------------------Lineup Number-------------------")
                print("Q/q) Back")
                print("------------------------------------------------")
                lineup_num = raw_input("Enter number of lineups to create> ")
                if lineup_num == 'Q' or lineup_num == 'q':
                    back = 1
                    break
                try:
                    lineup_num = int(lineup_num)
                except ValueError:
                    print "Invalid Input"
                    continue
                if lineup_num <= 0:
                    print "Invalid Input"
                    continue
                error = 0
            if back == 1:
                back = 0
                continue
            if lineup_num > 1:
                error = 1
                while error == 1:
                    print("----Daily Fantasy Football Lineup Generator-----")
                    print("---------------Duplicate Players----------------")
                    print("Q/q) Back")
                    print("------------------------------------------------")
                    duplicate_players = raw_input("Enter the maximum duplicate players between two lineups> ")
                    if duplicate_players == 'Q' or duplicate_players == 'q':
                        back = 1
                        break
                    try:
                        duplicate_players = int(duplicate_players)
                    except ValueError:
                        print "Invalid Input"
                        continue
                    if duplicate_players < 0:
                        print "Invalid Input"
                        continue
                    error = 0
                if back == 1:
                    back = 0
                    continue
                error = 1
                while error == 1:
                    print("----Daily Fantasy Football Lineup Generator-----")
                    print("----------------Max Times Used------------------")
                    print("Q/q) Back")
                    print("------------------------------------------------")
                    max_times_used = raw_input("Enter the maximum times a player can be used> ")
                    if max_times_used == 'Q' or max_times_used == 'q':
                        back = 1
                        break
                    try:
                        max_times_used = int(max_times_used)
                    except ValueError:
                        print "Invalid Input"
                        continue
                    if max_times_used < 0:
                        print "Invalid Input"
                        continue
                    error = 0
                if back == 1:
                    back = 0
                    continue
                error = 0
            else:
                duplicate_players = 1
                max_times_used = 1
            create_lineups(plist.players, plist.defense, site, professional, game_type, slate, stacking, lineup_num, duplicate_players, max_times_used)
        else:
            print("Invalid Input: " + str(choice))

#This starts the main method when this is executed
if __name__ == "__main__":
    main()
