import sys
import pandas as pd 
import numpy as np 
import sklearn.linear_model as lm 
from pathlib import Path 
import re

#####################
### DATA CLEANING ###
#####################
def clean_data(data):
    data.drop(drop_cols, axis=1, inplace=True)
    missing_match_stats = data[data['w_ace'].isnull()]
    data.drop(missing_match_stats.index, axis=0, inplace=True)
    data['tourney_date'] = pd.to_datetime(data['tourney_date'].apply(lambda date: str(date)))
    return data

###########################
### FEATURE ENGINEERING ###
###########################
def convert_match_stats_to_percent(data):
    data['w_ace%'] = data['w_ace'] / data['w_svpt']
    data['l_ace%'] = data['l_ace'] / data['l_svpt']
    
    data['w_df%'] = data['w_df'] / data['w_svpt']
    data['l_df%'] = data['l_df'] / data['l_svpt']
    
    data['w_1stIn%'] = data['w_1stIn'] / data['w_svpt']
    data['l_1stIn%'] = data['l_1stIn'] / data['l_svpt']
    
    data['w_1stWon%'] = data['w_1stWon'] / data['w_1stIn']
    data['l_1stWon%'] = data['l_1stWon'] / data['l_1stIn']
    
    data['w_2ndWon%'] = data['w_2ndWon'] / (data['w_svpt'] - data['w_1stIn'])
    data['l_2ndWon%'] = data['l_2ndWon'] / (data['l_svpt'] - data['l_1stIn'])
    
    data['w_bpSaved%'] = data['w_bpSaved'] / data['w_bpFaced']
    data['l_bpSaved%'] = data['l_bpSaved'] / data['l_bpFaced']
    
    data['w_bpFaced%'] = data['w_bpFaced'] / data['w_svpt']
    data['l_bpFaced%'] = data['l_bpFaced'] / data['l_svpt']
    
    return data


def replace_nan_bp(data):
    data['w_bpSaved%'].replace(to_replace=np.NaN, value=1, inplace=True)
    data['l_bpSaved%'].replace(to_replace=np.NaN, value=1, inplace=True)
    return data


def get_matches_for_player(data, player):
    """Returns a DataFrame containing all matches in DATA in which PLAYER played."""
    return data[(data['winner_name'] == player) | (data['loser_name'] == player)]


def add_win_loss(data, player):
    data['result'] = data['winner_name'].apply(lambda w: 1 if w == player else 0)
    return data


def compute_win_streak(results):
    result_string = results.apply(lambda x: str(x)).str.cat()
    streak = re.findall(r"1+$", result_string)
    if not streak:
        return 0
    else:
        return len(streak[0])


def add_win_streak(data):
    """Adds a column to DATA containing the win streak for every match. Also sorts DATA chronologically."""
    data.sort_values(by=['tourney_date', 'match_num'], inplace=True)
    result = data['result']
    streaks = []
    for i in range(1, len(result)+1):
        so_far = result.iloc[:i]
        streaks.append(compute_win_streak(so_far))
    data['win_streak'] = pd.Series(streaks, index=data.index)
    return data


def compute_head_to_head(data, row, player):
    match_date = row['tourney_date']
    if row['winner_name'] == player:
        other_player = row['loser_name']
    else:
        other_player = row['winner_name']
    head_to_head_matches = data[((data['winner_name'] == other_player) | (data['loser_name'] == other_player))
                               & (data['tourney_date'] <= match_date)]
    total_matches = head_to_head_matches.shape[0]
    player_wins = head_to_head_matches[head_to_head_matches['winner_name'] == player].shape[0]
    return player_wins/total_matches

def add_head_to_head(data):
    data['head_to_head'] = data.apply(lambda row: compute_head_to_head(data, row, "Roger Federer"), axis=1)
    return data


def get_opponent_hand(row, player):
    if row['winner_name'] == player:
        return row['loser_hand']
    else:
        return row['winner_hand']


def add_opponent_hand(data):
    data['opponent_hand'] = data.apply(lambda row: get_opponent_hand(row, "Roger Federer"), axis=1)
    return pd.get_dummies(data, columns=['opponent_hand'])


def add_player_v_opponent_stats(data, player):
    match_stats = ['ace%', 'df%', '1stIn%', '1stWon%', '2ndWon%', 'bpSaved%', 'bpFaced%']
    for stat in match_stats:
        player_stat_list = []
        opp_stat_list = []
        for i in range(data.shape[0]):
            if data['winner_name'].iloc[i] == player:
                player_stat_list.append(data['w_'+stat].iloc[i])
                opp_stat_list.append(data['l_'+stat].iloc[i])
            else:
                player_stat_list.append(data['l_'+stat].iloc[i])
                opp_stat_list.append(data['w_'+stat].iloc[i])
        data['player_'+stat] = pd.Series(player_stat_list, index=data.index)
        data['opp_'+stat] = pd.Series(opp_stat_list, index=data.index)
    return data


##########################
### FEATURE ESTIMATION ###
##########################
def get_stat(row, statistic, player):
    if row['winner_name'] == player:
        return row['w_'+statistic]
    else:
        return row['l_'+statistic]

def get_estimate(data, statistic, player):
    player_data = get_matches_for_player(data, player)
    s = player_data.apply(lambda row: get_stat(row, statistic, player), axis=1)
    return s.mean()


#####################
### MODEL FITTING ###
#####################
features = ['player_ace%', 'opp_ace%', 'player_df%', 'opp_df%', 'player_1stIn%', 'opp_1stIn%',
            'player_1stWon%', 'opp_1stWon%', 'player_2ndWon%', 'opp_2ndWon%', 'player_bpSaved%',
            'opp_bpSaved%', 'player_bpFaced%', 'opp_bpFaced%', 'win_streak',
            'head_to_head', 'opponent_hand_L', 'opponent_hand_R']

def process_data(data):
    """Executes the feature engineering process on DATA by applying the necessary transformations."""
    X = (
        data
            .pipe(convert_match_stats_to_percent)
            .pipe(replace_nan_bp)
            .pipe(add_win_loss, ("Roger Federer"))
            .pipe(add_win_streak)
            .pipe(add_head_to_head)
            .pipe(add_opponent_hand)
            .pipe(add_player_v_opponent_stats, ("Roger Federer"))
    )
    return X


def make_train_matrix(data, player, test=False):
    all_data = clean_data(data)
    player_data = get_matches_for_player(all_data, player)
    if test:
        return (process_data(player_data)[features], process_data(player_data)['result'])
    return process_data(player_data)[features]

##################
### PREDICTION ###
##################
