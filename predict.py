import sys
import pandas as pd 
import numpy as np 
import sklearn.linear_model as lm 
from pathlib import Path 
import re
import warnings
warnings.filterwarnings("ignore")

#####################
### DATA CLEANING ###
#####################
def clean_data(data):
    """Prepares the data for feature engineering by getting rid of useless information, deleting records with missing data,
    and converting certain data into more manageable formats."""
    drop_cols = ['winner_seed', 'loser_seed', 'winner_entry', 'loser_entry', 'winner_ht', 'loser_ht']
    if set(drop_cols).issubset(data.columns):
        data.drop(drop_cols, axis=1, inplace=True)
    missing_match_stats = data[data['w_ace'].isnull()]
    data.drop(missing_match_stats.index, axis=0, inplace=True)
    data['winner_rank'].replace(to_replace=np.NaN, value=data['winner_rank'].max(), inplace=True)
    data['loser_rank'].replace(to_replace=np.NaN, value=data['loser_rank'].max(), inplace=True)
    data['tourney_date'] = pd.to_datetime(data['tourney_date'])
    return data

###########################
### FEATURE ENGINEERING ###
###########################
def convert_match_stats_to_percent(data):
    """Converts the raw counts of match statistics to percents for fair comparison between matches."""
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
    """Replaces undefined break point saved values with 1."""
    data['w_bpSaved%'].replace(to_replace=np.NaN, value=1, inplace=True)
    data['l_bpSaved%'].replace(to_replace=np.NaN, value=1, inplace=True)
    return data


def get_matches_for_player(data, player):
    """Returns a DataFrame containing all matches in DATA in which PLAYER played."""
    return data[(data['winner_name'] == player) | (data['loser_name'] == player)]


def add_win_loss(data, player):
    """Adds a column to DATA indicating whether PLAYER won or lost each match."""
    data['result'] = data['winner_name'].apply(lambda w: 1 if w == player else 0)
    return data


def compute_win_streak(results):
    """Helper function for add_win_streak."""
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
    """Helper function for add_head_to_head."""
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

def add_head_to_head(data, player):
    """Adds a column to DATA containing the head-to-head statistic for every match."""
    data['head_to_head'] = data.apply(lambda row: compute_head_to_head(data, row, player), axis=1)
    return data


def get_opponent_hand(row, player):
    """Helper function for add_opponent_hand."""
    if row['winner_name'] == player:
        return row['loser_hand']
    else:
        return row['winner_hand']


def add_opponent_hand(data, player):
    """Adds a column to DATA that indicates the hand of PLAYER's opponent."""
    data['opponent_hand'] = data.apply(lambda row: get_opponent_hand(row, player), axis=1)
    return pd.get_dummies(data, columns=['opponent_hand'])


def add_player_v_opponent_stats(data, player):
    """Adds columns to DATA containing the match statistics for PLAYER and the opponent."""
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

def add_player_v_opponent_rankings(data, player):
    player_ranks = []
    opp_ranks = []
    for i in range(data.shape[0]):
        if data['winner_name'].iloc[i] == player:
            player_ranks.append(data['winner_rank'].iloc[i])
            opp_ranks.append(data['loser_rank'].iloc[i])
        else:
            player_ranks.append(data['loser_rank'].iloc[i])
            opp_ranks.append(data['winner_rank'].iloc[i])
    data['player_rank'] = pd.Series(player_ranks, index=data.index)
    data['opp_rank'] = pd.Series(opp_ranks, index=data.index)
    return data

##########################
### FEATURE ESTIMATION ###
##########################

def get_estimate(data, statistic, player):
    """Returns the estimate of STATISTIC for PLAYER for a future match."""
    player_data = get_matches_for_player(data, player)
    s = process_data(player_data, player)
    s = add_player_v_opponent_stats(player_data, player)
    return s[statistic].mean()

def head_to_head(data, player1, player2):
    """Returns the head-to-head statistic for PLAYER1 against PLAYER2."""
    player_data = get_matches_for_player(data, player1)
    h2h_matches = player_data[(player_data['winner_name'] == player2) | (player_data['loser_name'] == player2)]
    total_matches = h2h_matches.shape[0]
    player_wins = h2h_matches[h2h_matches['winner_name'] == player1].shape[0]
    if total_matches == 0:
        return 0
    return player_wins/total_matches

def win_streak(data, player):
    """Returns the win-streak of PLAYER."""
    player_data = get_matches_for_player(data, player)
    player_data.sort_values(by=['tourney_date', 'match_num'], inplace=True)
    results = player_data.pipe(add_win_loss, (player))['result']
    #print(results)
    return compute_win_streak(results)

def get_ranking(training_data, rankings, player):
    player_data = get_matches_for_player(training_data, player)
    player_wins = player_data[player_data['winner_name'] == player]
    player_losses = player_data[player_data['loser_name'] == player]
    if player_wins.shape[0] > 0:
        player_id = player_wins['winner_id'].iloc[0]
    elif player_losses.shape[0] > 0:
        player_id = player_losses['loser_id'].iloc[0]
    player_ranks = rankings[rankings['player_id'] == player_id]
    return player_ranks['rank'].mean()

#####################
### MODEL FITTING ###
#####################

features = ['player_ace%', 'player_df%', 'player_1stIn%', 'player_1stWon%', 'player_2ndWon%', 'player_bpSaved%', 'player_bpFaced%',
            'opp_ace%', 'opp_df%', 'opp_1stIn%', 'opp_1stWon%', 'opp_2ndWon%', 'opp_bpSaved%', 'opp_bpFaced%', 'win_streak',
            'head_to_head', 'player_rank', 'opp_rank']#, 'opponent_hand_L', 'opponent_hand_R']  

def process_data(data, player):
    """Executes the feature engineering process on DATA by applying the necessary transformations."""
    X = (
        data
            .pipe(convert_match_stats_to_percent)
            .pipe(replace_nan_bp)
            .pipe(add_win_loss, (player))
            .pipe(add_win_streak)
            .pipe(add_head_to_head, (player))
            .pipe(add_opponent_hand, (player))
            .pipe(add_player_v_opponent_stats, (player))
            .pipe(add_player_v_opponent_rankings, (player))
            .fillna(value=0, axis=1)
    )
    return X


def make_train_matrix(data, player, test=False):
    """Transforms DATA into a matrix for PLAYER's model to be trained with."""
    all_data = clean_data(data)
    player_data = get_matches_for_player(all_data, player)
    if test:
        return (process_data(player_data, player)[features], process_data(player_data, player)['result'])
    return process_data(player_data)[features]

##################
### PREDICTION ###
##################

def get_training_data(start, end=2018, years=None):
    """Puts ATP match data from year START to year END in a DataFrame."""
    #print("Getting training data")
    data = []
    if years:
        for year in years:
            year_string = str(year)
            data.append(pd.read_csv(Path(f'./data/jeff_sackman_data/atp_matches_{year_string}.csv')))
        return pd.concat(data)
    else:
        for year in range(start, end+1):
            year_string = str(year)
            data.append(pd.read_csv(Path(f'./data/jeff_sackman_data/atp_matches_{year_string}.csv')))
        return pd.concat(data)


def compute_model(data, player):
    """Trains a model for PLAYER."""
    #print(f"Computing model for {player}")
    train_matrix, results = make_train_matrix(data, player, True)
    model = lm.LogisticRegression(penalty='l2', C=1.0, fit_intercept=True, multi_class='ovr')
    model.fit(train_matrix, results)
    return model

def decide_winner(player1, player2, p1_win_prob, p2_win_prob):
    """Decides whether PLAYER1 ir PLAYER2 will win given their win probabilities."""
    if p1_win_prob >= .5 and p2_win_prob < 0.5:
        print(f"{player1} will defeat {player2}.")
        return player1
    elif p2_win_prob >= .5 and p1_win_prob < 0.5:
        print(f"{player2} will defeat {player1}.")
        return player2
    elif p1_win_prob == p2_win_prob:
        print("Both players have an equal chance of winning against the other. This program cannot decide a winner.")
        return
    else:
        if max(p1_win_prob, p2_win_prob) == p1_win_prob:
            print(f"{player1} will defeat {player2}.")
            return player1
        else:
            print(f"{player2} will defeat {player1}.")
            return player2

def predict_outcome(training_data, player1, player2, start_year, end_year, rankings):
    """Predicts the outcome of a match between PLAYER1 and PLAYER2"""
    try:
        p1_model = compute_model(training_data, player1)
        player1_estimates = [get_estimate(training_data, stat, player1) for stat in features[:7]]
        player2_estimates = [get_estimate(training_data, stat, player2) for stat in features[:7]]
        player1_rank = get_ranking(training_data, rankings, player1)
        player2_rank = get_ranking(training_data, rankings, player2)
        player1_features = player1_estimates + player2_estimates + [win_streak(training_data, player1)] + [head_to_head(training_data, player1, player2)] + [player1_rank] + [player2_rank]
        print(player1_features)
        p1_win_prob = p1_model.predict_proba(np.reshape(player1_features, (1, -1)))[0][1]
        print(f"{player1}'s model predicts that {player1} has a {p1_win_prob} chance of winning against {player2}.")
        p2_model = compute_model(training_data, player2)
        player2_features = player2_estimates + player1_estimates + [win_streak(training_data, player2)] + [head_to_head(training_data, player2, player1)] + [player2_rank] + [player1_rank]
        p2_win_prob = p2_model.predict_proba(np.reshape(player2_features, (1, -1)))[0][1]
        print(f"{player2}'s model predicts that {player2} has a {p2_win_prob} chance of winning against {player1}.")
        return decide_winner(player1, player2, p1_win_prob, p2_win_prob)
    except ValueError:
        print(f"Insufficient data to determine the winner of {player1} vs {player2}")


if __name__ == "__main__":
    player1 = sys.argv[1]
    player2 = sys.argv[2]
    if len(sys.argv) > 3:
        start_year = int(sys.argv[3])
        end_year = 2018
        if len(sys.argv) > 4:
            end_year = int(sys.argv[4])
    training_data = get_training_data(start_year, end=end_year)
    rankings = pd.read_csv(Path('./data/atp_rankings_current.csv'))
    predict_outcome(training_data, player1, player2, start_year, end_year, rankings)