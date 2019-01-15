from predict import *
import sys

def check_winner(training_data, row, start, end):
	player1 = row['winner_name']
	player2 = row['loser_name']
	pred = predict_outcome(training_data, player1, player2, start, end)
	if pred == row['winner_name']:
		return 1
	return 0


def test(player, start, end, predict_year):
	training_data = get_training_data(start, end=end)
	predict_data = get_training_data(predict_year, end=predict_year)
	player_pred_data = get_matches_for_player(predict_data, player)
	num_correct = player_pred_data.apply(lambda row: check_winner(training_data, row, start, end), axis=1).sum()
	total_matches = player_pred_data.shape[0]
	accuracy = num_correct/total_matches
	print(f"Trained on {start} to {end} data, the model is {accuracy*100}% correct on {player}'s {predict_year} matches.")


if __name__ == "__main__":
    player = sys.argv[1]
    start_year = int(sys.argv[2])
    end_year = 2017
    predict_year = 2018
    if len(sys.argv) > 3:
    	end_year = int(sys.argv[3])
    	if len(sys.argv) > 4:
    		predict_year = int(sys.argv[4])
    test(player, start_year, end_year, predict_year)