# GameSetMatch

An algorithm that uses logistic regression to predict the winner of professional tennis matches on the ATP tour. The `.ipynb` and `.py` files contain mostly the same code, but the `.ipynb` files contain explanations of decisions I made as well as my general thought process.


## Changes In Progress
- Enhancments to the algorithm using deep learning
- Rethinking some of the imputation decisions I made to account for missing values
- Repurposing the algorithm for placing bets on tennis matches
- Creating a deployed version for public use


## Using the Model to Make Predictions

To see the algorithm's prediction for a match, run `python predict.py player_1 player_2 start_year [end_year]` from the root directory of the repo.

This command will print the predicted winner of the match between `player_1` and `player_2` (both strings) using match data from `start_year` to `end_year` as training data. By default, `end_year` is 2018. At the time of implementation of the algorithm, no 2019 data was available.

**Example**: `python predict.py 'Roger Federer' 'Rafael Nadal' 2017` will output the predicted winner of a match between Roger Federer and Rafael Nadal using their matches from 2017 and 2018 as trainin data.
