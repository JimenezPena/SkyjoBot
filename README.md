This repository contains a DQN algorithm trained to play the card game Skyjo™ and the code to play human-vs-AI though a web server. 

Try it out in: skyjobot.onrender.com

The code is structured a follows: skyjo_env.py contains the game environment (board, card deck, game rules, etc), skyjo_model contains the current model architecture and skyjo_dqn_model_X.0.pth its actual weights.

The code for starting the web server and the game actions is within online_app.py (app.py was the first local only version). 

The website design is controlled with the script templates/index.html. 

Finally, all the code used for the training of the model is within TrainingFolder.

Previous versions of the DQN weights (often incompatible) are saved in PreviousVersions

The history of versions and changes is the following:

skyjo_dqn_model_4.0.pth (current) :  Increases the number of actions to 26, such that when the card from the deck is discarded, the dqn (and the human player) can choose which of the unrevealed cards to flip, instead of automatically flipping the first unrevealed card.

skyjo_dqn_model_3.0.pth :  Solves a feature leak that allowed to see unrevealead cards. Adds aditional observables, like number of own unrevelaed cards, the lowest number of unrevealed cards of any player, self visible points, estimation of self total points, comparison with other players, etc

skyjo_dqn_model_2.0.pth: First version uploaded to Git. Has 15 actions available: pick from discard pile, pick from the deck, replace those with one of the 12 cards, and if pick from deck, discard and reveal the first unrevealed card.
