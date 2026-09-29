This repository contains a DQN algorithm trained to play the card game Skyjo™ and the code to play human-vs-AI though a web server. Try it out in skyjobot.onrender.com
The code is structured a follows: skyjo_env.py contains the game environment (board, card deck, game rules, etc), skyjo_model contains the current model architecture and skyjo_dqn_model_2.0.pth its actual weights.
The code for starting the web server and the game actions is within online_app.py (app.py was the first local only version). 
The website design is controlled with the script templates/index.html. 
Finally, all the code used for the training of the model is within TrainingFolder.
