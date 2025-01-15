import random
import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import best_response
import numpy as np

import timeit

from dark_chess_fen_parser import Game as fp

# small fen example: "r1kr/pppp/PPPP/R1KR w - - 0 1"
# standard fen example: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

def print_observation_tensor(observation_tensor, board_size):
  for i in range(len(observation_tensor)):
    if i % board_size**2 == 0:
      print()
      print(f"layer {i//board_size**2}:")
    print(round(observation_tensor[i], 3), end="\t")
  print()

def play_one_game():
  # start the game within TabularPolicy
  board_size = 4
  game_params = {
    "board_size": board_size,
    # "fen": "r1kr/pppp/PPPP/R1KR w - - 0 1",
    # "fen": "nqkb/pppp/PPPP/NQKB w - - 0 1",
    "fen": "r2k/4/4/R2K w - - 0 1",
    # "fen": "rnqkb/ppppp/5/PPPPP/RNQKB w - - 0 1",
    # "fen": "rnqkbr/pppppp/6/6/PPPPPP/RNQKBR w - - 0 1",
    # "fen": "rnbkbnr/ppppppp/7/7/7/PPPPPPP/RNBKBNR w - - 0 1",
    # "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    
  }
  game = pyspiel.load_game("dark_chess", game_params)
  state = game.new_initial_state()

  print(game.observation_tensor_shape())
  print(game.state_tensor_shape())

  fp_game = fp(game_params["board_size"], game_params["fen"])
  fp_game.update_state(global_fen= state.to_string(), 
                       white_fen= state.observation_string(1), 
                       black_fen= state.observation_string(0))
  

  i = 0 # number of mismatches
  while not state.is_terminal():
    player = state.current_player()
   
    # fp_ot = fp_game.get_observation_tensor(player).reshape(board_size**2,16).transpose().flatten() # observation tensor
    fp_ot = fp_game.get_state_tensor().reshape(board_size**2,14).transpose().flatten() # state tensor

    
    # ot = np.array(state.observation_tensor(player)) # observation tensor
    ot = np.array(state.state_tensor()) # state tensor

    is_close = np.isclose(fp_ot, ot)

    # Get the indices of mismatched elements
    mismatch_indices = np.where(~is_close)[0]

    if mismatch_indices.size > 0:
        first_mismatch_index = mismatch_indices[0]
        print(f"First mismatched element index: {first_mismatch_index} of layer {first_mismatch_index//board_size**2}")
        print(f"Values: fp_ot[{first_mismatch_index}] = {fp_ot[first_mismatch_index]}, ot[{first_mismatch_index}] = {ot[first_mismatch_index]}")
        
        print(state.to_string())
        print(state.observation_string(player))
        print(f"FEN observation tensor for player {player}:")
        print_observation_tensor(fp_ot, board_size)
        print(f"STATE observation tensor for player {player}:")
        print_observation_tensor(ot, board_size)
        i+=1
        
    legal_actions = state.legal_actions()
    action = random.choice(legal_actions)
    state.apply_action(action)
    fp_game.update_state(global_fen= state.to_string(), 
                         white_fen= state.observation_string(1), 
                         black_fen= state.observation_string(0))
  return i


if __name__ == "__main__":
  total_games = 100
  erros = 0
  for i in range(total_games):
    print(f"Game {i+1}/{total_games}")
    erros += play_one_game()
  # print(timeit.timeit(play_one_game, number=total_games))
  
  print(f"Total errors: {erros}/{total_games}")