import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import best_response

from fen_parser import Game as fp

# small fen example: "r1kr/pppp/PPPP/R1KR w - - 0 1"
# standard fen example: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

def main():
  # start the game within TabularPolicy
  game_params = {
    "board_size": 4,
    "fen": "r1kr/pppp/PPPP/R1KR w - - 0 1",
    # "fen": "nqkb/pppp/PPPP/NQKB w - - 0 1",
    # "board_size": 8,
    # "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    
  }
  game = pyspiel.load_game("dark_chess", game_params)
  state = game.new_initial_state()
  
  fp_game = fp(game_params["board_size"], game_params["fen"])
  fp_game.update_state(global_fen= state.to_string(), 
                       white_fen= state.observation_string(0), 
                       black_fen= state.observation_string(1))
  
  parsed_state_tensor = fp_game.get_state_tensor()
  print(f"State tensor {parsed_state_tensor.shape}:\n{parsed_state_tensor}\n")
  print(f"flatten state tensor:\n{parsed_state_tensor.flatten()}\n")
  

  parsed_observation_tensor = fp_game.get_observation_tensor(0)
  print(f"Observation tensor {parsed_observation_tensor.shape}:\n{parsed_observation_tensor}\n")
  print(f"flatten observation tensor:\n{parsed_observation_tensor.flatten()}\n")

  # print(state.observation_tensor(1))
  # tab_policy = policy.TabularPolicy(game)
  while not state.is_terminal():
    player = state.current_player()
    print(state.observation_string(1 - player))
    legal_actions = state.legal_actions()
    action = legal_actions[0]
    state.apply_action(action)
    # ot = state.state_tensor()
    ot = state.observation_tensor(player)
    # print(f"Observation tensor:\n{ot}\n")
  

  print("Game over")


if __name__ == "__main__":
  main()