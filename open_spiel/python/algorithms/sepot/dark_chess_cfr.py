import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import best_response

# small fen example: "r1kr/pppp/PPPP/R1KR w - - 0 1"
# standard fen example: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

def main():
  # start the game within TabularPolicy
  game_params = {
    "board_size": 4,
    "fen": "r1kr/pppp/PPPP/R1KR w - - 0 1",
    
  }
  game = pyspiel.load_game("dark_chess", game_params)
  state = game.new_initial_state()
  # tab_policy = policy.TabularPolicy(game)
  while not state.is_terminal():
    print(str(state))
    player = state.current_player()
    legal_actions = state.legal_actions()
    action = legal_actions[0]
    state.apply_action(action)
    # ot = state.state_tensor()
    ot = state.observation_tensor(player)
    print(f"Observation tensor:\n{len(ot)}\n")
  

  print("Game over")
  print(f"Final state:\n{str(state)}")


if __name__ == "__main__":
  main()