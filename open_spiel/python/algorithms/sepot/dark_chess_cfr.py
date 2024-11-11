import pyspiel

# small fen example: "r1kr/pppp/PPPP/R1KR w - - 0 1",

def main():
  game_params = {
    "board_size": 4,
    "fen": "r1kr/pppp/PPPP/R1KR w - - 0 1",
    
  }

  game = pyspiel.load_game("dark_chess", game_params)
  print("Game loaded")
  state = game.new_initial_state()
  print("Initial state created")
  while not state.is_terminal():
    print(str(state))
    player = state.current_player()
    legal_actions = state.legal_actions()
    action = legal_actions[0]
    state.apply_action(action)
    o_t = state.observation_tensor(player)
    print(f"Observation string:\n{o_t}\n")
  

  print("Game over")
  print("Final state:\n{}".format(str(state)))


if __name__ == "__main__":
  main()