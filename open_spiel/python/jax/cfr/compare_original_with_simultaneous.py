# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This compares the speed/results of Jax CFR to of the original impl of CFR.

The results slightly differ due to different rounding of regrets between
original implmentation and CFR. When setting clamping of regrets to 1e-8 the
results are exactly the same.
"""


# pylint: disable=g-importing-member

import time
import argparse
from open_spiel.python.algorithms.best_response import BestResponsePolicy
from open_spiel.python.algorithms.cfr import CFRPlusSolver
from open_spiel.python.jax.cfr.jax_cfr import JaxCFR
from open_spiel.python.jax.cfr.jax_simultaneous_cfr import SimultaneousJaxCFR
from open_spiel.python.algorithms import exploitability
import pyspiel

parser = argparse.ArgumentParser()

parser.add_argument("--cards", default=5, type=int, help="Number of goofspiel cards")
parser.add_argument("--test_turn_based", default= False, type=bool, help= "Whether to load the game for SM-CFR as turn-based or not")


def compare_cfr_with_jax_cfr(game):
  """Do the comparison."""

  start = time.time()
  jax_cfr = JaxCFR(game)
  print(time.time() - start)
  jax_cfr.multiple_steps(10000)
  print(time.time() - start)

  start = time.time()
  cfr = CFRPlusSolver(game)
  print(time.time() - start)
  for _ in range(1000):
    cfr.evaluate_and_update_policy()

  print(time.time() - start)

  jax_strat = jax_cfr.average_policy()
  jax_br1 = BestResponsePolicy(jax_cfr.game, 1, jax_strat)
  jax_br2 = BestResponsePolicy(jax_cfr.game, 0, jax_strat)

  cfr_strat = cfr.average_policy()
  cfr_br1 = BestResponsePolicy(jax_cfr.game, 1, cfr_strat)
  cfr_br2 = BestResponsePolicy(jax_cfr.game, 0, cfr_strat)

  print("Jax P1: ", jax_br1.value(jax_cfr.game.new_initial_state()))
  print("CFR P1: ", cfr_br1.value(jax_cfr.game.new_initial_state()))
  print("Jax P2: ", jax_br2.value(jax_cfr.game.new_initial_state()))
  print("CFR P2: ", cfr_br2.value(jax_cfr.game.new_initial_state()))

def compare_simultaneous_with_jax(game_name, game_params, load_as_turn_based) :

  game = pyspiel.load_game_as_turn_based(game_name, game_params)

  start = time.time()
  jax_cfr = JaxCFR(game)
  print("CFR init", time.time() - start)
  jax_cfr.multiple_steps(10000)
  print("CFR run", time.time() - start)
  
  if load_as_turn_based:
    print("Loading as turn based")
    game = pyspiel.load_game_as_turn_based(game_name, game_params)
  else:
    print("Loading as simultaneous")
    game = pyspiel.load_game(game_name, game_params)

  start = time.time()
  simultaneous_jax_cfr = SimultaneousJaxCFR(game)
  print("SM-CFR init", time.time() - start)
  simultaneous_jax_cfr.multiple_steps(10000)
  print("SM-CFR run", time.time() - start)

  jax_strat = jax_cfr.average_policy()
  jax_br1 = BestResponsePolicy(jax_cfr.game, 1, jax_strat)
  jax_br2 = BestResponsePolicy(jax_cfr.game, 0, jax_strat)
  
  #print("Finished computing jax")

  simultaneous_jax_strat = simultaneous_jax_cfr.average_policy()
  simultaneous_jax_br1 = BestResponsePolicy(simultaneous_jax_cfr.game, 1, simultaneous_jax_strat)
  simultaneous_jax_br2 = BestResponsePolicy(simultaneous_jax_cfr.game, 0, simultaneous_jax_strat)

  print("Simultaneous Jax P1: ", simultaneous_jax_br1.value(simultaneous_jax_cfr.game.new_initial_state()))
  print("Jax P1: ", jax_br1.value(jax_cfr.game.new_initial_state()))
  print("Simultaneous Jax P2: ", simultaneous_jax_br2.value(simultaneous_jax_cfr.game.new_initial_state()))
  print("Jax P2: ", jax_br2.value(jax_cfr.game.new_initial_state()))
  # exploitability only works on turn based games
  if load_as_turn_based:
    jax_exploitability = exploitability.exploitability(jax_cfr.game, jax_strat)
    simultaneous_jax_exploitability = exploitability.exploitability(simultaneous_jax_cfr.game, simultaneous_jax_strat)
    print("Simultaneous Jax exploitability: ", simultaneous_jax_exploitability)
    print("Jax exploitability: ", jax_exploitability)


if __name__ == "__main__":
  args = parser.parse_args()
  game_params = {"num_cards": args.cards, "imp_info": True, "points_order": "descending"}
  compare_simultaneous_with_jax(game_name="goofspiel", game_params=game_params, load_as_turn_based=args.test_turn_based)
