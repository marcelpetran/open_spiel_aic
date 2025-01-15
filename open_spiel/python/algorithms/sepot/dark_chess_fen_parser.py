import jax
import jax.numpy as jnp

WHITE_PLAYER = 1
BLACK_PLAYER = 0

class Game:
  def __init__(self, board_size=8, fen=None):
    self.board_size = board_size
    self.fen = fen

    # 0 for black, 1 for white
    self.current_player = 1
    self.terminal = False

    self.fullmove_number = 0
    self.halfmove_clock = 0

    self.en_passant_targets = []

    self.repetitions = 0
    self.board_repetitions = {}

    self.white_can_castle_kingside = False
    self.white_can_castle_queenside = False
    self.black_can_castle_kingside = False
    self.black_can_castle_queenside = False

    self.w_queens = 0
    self.w_bishops = 0
    self.w_knights = 0
    self.w_rooks = 0
    self.w_pawns = 0

    self.b_queens = 0
    self.b_bishops = 0
    self.b_knights = 0
    self.b_rooks = 0
    self.b_pawns = 0

    if fen is not None:
      self.board = self.parse_fen(fen, init=True)
    else:
      self.board = [[" " for _ in range(board_size)]
                    for _ in range(board_size)]

    self.observations = [[[" " for _ in range(board_size)] for _ in range(board_size)],
                         [[" " for _ in range(board_size)] for _ in range(board_size)]]
    self.last_seen = [[[-1 for _ in range(board_size * board_size)] for _ in range(6)],
                      [[-1 for _ in range(board_size * board_size)] for _ in range(6)]]
    self.max_memory = 40

    self.state_tensor_shape = (board_size, board_size, 14)
    self.observation_tensor_shape = (board_size, board_size, 16)
    self.state_tensor = jnp.zeros(self.state_tensor_shape)
    self.observation_tensor = jnp.zeros(self.observation_tensor_shape)

  def add_piece(self, type):
    match type:
      case "P":
        self.w_pawns += 1
      case "p":
        self.b_pawns += 1
      case "N":
        self.w_knights += 1
      case "n":
        self.b_knights += 1
      case "B":
        self.w_bishops += 1
      case "b":
        self.b_bishops += 1
      case "R":
        self.w_rooks += 1
      case "r":
        self.b_rooks += 1
      case "Q":
        self.w_queens += 1
      case "q":
        self.b_queens += 1
      case _:
        pass

  def parse_fen(self, fen, player=None, init=False):
    board = [[" " for _ in range(self.board_size)]
             for _ in range(self.board_size)]
    fen_parts = fen.split(" ")
    board_fen = fen_parts[0]
    board_fen = list(reversed(board_fen.split("/")))
    for i, row in enumerate(board_fen):
      j = 0
      for char in row:
        if char.isdigit():
          j += int(char)
        else:
          board[i][j] = char
          if init:
            self.add_piece(char)
          j += 1
    self.current_player = WHITE_PLAYER if fen_parts[1] == "w" else BLACK_PLAYER
    if player is None:
      self.white_can_castle_kingside = "K" in fen_parts[2]
      self.white_can_castle_queenside = "Q" in fen_parts[2]
      self.black_can_castle_kingside = "k" in fen_parts[2]
      self.black_can_castle_queenside = "q" in fen_parts[2]
    elif player == WHITE_PLAYER:
      self.white_can_castle_kingside = "K" in fen_parts[2]
      self.white_can_castle_queenside = "Q" in fen_parts[2]
    else:
      self.black_can_castle_kingside = "k" in fen_parts[2]
      self.black_can_castle_queenside = "q" in fen_parts[2]
    if fen_parts[3] != "-":
      self.en_passant_targets.append(
          (int(fen_parts[3][1]), ord(fen_parts[3][0]) - 97))
    self.halfmove_clock = int(fen_parts[4])
    self.fullmove_number = int(fen_parts[5])
    return board

  def get_board_hash(self):
    return "".join(["".join(row) for row in self.board])

  def update_state(self, global_fen, white_fen, black_fen, init=False):
    self.board = self.parse_fen(global_fen, init)

    board_str = self.get_board_hash()

    self.board_repetitions[board_str] = self.board_repetitions.get(
      board_str, 0) + 1
    self.repetitions = self.board_repetitions[board_str] - 1

    self.observations[WHITE_PLAYER] = self.parse_fen(white_fen, player=WHITE_PLAYER)
    self.observations[BLACK_PLAYER] = self.parse_fen(black_fen, player=BLACK_PLAYER)

    self.increase_last_seen()
    self.update_last_seen(WHITE_PLAYER)
    self.update_last_seen(BLACK_PLAYER)

  def match_piece(self, piece):
    match piece:
      case "K":
        return 0
      case "Q":
        return 1
      case "R":
        return 2
      case "B":
        return 3
      case "N":
        return 4
      case "P":
        return 5
      case "k":
        return 6
      case "q":
        return 7
      case "r":
        return 8
      case "b":
        return 9
      case "n":
        return 10
      case "p":
        return 11
      case _:
        return -1

  def update_last_seen(self, player):
    for i in range(self.board_size):
      for j in range(self.board_size):
        piece = self.observations[player][i][j]
        if player == WHITE_PLAYER:
          piece_index = self.match_piece(piece) - 6
        else:
          piece_index = self.match_piece(piece)
          
        if piece_index >= 0 and piece_index < 6:
          self.last_seen[player][piece_index][i *
                                              self.board_size + j] = 0

  def increase_last_seen(self):
    for p in range(2):
      for i in range(6):
        for j in range(self.board_size * self.board_size):
          if self.last_seen[p][i][j] != -1:
            self.last_seen[p][i][j] += 1
          elif self.last_seen[p][i][j] >= self.max_memory:
            self.last_seen[p][i][j] = -1

  def get_state_tensor(self):
    piece_map = {
        "K": 0, "Q": 1, "R": 2, "B": 3, "N": 4, "P": 5,
        "k": 6, "q": 7, "r": 8, "b": 9, "n": 10, "p": 11,
        " ": 12
    }

    piece_indices = jnp.array([[piece_map[self.board[i][j]]
                                if self.board[i][j] in piece_map else 12
                                for j in range(self.board_size)]
                               for i in range(self.board_size)])
    num_channels = len(piece_map)
    state_tensor = jax.nn.one_hot(piece_indices, num_classes=num_channels)

    state_tensor = state_tensor.reshape(
        self.board_size, self.board_size, num_channels)

    layer_14 = jnp.zeros((self.board_size, self.board_size), dtype=jnp.float32)
    # acting player
    layer_14 = layer_14.at[0, 0].set(
      1) if self.current_player == WHITE_PLAYER else layer_14.at[0, 1].set(1)
    curr_row = 1
    if self.board_size > 5:
      curr_row += 1
    # repetitions
    for col in range(3):
      layer_14 = layer_14.at[curr_row, col].set(
        1) if self.repetitions == col else layer_14.at[curr_row, col].set(0)

    # black castling kingside
    layer_14 = layer_14.at[self.board_size - 2, 1].set(
      1) if self.black_can_castle_kingside else layer_14.at[self.board_size - 2, 0].set(1)
    # black castling queenside
    layer_14 = layer_14.at[self.board_size - 2, self.board_size - 1].set(
      1) if self.black_can_castle_queenside else layer_14.at[self.board_size - 2, self.board_size - 2].set(1)

    # white castling kingside
    layer_14 = layer_14.at[self.board_size - 1, 1].set(
      1) if self.white_can_castle_kingside else layer_14.at[self.board_size - 1, 0].set(1)
    # white castling queenside
    layer_14 = layer_14.at[self.board_size - 1, self.board_size - 1].set(
      1) if self.white_can_castle_queenside else layer_14.at[self.board_size - 1, self.board_size - 2].set(1)

    return jnp.concatenate([state_tensor, layer_14.reshape(self.board_size, self.board_size, 1)], axis=-1)

  def get_observation_tensor(self, player):
    opp_queens = 0
    opp_bishops = 0
    opp_knights = 0
    opp_rooks = 0
    opp_pawns = 0

    def write_remaining_opponent_pieces(matrix: jnp.array,
                                        val: int, max: int,
                                        curr_row: int, curr_col: int):
      for alive_piece in range(min(val, max)):
        matrix = matrix.at[curr_row, curr_col].set(1)
        curr_col += 1
        if curr_col == self.board_size:
          curr_col = 0
          curr_row += 2
      # to dead indexes
      curr_row += 1

      for dead_piece in range(max - val):
        matrix = matrix.at[curr_row, curr_col].set(1)
        curr_col += 1
        if curr_col == self.board_size:
          curr_col = 0
          curr_row += 2
      # back to alive indexes
      curr_row -= 1

      return matrix, curr_row, curr_col

    layer_15 = jnp.zeros((self.board_size, self.board_size), dtype=jnp.float32)

    if player == WHITE_PLAYER:
      piece_map = {
          "K": 0, "Q": 1, "R": 2, "B": 3, "N": 4, "P": 5,
          "k": 6, "q": 7, "r": 8, "b": 9, "n": 10, "p": 11,
          " ": 12, "?": 13
      }

      init_queens = self.b_queens
      init_bishops = self.b_bishops
      init_knights = self.b_knights
      init_rooks = self.b_rooks
      init_pawns = self.b_pawns

      for i in range(self.board_size):
        for j in range(self.board_size):
          match self.board[i][j]:
            case "q":
              opp_queens += 1
            case "b":
              opp_bishops += 1
            case "n":
              opp_knights += 1
            case "r":
              opp_rooks += 1
            case "p":
              opp_pawns += 1

      king_castle = self.white_can_castle_kingside
      queen_castle = self.white_can_castle_queenside
    else:
      piece_map = {
          "k": 0, "q": 1, "r": 2, "b": 3, "n": 4, "p": 5,
          "K": 6, "Q": 7, "R": 8, "B": 9, "N": 10, "P": 11,
          " ": 12, "?": 13
      }

      init_queens = self.w_queens
      init_bishops = self.w_bishops
      init_knights = self.w_knights
      init_rooks = self.w_rooks
      init_pawns = self.w_pawns

      for i in range(self.board_size):
        for j in range(self.board_size):
          match self.board[i][j]:
            case "Q":
              opp_queens += 1
            case "B":
              opp_bishops += 1
            case "N":
              opp_knights += 1
            case "R":
              opp_rooks += 1
            case "P":
              opp_pawns += 1

      king_castle = self.black_can_castle_kingside
      queen_castle = self.black_can_castle_queenside

    my_pieces = jnp.zeros(
        (self.board_size, self.board_size, 6), dtype=jnp.float32)
    for i in range(self.board_size):
      for j in range(self.board_size):
        piece_type = self.observations[player][i][j]
        if piece_map[piece_type] >= 0 and piece_map[piece_type] < 6:
          piece_idex = piece_map[piece_type]
        else:
          continue
        my_pieces = my_pieces.at[i, j, piece_idex].set(1)

    last_seen = jnp.zeros(
        (self.board_size, self.board_size, 6), dtype=jnp.float32)
    for i in range(6):
      for j in range(self.board_size * self.board_size):
        clamped_last_seen = self.max_memory if self.last_seen[player][i][j] < 0 else min(
            self.last_seen[player][i][j], self.max_memory)
        out = (self.max_memory - clamped_last_seen) / self.max_memory

        last_seen = last_seen.at[j // self.board_size, j %
                                self.board_size, i].set(out)

    empty_unknown = jnp.zeros(
        (self.board_size, self.board_size, 2), dtype=jnp.float32)
    for i in range(self.board_size):
      for j in range(self.board_size):
        piece_type = self.observations[player][i][j]
        if piece_map[piece_type] == 12:
          empty_unknown = empty_unknown.at[i, j, 0].set(1)
        elif piece_map[piece_type] == 13:
          empty_unknown = empty_unknown.at[i, j, 1].set(1)

    state_tensor = jnp.concatenate(
        [my_pieces, last_seen, empty_unknown], axis=-1)

    num_channels = len(piece_map)  # 14

    state_tensor = state_tensor.reshape(
        self.board_size, self.board_size, num_channels)

    # create layer 14 - remaining opponent pieces
    # write queens
    layer_15, curr_row, curr_col = write_remaining_opponent_pieces(
      layer_15, opp_queens, init_queens, 0, 0)
    
    # write rooks
    layer_15, curr_row, curr_col = write_remaining_opponent_pieces(
      layer_15, opp_rooks, init_rooks, curr_row, curr_col)

    # write bishops
    layer_15, curr_row, curr_col = write_remaining_opponent_pieces(
      layer_15, opp_bishops, init_bishops, curr_row, curr_col)

    # write knights
    layer_15, curr_row, curr_col = write_remaining_opponent_pieces(
      layer_15, opp_knights, init_knights, curr_row, curr_col)

    # write pawns
    layer_15, curr_row, curr_col = write_remaining_opponent_pieces(
      layer_15, opp_pawns, init_pawns, curr_row, curr_col)

    # additional info
    layer_16 = jnp.zeros((self.board_size, self.board_size), dtype=jnp.float32)
    # acting player
    layer_16 = layer_16.at[0, 0].set(
      1) if player == WHITE_PLAYER else layer_16.at[0, 1].set(1)
    curr_row = 1
    if self.board_size > 4:
      curr_row += 1
    # repetitions
    for col in range(3):
      layer_16 = layer_16.at[curr_row, col].set(
        1) if self.repetitions == col else layer_16.at[curr_row, col].set(0)

    # castling kingside
    layer_16 = layer_16.at[self.board_size - 1, 1].set(
      1) if king_castle else layer_16.at[self.board_size - 1, 0].set(1)
    # castling queenside
    layer_16 = layer_16.at[self.board_size - 1, self.board_size - 1].set(
      1) if queen_castle else layer_16.at[self.board_size - 1, self.board_size - 2].set(1)

    return jnp.concatenate([state_tensor, layer_15.reshape(self.board_size, self.board_size, 1), layer_16.reshape(self.board_size, self.board_size, 1)], axis=-1)
