from board import Board
from players import Player
import numpy as np


class Game:
    def __init__(
        self,
        player_1: Player,
        player_2: Player,
        board: Board | None = None,
        plot_rez: bool = False,
    ):
        self.player_1 = player_1
        self.player_2 = player_2
        self.plot_rez = plot_rez
        self.board = (
            board if board is not None else Board()
        )  # start with empty board is no board given
        self.game_states = []
        self.move_history = []  # Track column moves (e.g., [3, 1, 4, 5])
        self.player = 1
        self.winner = None
        self.move_num = 0
        self.states = []
        self.winners = []

    def simulate(self):
        while self.winner is None:
            if self.player == 1:
                move = self.player_1.make_move(self.board, self.player)
            else:
                move = self.player_2.make_move(self.board, self.player)

            # Track the move (column number)
            if move is not None:
                self.move_history.append(int(move))

            self.game_states.append(self.board.s.copy())

            if self.board.winning_move(self.player):
                self.winner = self.player
                break

            self.move_num += 1
            if self.move_num == 42:
                self.winner = 0

            self.player *= -1

        self.append_game_results()

        # X and y for training
        return self.states, self.winners, self.winner, self.move_history

    def append_game_results(self):
        game_states_np = np.array(self.game_states).astype("int8")

        # who won from current position:
        turns = np.empty((len(self.game_states),))
        turns[::2] = 1
        turns[1::2] = -1
        turns = turns * self.winner if self.winner is not None else turns
        self.states = game_states_np
        self.winners = turns

    def get_padded_move_history(self) -> list[int]:
        """
        Returns move history padded to 42 moves (6*7 board size).
        Pads with:
        - 10 if player 1 wins
        - 11 if player 2 wins
        - 12 if draw
        """
        # Determine padding token based on winner
        pad_token = 10 if self.winner == 1 else (11 if self.winner == -1 else 12)

        # Fast padding using list multiplication
        padding_needed = 42 - len(self.move_history)
        return self.move_history + [pad_token] * padding_needed
