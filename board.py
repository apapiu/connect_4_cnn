import numpy as np
from scipy.signal import convolve2d


class Board:
    # convolutional kernels to detect a win:
    detection_kernels = [
        np.array([[1, 1, 1, 1]]),  # horizontal_kernel
        np.array([[1], [1], [1], [1]]),  # vertical_kernel
        np.eye(4, dtype=np.uint8),  # diag1_kernel
        np.fliplr(np.eye(4, dtype=np.uint8)),  # diag2_kernel
    ]

    """A board"""

    def __init__(self):
        self.s = np.zeros([6, 7])
        # self.player?

    def get_possible_moves(self) -> np.ndarray:
        moves = 6 - (self.s == 0).sum(0)
        return np.nonzero(moves < 6)[0]

    def make_move_inplace(self, i: int, player: int) -> None:
        lev = 6 - np.count_nonzero(self.s[:, i] == 0)
        self.s[lev, i] = player

    @staticmethod  # note: doing this on GPU makes it slower
    def check_winning_move(s: np.ndarray, player: int) -> bool:
        for kernel in Board.detection_kernels:
            if (convolve2d(s == player, kernel, mode="valid") == 4).any():
                return True
        return False

    def winning_move(self, player: int) -> bool:
        return self.check_winning_move(self.s, player)
