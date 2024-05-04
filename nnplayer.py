from typing import Protocol, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from scipy.signal import convolve2d
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


######Moves:


@torch.no_grad
def get_nn_preds(board: Board, model: nn.Module, player: int) -> pd.Series:
    possible_moves = board.get_possible_moves()
    moves_np = []
    for i in possible_moves:
        s_new = board.s.copy()
        board.make_move_inplace(i, player)
        moves_np.append(board.s)
        board.s = s_new.copy()  # Revert move

    moves_np = np.array(moves_np).reshape(-1, 1, 6, 7)

    model.eval()
    moves_np = torch.tensor(moves_np).float().to(device)
    preds = model(moves_np).cpu().numpy()[:, 0]

    return pd.Series(preds, possible_moves)


def optimal_nn_move(board: Board, model: nn.Module, player: int) -> None:
    preds = get_nn_preds(board, model, player)
    best_move = preds.idxmax()
    board.make_move_inplace(cast(int, best_move), player)


def optimal_nn_move_noise(
    board: Board, player: int, model: nn.Module, std_noise: int = 0
) -> None:
    preds = get_nn_preds(board, model, player)
    preds = preds + np.random.normal(0, std_noise, len(preds))
    best_move = preds.idxmax()
    board.make_move_inplace(best_move, player)


def random_move(board: Board, player: int, use_2ply_check: bool) -> None:
    possible_moves = board.get_possible_moves()

    if use_2ply_check:
        for i in possible_moves:
            s_new = board.s.copy()
            board.make_move_inplace(i, player)
            if board.winning_move(player):
                return  # state is changed inplace
            board.s = s_new.copy()

        for i in possible_moves:
            s_new = board.s.copy()
            board.make_move_inplace(i, (-1) * player)
            if board.winning_move((-1) * player):
                board.s = s_new.copy()
                board.make_move_inplace(i, player)
                return
            board.s = s_new.copy()

        bad_moves = []
        for i in possible_moves:
            s_new = board.s.copy()
            board.make_move_inplace(i, player)
            pos_mov2 = board.get_possible_moves()

            for j in pos_mov2:
                s_new2 = board.s.copy()
                board.make_move_inplace(j, player * (-1))

                if board.winning_move((-1) * player):
                    bad_moves.append(i)
                board.s = s_new2.copy()

            board.s = s_new.copy()

        non_lose = np.setdiff1d(possible_moves, bad_moves)
        if len(non_lose) > 0:
            possible_moves = non_lose

    rand_move = np.random.choice(possible_moves)
    board.make_move_inplace(rand_move, player)


class Player(Protocol):
    # makes move inplace on board:
    def make_move(self, board: Board, player: int) -> None: ...


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
        self.player = 1
        self.winner = None
        self.move_num = 0
        self.states = []
        self.winners = []

    def simulate(self):
        while self.winner is None:
            if self.player == 1:
                self.player_1.make_move(self.board, self.player)
            else:
                self.player_2.make_move(self.board, self.player)

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
        return self.states, self.winners, self.winner

    def append_game_results(self):
        game_states_np = np.array(self.game_states).astype("int8")

        # who won from current position:
        turns = np.empty((len(self.game_states),))
        turns[::2] = 1
        turns[1::2] = -1
        turns = turns * self.winner if self.winner is not None else turns
        self.states = game_states_np
        self.winners = turns


use_wandb = False


class RandomPlayer:
    def __init__(self, move_function, plus):
        self.move_function = move_function
        self.plus = plus
        self.name = "random_player_2ply" if self.plus else "random_player"

    def make_move(self, board, player):
        return self.move_function(board, player, self.plus)


class NNPlayer:
    random_player_plus = RandomPlayer(random_move, True)

    def __init__(self, move_function, model, noise):
        RandomPlayer(random_move, True)
        self.move_function = move_function
        self.noise = noise
        self.model = model
        self.games = 0
        self.states = []
        self.winners = []
        self.winner_eval = []

    def make_move(self, board, player):
        return self.move_function(board, player, self.model, self.noise)

    def simulate_random_games(self, n=500):
        for _ in tqdm(range(n)):
            states, winners, winner = Game(
                self.random_player_plus, self.random_player_plus
            ).simulate()
            self.states.append(states)
            self.winners.append(winners)

    def simulate_noisy_game(self, n=100, noise_level=0.2):
        for _ in tqdm(range(n)):
            states, winners, winner = Game(
                NNPlayer(optimal_nn_move_noise, self.model, noise_level),
                NNPlayer(optimal_nn_move_noise, self.model, noise_level),
            ).simulate()
            self.states.append(states)
            self.winners.append(winners)

        self.games += n

        print(f"Model has been seen {self.games} self-play games")

    def eval_model_battle(self, opp, n=30, first=True):
        results = []
        for _ in tqdm(range(n)):
            game = Game(self, opp) if first else Game(opp, self)
            states, winners, winner = game.simulate()
            results.append(winner)

        self.winner_eval += results
        model_win = 1 if first else -1
        winning_perc = (pd.Series(results) == model_win).mean() * 100

        print(f"{np.round(winning_perc, 2)}% winning against {opp.name}")
        if use_wandb:
            wandb.log(
                {"winning_perc": winning_perc, "opp": opp.name},
                step=self.model.global_step,
            )

    def train_model(self, ntrain=10000, last_n_games=15000, save_every_n_games=2000):
        self.states = self.states[-last_n_games:]
        self.winners = self.winners[-last_n_games:]

        X = np.concatenate(self.states)
        y = np.concatenate(self.winners)

        moves_away = np.concatenate([np.arange(i.shape[0], 0, -1) for i in self.states])
        sample_weights = 1 / moves_away

        X_curr = X.reshape(-1, 1, 6, 7).astype("int8")
        y_curr = y

        print(X.shape)
        print(y.shape)

        if self.games % save_every_n_games == 0 and use_wandb:
            print("Saving Data:")
            np.save("X.npy", X_curr)
            np.save("y.npy", y_curr)
            np.save("sample_weights.npy", sample_weights)
            wandb.save("X.npy")
            wandb.save("y.npy")
            wandb.save("sample_weights.npy")

        choices = np.random.choice(np.arange(X_curr.shape[0]), ntrain)
        tr_x = X_curr[choices]
        tr_y = y_curr[choices]
        sample_weights = sample_weights[choices]

        dataset = TensorDataset(
            torch.tensor(tr_x).float(),
            torch.tensor(tr_y).float(),
            torch.tensor(sample_weights).float(),
        )

        data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

        # tr_x = np.flip(tr_x, 2)
        print("Training model")
        for batch_idx, (x, y, sample_weights) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            sample_weights = sample_weights.to(device)
            self.model.train()
            loss = self.model.train_step(x, y, sample_weights)
            if use_wandb:
                wandb.log({"train_loss": loss}, step=self.model.global_step)

        if self.games % save_every_n_games == 0:
            checkpoint_path = f"model_checkpoint_{self.model.global_step}.pth"
            torch.save(self.model, checkpoint_path)
            if use_wandb:
                wandb.save(checkpoint_path)


def play_vs(player_1, player_2, n_games=100):
    winner_eval = []

    for i in tqdm(range(n_games)):
        _, _, winner = Game(player_1, player_2).simulate()

        winner_eval.append(winner)

    return pd.Series(winner_eval).value_counts(normalize=True)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_mid, c_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_mid, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(c_mid),
            nn.Conv2d(c_mid, c_out, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(c_out),
        )

        self.resid = (
            nn.Conv2d(c_in, c_out, 1, padding="same")
            if c_in != c_out
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.resid(x)


class ResTower(nn.Module):
    def __init__(self, c_in, img_dim):
        super().__init__()
        self.tower = nn.Sequential(
            ResBlock(3, 1 * c_in, 1 * c_in),
            ResBlock(1 * c_in, 1 * c_in, 1 * c_in),
            nn.MaxPool2d(2),
            ResBlock(1 * c_in, 2 * c_in, 2 * c_in),
            ResBlock(2 * c_in, 2 * c_in, 2 * c_in),
            nn.MaxPool2d(2),
            ResBlock(2 * c_in, 4 * c_in, 4 * c_in),
            ResBlock(4 * c_in, 4 * c_in, 4 * c_in),
            nn.AvgPool2d(img_dim // 4),  # 256 dim for 32 input
            nn.Flatten(),
            nn.Linear(4 * c_in, 4 * c_in),
            nn.ReLU(),
            nn.Linear(4 * c_in, 10),
        )

    def forward(self, x):
        return self.tower(x)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.global_step = 0

        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, 4, padding="same"),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding="same"),
            nn.ReLU(),
            # nn.Conv2d(256, 256, 3, padding='same'),
            # nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10000, gamma=0.1
        )
        self.scaler = GradScaler()

    def forward(self, x):
        x = self.conv(x)
        return x

    def weighted_mse_loss(self, input, target, weight):
        return (weight * (input - target) ** 2).mean()

    def train_step(self, x, y, weight):
        with autocast():
            preds = self(x)
            y = y.view(-1, 1)
            weight = weight.view(-1, 1)
            loss = self.weighted_mse_loss(preds, y, weight)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.scheduler.step()
        self.global_step += 1

        return loss.item()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    random_player_plus = RandomPlayer(random_move, True)
    random_player_reg = RandomPlayer(random_move, False)

    # TODO: put this in a yaml file:
    n_iter = 100
    warm_start = False
    n_games = 1000
    ntrain = n_games * 50
    eval_games = 50
    noise_lb = noise_ub = 0.2
    noise_decay = 0.99
    noise_min = 0.15
    name = "simple_15k_pytorch"
    lr = 0.003
    opp = random_player_plus
    last_n_games = 50000
    save_every_n_games = 10000

    model = Model().to(device)
    num_params = count_parameters(model)
    print(num_params)

    nnplayer_regular = NNPlayer(optimal_nn_move_noise, model, 0)
    nnplayer_noise = NNPlayer(optimal_nn_move_noise, model, 0.2)

    config = {
        k: v
        for k, v in locals().items()
        if k
        in [
            "n_iter",
            "warm_start",
            "n_games",
            "ntrain",
            "eval_games",
            "noise_lb",
            "noise_ub",
            "name",
            "lr",
            "opp",
            "last_n_games",
        ]
    }

    wandb.init(project="connect_4", config=config)

    for i in range(n_iter):
        noise_lb = noise_ub = max(noise_lb * noise_decay, noise_min)
        nnplayer_regular.eval_model_battle(n=50, opp=opp, first=False)
        noise = np.random.uniform(noise_lb, noise_ub)
        print(noise)
        nnplayer_regular.simulate_noisy_game(n=n_games, noise_level=noise)
        nnplayer_regular.train_model(
            ntrain=ntrain,
            last_n_games=last_n_games,
            save_every_n_games=save_every_n_games,
        )
    wandb.finish()
