import os 
import wandb
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import convolve2d


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from torch.cuda.amp import GradScaler, autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set os.environ["WANDB_API_KEY"]
# !wandb login

class Board:

     #convolutional kernels to detect a win:
    detection_kernels = [
        np.array([[1, 1, 1, 1]]), #horizontal_kernel
        np.array([[1], [1], [1], [1]]), #vertical_kernel
        np.eye(4, dtype=np.uint8), #diag1_kernel
        np.fliplr(np.eye(4, dtype=np.uint8)) #diag2_kernel
    ]

    """A board"""

    def __init__(self):
        self.s = np.zeros([6, 7])
        #self.player?

    def get_possible_moves(self):
        moves = 6 - (self.s == 0).sum(0)
        return np.nonzero(moves < 6)[0]

    def make_move_inplace(self, i, player):
        lev = 6 - np.count_nonzero(self.s[:, i] == 0)
        self.s[lev, i] = player

    @staticmethod #note: doing this on GPU makes it slower
    def check_winning_move(s, player):
        for kernel in Board.detection_kernels:
            if (convolve2d(s == player, kernel, mode="valid") == 4).any():
                return True
        return False

    def winning_move(self, player):
        return self.check_winning_move(self.s, player)

class Game:
    def __init__(self, player_1, player_2, board=None, plot_rez=False):
        self.player_1 = player_1
        self.player_2 = player_2
        self.plot_rez = plot_rez
        self.board = board if board is not None else Board() #start with empty board is no board given
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
        turns = turns * self.winner
        self.states = game_states_np
        self.winners = turns

def play_random_move(board, player):
    pos_moves = board.get_possible_moves()
    rand_move = np.random.choice(pos_moves)
    board.make_move_inplace(rand_move, player)

def get_nn_preds(board, model, player):
    possible_moves = board.get_possible_moves()
    moves_np = []
    for i in possible_moves:
        s_new = board.s.copy()
        board.make_move_inplace(i, player)
        moves_np.append(board.s)
        board.s = s_new.copy() # Revert move

    moves_np = np.array(moves_np).reshape(-1, 1, 6, 7)

    model.eval()
    with torch.no_grad():
        moves_np = torch.tensor(moves_np).float().to(device)
        preds = model(moves_np).cpu().numpy()[:, 0]

    return pd.Series(preds, possible_moves)


def optimal_nn_move(board, model, player):
    preds = get_nn_preds(board, model, player)
    best_move = preds.idxmax()
    board.make_move_inplace(best_move, player)


def optimal_nn_move_noise(board, player, model, std_noise=0):
    preds = get_nn_preds(board, model, player)
    preds = preds + np.random.normal(0, std_noise, len(preds))
    best_move = preds.idxmax()
    board.make_move_inplace(best_move, player)


def random_move(board, player, use_2ply_check):
    possible_moves = board.get_possible_moves()

    if use_2ply_check:
        for i in possible_moves:
            s_new = board.s.copy()
            board.make_move_inplace(i, player)
            if board.winning_move(player):
                return #state is changed inplace
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

class Player:
    pass

class NNPlayer:

    def __init__(self, move_function, model, noise):
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
            states, winners, winner = Game(random_player_plus, random_player_plus).simulate()
            self.states.append(states)
            self.winners.append(winners)

    def simulate_noisy_game(self, n=100, noise_level=0.2):

        for _ in tqdm(range(n)):
            states, winners, winner = Game(NNPlayer(optimal_nn_move_noise, self.model, noise_level),
                                           NNPlayer(optimal_nn_move_noise, self.model, noise_level)).simulate()
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
        wandb.log({"winning_perc": winning_perc, "opp": opp.name}, step=self.model.global_step)


    def train_model(self, ntrain=10000, last_n_games=15000):
        self.states = self.states[-last_n_games:]
        self.winners = self.winners[-last_n_games:]

        X = np.concatenate(self.states)
        y = np.concatenate(self.winners)

        moves_away = np.concatenate([np.arange(i.shape[0], 0, -1) for i in self.states])
        sample_weights = (1 / moves_away )

        X_curr = X.reshape(-1, 1, 6, 7).astype("int8")
        y_curr = y

        print(X.shape)
        print(y.shape)

        choices = np.random.choice(np.arange(X_curr.shape[0]), ntrain)
        tr_x = X_curr[choices]
        tr_y = y_curr[choices]
        sample_weights = sample_weights[choices]

        dataset = TensorDataset(torch.tensor(tr_x).float(),
                                torch.tensor(tr_y).float(),
                                torch.tensor(sample_weights).float()
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
            wandb.log({"train_loss":loss}, step=self.model.global_step)

    def simulate_and_train(self, iterations=5):
        for i in range(iterations):
            self.simulate_random_games()
            self.train_model()
            self.eval_model_battle()


class RandomPlayer:

    def __init__(self, move_function, plus):
        self.move_function = move_function
        self.plus = plus
        self.name = 'random_player_2ply' if self.plus else 'random_player'

    def make_move(self, board, player):
        return self.move_function(board, player, self.plus)

def play_vs(player_1, player_2,
            n_games=100):
    winner_eval = []

    for i in tqdm(range(n_games)):
        _, _, winner = Game(player_1,
                            player_2).simulate()

        winner_eval.append(winner)

    return pd.Series(winner_eval).value_counts(normalize=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.global_step = 0

        self.conv = model = nn.Sequential(
                nn.Conv2d(1, 128, 4, padding='same'),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, padding='same'),
                nn.ReLU(),
                # nn.Conv2d(256, 256, 3, padding='same'),
                # nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(256*3*3, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        self.optimizer=torch.optim.Adam(self.parameters(), lr=0.0003)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.1)
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


if __name__ == '__main__':
     random_player_plus = RandomPlayer(random_move, True)
     random_player_reg = RandomPlayer(random_move, False)
     
     # TODO: put this in a yaml file:
     n_iter = 750
     warm_start = False
     n_games = 1000
     ntrain = n_games*30
     eval_games = 50
     noise_lb = noise_ub = 0.2
     noise_decay = 0.99
     noise_min = 0.15
     name = "simple_15k_pytorch"
     lr = 0.003
     opp = random_player_plus
     last_n_games = 40000
     
     
     model = Model().to(device)
     num_params = count_parameters(model)
     print(num_params)
     
     nnplayer_regular = NNPlayer(optimal_nn_move_noise, model, 0)
     nnplayer_noise = NNPlayer(optimal_nn_move_noise, model, 0.2)
     
     
     config = {k: v for k, v in locals().items() if k in ['n_iter', 'warm_start', 'n_games',
                                                             'ntrain', 'eval_games', 'noise_lb',
                                                             'noise_ub', 'name', 'lr', 'opp', 'last_n_games']}
     
     wandb.init(
         project="connect_4",
         config = config)
     
     for i in range(n_iter):
         noise_lb = noise_ub = max(noise_lb*noise_decay, noise_min)
         nnplayer_regular.eval_model_battle(n=50, opp=opp, first=False)
         noise = np.random.uniform(noise_lb, noise_ub)
         print(noise)
         nnplayer_regular.simulate_noisy_game(n=n_games, noise_level=noise)
         nnplayer_regular.train_model(ntrain=ntrain, last_n_games=last_n_games)
     wandb.finish()
