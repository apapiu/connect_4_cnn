import os
import yaml

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import convolve2d

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization
from tensorflow.keras.experimental import CosineDecay

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Add
from tensorflow.keras.models import Model


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

    @staticmethod
    def check_winning_move(s, player):
        for kernel in Board.detection_kernels:
            if (convolve2d(s == player, kernel, mode="valid") == 4).any():
                return True
        return False
    
    def winning_move(self, player):
        return self.check_winning_move(self.s, player)
    

######Moves:

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
        
    moves_np = np.array(moves_np).reshape(-1, 6, 7, 1)
    preds = model(moves_np).numpy()[:, 0]
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

    def train_model(self, ntrain=10000, last_n_games=15000):
        self.states = self.states[-last_n_games:]
        self.winners = self.winners[-last_n_games:]

        X = np.concatenate(self.states)
        y = np.concatenate(self.winners)

        moves_away = np.concatenate([np.arange(i.shape[0], 0, -1) for i in self.states])
        sample_weights = (1 / np.sqrt(moves_away))

        X_curr = X.reshape(-1, 6, 7, 1).astype("int8")
        y_curr = y

        print(X.shape)

        choices = np.random.choice(np.arange(X_curr.shape[0]), ntrain)
        tr_x = X_curr[choices]
        tr_y = y_curr[choices]
        sample_weights = sample_weights[choices]

        # tr_x = np.flip(tr_x, 2)

        self.model.fit(x=tr_x,
                       y=tr_y,
                       epochs=1,
                       sample_weight=sample_weights,
                       batch_size=256)

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


def build_model(lr=0.001, resnet=False, n_blocks=6):
    
    if resnet:
        input_layer = Input(shape=(6, 7, 1))

        x = Conv2D(128, (3, 3), padding="same", activation="relu")(input_layer)

        # Residual blocks
        for _ in range(n_blocks): 
            residual = x  

            x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
            x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
            x = Add()([x, residual])

        x = Flatten()(x)
        x = Dense(units=128, activation='relu')(x)
        x = Dense(units=128, activation='relu')(x)
        output_layer = Dense(units=1, activation='linear', dtype='float32')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
    else:
       
        model = Sequential()
        model.add(Conv2D(64, (3, 3), padding="same", activation='relu', input_shape=(6, 7, 1)))
        model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
        model.add(Conv2D(256, (3, 3), padding="same", activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=1, activation='linear', dtype='float32'))
    
    
    #lr_schedule = CosineDecay(initial_learning_rate=1e-2, decay_steps=10000)
    opt = Adam(learning_rate=lr)

    model.compile(loss="mean_squared_error",
                  optimizer=opt)

    return model


def play_vs(player_1, player_2,
            n_games=100):
    winner_eval = []

    for i in tqdm(range(n_games)):
        _, _, winner = Game(player_1,
                            player_2).simulate()

        winner_eval.append(winner)

    return pd.Series(winner_eval).value_counts(normalize=True)


if __name__ == "__main__":
    
    random_player_plus = RandomPlayer(random_move, True)
    random_player_reg = RandomPlayer(random_move, False)

    # TODO: put this in a yaml file:
    n_iter = 750
    warm_start = False
    n_games = 250
    ntrain = n_games*40
    eval_games = 50
    noise_lb = 0.10
    noise_ub = 0.25
    name = "simple_15k_mem"
    lr = 0.001
    opp = random_player_reg
    last_n_games = 15000
    
    
    # model = load_model('my_model_varying_noise_resnet.keras')
    model = build_model(lr=lr)
    print(model.summary())
    
    if not os.path.exists(name):
        os.makedirs(name)
    
    nnplayer_regular = NNPlayer(optimal_nn_move_noise, model, 0)
    nnplayer_noise = NNPlayer(optimal_nn_move_noise, model, 0.2)
    
    config = {k: v for k, v in locals().items() if k in ['n_iter', 'warm_start', 'n_games', 
                                                         'ntrain', 'eval_games', 'noise_lb', 
                                                         'noise_ub', 'name', 'lr', 'opp', 'last_n_games']}
    
    with open(os.path.join(name, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(config, yaml_file)
        
  
    with open(os.path.join(name, 'model_config.yaml'), 'w') as yaml_file:
        yaml_file.write(model.to_json())

    for i in range(n_iter):
        nnplayer_regular.eval_model_battle(n=50, opp=opp, first=False)
        noise = np.random.uniform(noise_lb, noise_ub)
        print(noise)
        nnplayer_regular.simulate_noisy_game(n=n_games, noise_level=noise)
        nnplayer_regular.train_model(ntrain=ntrain, last_n_games=last_n_games)
        
        results = pd.Series(nnplayer_regular.winner_eval) == -1
        results_g = results.groupby(lambda x: x // eval_games).mean()
        results_g.to_csv(os.path.join(name, "results.csv"))
        nnplayer_regular.model.save(os.path.join(name, 'my_model.keras'))   