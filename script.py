import psycopg2
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from collections import defaultdict

from plotly.offline import iplot, init_notebook_mode
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()

%config InlineBackend.figure_format = 'retina'
import os
import datetime

import warnings
warnings.filterwarnings('ignore')
from numba import jit, njit

from hightower import forecasting

plt.style.use('seaborn')

import tensorflow as tf

#Assume that the number of cores per socket in the machine is denoted as NUM_PARALLEL_EXEC_UNITS
#  when NUM_PARALLEL_EXEC_UNITS=0 the system chooses appropriate settings 

# config = tf.ConfigProto(intra_op_parallelism_threads=2, 
#                         inter_op_parallelism_threads=2, 
#                         allow_soft_placement=True,
#                         device_count = {'CPU': 2})

# session = tf.Session(config=config)

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import CategoricalCrossentropy as CC
from tensorflow.keras.losses import SparseCategoricalCrossentropy as SCC
import tensorflow as tf

import pickle

from scipy.signal import convolve2d

def make_move(s, i, player):
    col = s[:, i]
    lev = 6 - (col == 0).sum()
    s_new = s.copy()
    s_new[lev, i] = player
    return s_new

def get_possible_moves(s, player):
    moves = 6-(s == 0).sum(0)
    
    return np.nonzero(moves < 6)[0]

def play_random_move(s, player):
    pos_moves = get_possible_moves(s, player)
    
    rand_move = np.random.choice(pos_moves)
    
    s_new = make_move(s, rand_move, player)
    
    return s_new
        
    
horizontal_kernel = np.array([[ 1, 1, 1, 1]])
vertical_kernel = horizontal_kernel.T
diag1_kernel = np.eye(4, dtype=np.uint8)
diag2_kernel = np.fliplr(diag1_kernel)
detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]    
 
def winning_move(board, player):
    for kernel in detection_kernels:
        if (convolve2d(board == player, kernel, mode="valid") == 4).any():
            return True
    return False


def get_nn_preds(s, model, player):
    possible_moves = get_possible_moves(s, player)
    moves_np = []
    for i in possible_moves:
        s_new = s.copy()
        s_new = make_move(s_new, i, player)
        moves_np.append(s_new)

    moves_np = np.array(moves_np).reshape(-1,6,7,1)

    preds = model.predict(moves_np)[:, 0]
    preds = pd.Series(preds, possible_moves)
    
    return preds

def optimal_nn_move(s, player, model, *args):
    
    preds = get_nn_preds(s, model, player)
    preds = preds 
    best_move = preds.idxmax()
    s_new = make_move(s, best_move, player)
    
    return s_new

def get_ucb_for_next_moves(s, player):
    possible_moves = get_possible_moves(s, player)
    ucb_counts = []
    for i in possible_moves:
                s_new = s.copy()
                s_new = make_move(s_new, i, player)
                #print(s_new)
                #print(get_count(s_new))
                ucb_counts.append(get_count(s_new))

    ucb_counts = np.array(ucb_counts)

    return (1/(ucb_counts+1))*1/2


def optimal_nn_move_noise(s, player, model, use_2ply_check, std_noise):
    
    if use_2ply_check:
        possible_moves = get_possible_moves(s, player)

        for i in possible_moves:
            s_new = s.copy()
            s_new = make_move(s_new, i, player)

            if winning_move(s_new, player):
                return s_new

        for i in possible_moves:
            s_new = s.copy()
            s_new = make_move(s_new, i, (-1)*player)

            if winning_move(s_new, (-1)*player):
                s_new = s.copy()
                s_new = make_move(s_new, i, player)
                return s_new    
    
    preds = get_nn_preds(s, model, player)
   

    ucb_bonus = get_ucb_for_next_moves(s, player)
    #print(ucb_bonus)
    preds = preds + np.random.normal(0, std_noise, len(preds)) + use_2ply_check*ucb_bonus
    best_move = preds.idxmax()
    s_new = make_move(s, best_move, player)
    
    return s_new


def random_move(s, player, use_2ply_check):
    
    possible_moves = get_possible_moves(s, player)
    bad_moves = []
    if use_2ply_check:
        for i in possible_moves:
            s_new = s.copy()
            s_new = make_move(s_new, i, player)

            if winning_move(s_new, player):
                return s_new

        for i in possible_moves:
            s_new = s.copy()
            s_new = make_move(s_new, i, (-1)*player)

            if winning_move(s_new, (-1)*player):
                s_new = s.copy()
                s_new = make_move(s_new, i, player)
                return s_new
            
#         for i in possible_moves:
            
#             s_new = s.copy()
#             s_new = make_move(s_new, i, player)
            
#             pos_mov2 = get_possible_moves(s_new, player*(-1))
            
#             for j in pos_mov2:
#                 s_new2 = s_new.copy()
#                 s_new2 = make_move(s_new2, j, player*(-1))
                
#                 if winning_move(s_new2, (-1)*player):
#                     bad_moves.append(i)
    
#     non_lose = np.setdiff1d(possible_moves, bad_moves)
    
#     if len(non_lose) > 0:
#         possible_moves = non_lose
    
    rand_move = np.random.choice(possible_moves)
    s_new = make_move(s, rand_move, player)
    return s_new        
    
    
def simulate_game(player_1, 
                  player_2,
                  append_states = True,
                  plot_rez=False):
    s = np.zeros([6,7])
    game_states = []
    player = 1
    winner = None
    move_num = 0
    
    while winner is None:
        #display(plot_state(s))
        if player == 1:
            s = player_1.make_move(s, player)
        else: 
            s = player_2.make_move(s, player)
            
        game_states.append(s.copy())
        
        if winning_move(s, player):
            winner = player
            break
        
        move_num += 1
        if move_num == 42:
            winner = 0
            
        player *= -1
        
    
    if append_states:
        game_states = np.array(game_states).astype("int8")

        turns = np.empty((len(game_states),))
        turns[::2] = 1
        turns[1::2] = -1
        turns = turns*winner

        states.append(game_states)
        winners.append(turns)
        
    if plot_rez: 
        last_state = game_states[-1]
        prev_state = game_states[-3]
        print(get_nn_preds(game_states[-3], model , (-1)*player))
        display(plot_state(prev_state))
        print(get_nn_preds(game_states[-2], model , player))
        display(plot_state(last_state))
               
    return winner

def negamax(s, depth, player, pred_dict):
    
    if winning_move(s, player):
        return 1
    
    if (s == 0).sum() == 0:
        return 0

    if (depth == 0):
        return pred_dict[s.reshape(-1,6,7,1).tobytes()]

    v = -100
    
    possible_moves = get_possible_moves(s, player)
    
    ys = []
    for i in possible_moves:
        s_new = s.copy()
        s_new = make_move(s_new, i, -player)
        #display(plot_state(s_new))
        neg_v = negamax(s_new, depth-1, -player, pred_dict)
        #print(neg_v)
        ys.append(neg_v)
        #v = - np.max([v,neg_v])
    v = - np.max(ys)
    return v


def get_states(s, depth, player, term_states = []):
    
    if depth == 0:
        return term_states
    
    possible_moves = get_possible_moves(s, player)
    
    for i in possible_moves:
        s_new = s.copy()
        s_new = make_move(s_new, i, player)
        if depth == 1:
            term_states.append(s_new.reshape(1,6,7))
        elif winning_move(s_new, player):    
            term_states.append(s_new.reshape(1,6,7))
        elif (s_new == 0).sum() == 0:
            term_states.append(s_new.reshape(1,6,7))
            
        #print(depth)
        #display(plot_state(s_new))
        term_states = get_states(s_new, depth-1, player*(-1), term_states)
    return term_states
    
def get_negamax_move(s, player, model, plus=2, noise=0.1):
    
    depth = plus
    #display(plot_state(s))
    term_states = get_states(s, depth=depth, player=player, term_states=[])
    
    if noise == 1:
        preds = np.random.normal(0, 0.1, len(term_states))
    else:
        preds = model.predict(np.concatenate(term_states).reshape(-1, 6, 7, 1), batch_size=1024)[:, 0]

    keyz = [i.tobytes() for i in term_states]
    pred_dict = dict(zip(keyz, preds))
        
    possible_moves = get_possible_moves(s, player)

    vals = []
    states_int = []
    for i in possible_moves:
        s_new = s.copy()
        s_new = make_move(s_new, i, player)
        states_int.append(s_new)
        vals.append(negamax(s_new, depth-1, player, pred_dict))

    cr = (noise != 1)
    vals_negamax.append(vals)
    vals = pd.Series(vals, index=possible_moves) 
    vals = vals + (cr)*np.random.normal(0, noise, len(vals))
    print(vals)
    
    states_negamax.append(states_int)
    
    best_move = vals.idxmax()
    s_new = make_move(s, best_move, player)
    return s_new
            

def play_vs(player_1, player_2, n_games = 100,
            append_states=False, plot_rez=False):
    
    wins = pd.Series([simulate_game(player_1, player_2,  
                                    append_states=append_states, plot_rez=plot_rez) 
                  for i in range(n_games)])
    return wins.value_counts(normalize=True)

def build_model(lr = 0.001):
        
    model = Sequential()
    model.add(Conv2D(64, (4, 4), padding="same", activation='relu', input_shape=(6,7,1)))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dense(units = 1, activation = 'linear'))

    opt = Adam(learning_rate=lr)

    model.compile(loss="mean_squared_error",
                  optimizer=opt)
    
    return model

def build_bigger_model(lr=0.001):
    model = Sequential()
    model.add(Conv2D(64, (4, 4), padding="same", activation='relu', input_shape=(6,7,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dense(units = 1, activation = 'linear'))

    opt = Adam(learning_rate=lr)

    model.compile(loss="mean_squared_error",
                  optimizer=opt)

    return model

class NNPlayer:
    
    def __init__(self, move_function, model, noise, plus):
        self.move_function = move_function
        self.noise = noise
        self.model = model
        self.plus = plus
    
    def make_move(self, s, player):
        return self.move_function(s, player, self.model, self.plus, self.noise)
        
        
class RandomPlayer:
    
    def __init__(self, move_function, plus):
        self.move_function = move_function
        self.plus = plus
    
    def make_move(self, s, player):
        return self.move_function(s, player, self.plus)

def play_line(s, player, model, n=2):
    cur_player = player
    #print(s)
    if (s == 0).sum() == 0:
            return 0
    
    for i in range(n):
        #print(s)
        
        s = optimal_nn_move(s, model, cur_player)
        
        if winning_move(s, cur_player):
            return 1 if (cur_player==player) else -1
        
        if (s == 0).sum == 0:
            return 0
        
        cur_player *= -1
    return model.predict(s.reshape(-1,6,7,1))[0][0]   


def serializeAndCompress(value, verbose=True):
  serializedValue = pickle.dumps(value)
  if verbose:
    print('Lenght of serialized object:', len(serializedValue))
  c_data =  zlib.compress(serializedValue, 9)
  if verbose:
    print('Lenght of compressed and serialized object:', len(c_data))
  return b64.b64encode(c_data)

def decompressAndDeserialize(compresseData):
  d_data_byte = b64.b64decode(compresseData)
  data_byte = zlib.decompress(d_data_byte)
  value = pickle.loads(data_byte)
  return value


s = np.zeros([6,7])

player = []
states = []
winners = []
winz = []

model = build_model(lr=0.001)

nnplayer_regular = NNPlayer(optimal_nn_move_noise, model, 0, False)
random_player_plus = RandomPlayer(random_move, True)
nnplayer_noise = NNPlayer(optimal_nn_move_noise, model, 0.2, False)


winz = []


--%%time 

for i in range(20):
    
    wins = pd.Series([simulate_game(random_player_plus,
                                    nnplayer_regular, 
                                    append_states=False) 
                      for i in range(25)]).value_counts(normalize=True)
    winz.append(wins)
    print(len(states))
    print(wins)
    
    
    if i > 25:
        wins = pd.Series([simulate_game(nnplayer_noise, nnplayer_noise, plot_rez=False) 
                          for i in range(1000)])
    else:
        wins = pd.Series([simulate_game( RandomPlayer(random_move, False), 
                                         RandomPlayer(random_move, False), plot_rez=False) 
                          for i in range(1000)])
    
    print(wins.value_counts())
    avg_len = np.array([len(i) for i in states[-1000:]]).mean()
    print("average length: {0}".format(avg_len))

    X_c = np.concatenate(states[-1000:])
    enc_states = pd.Series([i.tobytes() for i in X_c])

    perc_unique = enc_states.unique().shape[0]/enc_states.shape[0]
    print("perc unique: {0}".format(perc_unique))
    
    X = np.concatenate(states[-300000:])
    y = np.concatenate(winners[-300000:])
    
    moves_away = np.concatenate([np.arange(i.shape[0], 0, -1) for i in states[-300000:]])
    sample_weights = (1/np.sqrt(moves_away))
    
    X_curr = X.reshape(-1,6,7,1).astype("int8")
    y_curr = y
        
    choices = np.random.choice(np.arange(X_curr.shape[0]), 40000)
    tr_x = X_curr[choices]
    tr_y = y_curr[choices]
    sample_weights = sample_weights[choices]
    
    if i%2 == 0:
        tr_x = np.flip(tr_x, 2)

    model.fit(x=tr_x,
              y=tr_y, 
              epochs=1, 
              sample_weight=sample_weights,
              batch_size=4*128)



pd.concat(winz,1).T.fillna(0).rolling(7).mean().plot()

%%time
play_vs(RandomPlayer(random_move, True),
        policy_player,
        10,
   plot_rez=True)


%%time
play_vs(NNPlayer(optimal_nn_move_noise, model_res, 0.1, False),  
             NNPlayer(optimal_nn_move_noise, model_bench, 0.1, False), 
            250, plot_rez=False)
