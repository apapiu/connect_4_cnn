
# TODO: put this in a Board class
# TODO: Use a generic Player class.

def make_move_inplace(s, i, player):
    lev = 6 - np.count_nonzero(s[:, i] == 0)
    s[lev, i] = player

def get_possible_moves(s):
    moves = 6-(s == 0).sum(0) 
    return np.nonzero(moves < 6)[0]

def play_random_move(s, player):
    pos_moves = get_possible_moves(s)
    rand_move = np.random.choice(pos_moves)
    make_move_inplace(s, rand_move, player)
        
    
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
    possible_moves = get_possible_moves(s)
    moves_np = []
    for i in possible_moves:
        s_new = s.copy()
        make_move_inplace(s_new, i, player)
        moves_np.append(s_new)

    moves_np = np.array(moves_np).reshape(-1, 6, 7, 1)
    preds = model.predict(moves_np, verbose=0)[:, 0]
    return pd.Series(preds, possible_moves)


def optimal_nn_move(s, model, player):
    preds = get_nn_preds(s, model, player)
    best_move = preds.idxmax()
    make_move_inplace(s, best_move, player)


def optimal_nn_move_noise(s, player, model, use_2ply_check=False, std_noise=0):
    
    preds = get_nn_preds(s, model, player)
    preds = preds + np.random.normal(0, std_noise, len(preds))
    best_move = preds.idxmax()
    make_move_inplace(s, best_move, player)

    
def random_move(s, player, use_2ply_check):
    
    possible_moves = get_possible_moves(s)

    if use_2ply_check:
        for i in possible_moves:
            s_new = s.copy()
            make_move_inplace(s_new, i, player)
            if winning_move(s_new, player):
                make_move_inplace(s, i, player)
                return s  

        for i in possible_moves:
            s_new = s.copy()
            make_move_inplace(s_new, i, (-1) * player)
            if winning_move(s_new, (-1) * player):
                make_move_inplace(s, i, player)
                return s  
    
    rand_move = np.random.choice(possible_moves)
    make_move_inplace(s, rand_move, player)

class Game:
    def __init__(self, player_1, player_2, plot_rez=False):
        self.player_1 = player_1
        self.player_2 = player_2
        self.plot_rez = plot_rez
        self.s = np.zeros([6, 7])
        self.game_states = []
        self.player = 1
        self.winner = None
        self.move_num = 0
        self.states = []  
        self.winners = []  

    def simulate(self):
        while self.winner is None:
            if self.player == 1:
                self.player_1.make_move(self.s, self.player)
            else:
                self.player_2.make_move(self.s, self.player)

            self.game_states.append(self.s.copy())

            if winning_move(self.s, self.player): 
                self.winner = self.player
                break

            self.move_num += 1
            if self.move_num == 42:
                self.winner = 0

            self.player *= -1

        self.append_game_results()

        #X and y for training
        return self.states, self.winners, self.winner

    def append_game_results(self):
        game_states_np = np.array(self.game_states).astype("int8")

        #who won from current position:
        turns = np.empty((len(self.game_states),))
        turns[::2] = 1
        turns[1::2] = -1
        turns = turns * self.winner
        self.states = game_states_np
        self.winners = turns


class NNPlayer:
    
    def __init__(self, move_function, model, noise, plus):
        self.move_function = move_function
        self.noise = noise
        self.model = model
        self.plus = plus
        self.states = []
        self.winners = []
        self.winner_eval = []
    
    def make_move(self, s, player):
        return self.move_function(s, player, self.model, self.plus, self.noise)

    def simulate_random_games(self, n=500):
        
        for i in tqdm(range(n)):

            states, winners, winner = Game(random_player_plus, random_player_plus).simulate()
            self.states.append(states)
            self.winners.append(winners)

    def simulate_noisy_game(self, n=100):
        
        for i in tqdm(range(n)):

            states, winners, winner = Game(NNPlayer(optimal_nn_move_noise, self.model, 0.2, False), 
                                           NNPlayer(optimal_nn_move_noise, self.model, 0.2, False)).simulate()
            self.states.append(states)
            self.winners.append(winners)
    

    def eval_model_battle(self, n=30):

        for i in tqdm(range(n)):
            states, winners, winner = Game(self, 
                                           random_player_plus).simulate()

            self.states.append(states)
            self.winners.append(winners)
            self.winner_eval.append(winner)


    def train_model(self, ntrain=10000):

        X = np.concatenate(self.states[-300000:])
        y = np.concatenate(self.winners[-300000:])

        moves_away = np.concatenate([np.arange(i.shape[0], 0, -1) for i in self.states[-300000:]])
        sample_weights = (1/np.sqrt(moves_away))

        X_curr = X.reshape(-1,6,7,1).astype("int8")
        y_curr = y
    
        print(X.shape)

        choices = np.random.choice(np.arange(X_curr.shape[0]), ntrain)
        tr_x = X_curr[choices]
        tr_y = y_curr[choices]
        sample_weights = sample_weights[choices]

        #tr_x = np.flip(tr_x, 2)
        
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
    
    def make_move(self, s, player):
        return self.move_function(s, player, self.plus)

def build_model(lr = 0.001):
        
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu', input_shape=(6,7,1)))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(Flatten())
    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dense(units = 32, activation = 'relu'))
    model.add(Dense(units = 1, activation = 'linear', dtype='float32'))

    opt = Adam(learning_rate=lr)

    model.compile(loss="mean_squared_error",
                  optimizer=opt)
    
    return model
