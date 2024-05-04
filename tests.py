import numpy as np
from nnplayer import (
    RandomPlayer,
    random_move,
    Game,
    Board,
    Model,
    device,
    count_parameters,
    NNPlayer,
    optimal_nn_move_noise,
)


###basic gameplay:
def test_win():
    board = Board()
    assert board.winning_move(1) is not True

    for _ in range(3):
        board.make_move_inplace(i=0, player=1)
        board.make_move_inplace(i=1, player=-1)
        print(board.s)
        assert board.winning_move(1) is not True
        assert board.winning_move(-1) is not True

    board.make_move_inplace(0, 1)
    assert board.winning_move(-1) is not True
    assert board.winning_move(1) is True


def test_random_play():
    random_player_1 = RandomPlayer(random_move, True)
    random_player_2 = RandomPlayer(random_move, False)

    game = Game(random_player_1, random_player_2)
    out = game.simulate()
    assert (len(out)) == 3


####test reinforcement learning parts:
def test_nn_model_load():
    pass


def test_nn_model_play():
    pass


def test_nn_model_train():
    pass


def test_end_to_end():
    #random_player_plus = RandomPlayer(random_move, True)
    random_player_reg = RandomPlayer(random_move, False)

    n_iter = 3
    n_games = 100
    ntrain = n_games * 50
    noise_lb = noise_ub = 0.2
    noise_decay = 0.99
    noise_min = 0.15
    opp = random_player_reg
    last_n_games = 50000
    save_every_n_games = 10000

    model = Model().to(device)
    num_params = count_parameters(model)
    print(num_params)

    nnplayer_regular = NNPlayer(optimal_nn_move_noise, model, 0)

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
