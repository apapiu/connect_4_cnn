from models import Model, device, count_parameters
import numpy as np
import wandb    
from players import RandomPlayer, NNPlayer, optimal_nn_move_noise, random_move

use_wandb = False

if __name__ == "__main__":
    random_player_plus = RandomPlayer(random_move, True)
    random_player_reg = RandomPlayer(random_move, False)

    # TODO: put this in a yaml file:
    n_iter = 2
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

    # Save move history data for transformer training
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"move_history_{timestamp}.json"
    with open(json_filename, "w") as f:
        json.dump(nnplayer_regular.move_history, f)
