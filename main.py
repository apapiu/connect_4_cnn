from models import Model, device, count_parameters
import numpy as np
import wandb    
from players import RandomPlayer, NNPlayer, optimal_nn_move_noise, random_move
from config import config

if __name__ == "__main__":
    random_player_plus = RandomPlayer(random_move, True)
    random_player_reg = RandomPlayer(random_move, False)

    # Configuration
    opp = random_player_plus

    model = Model().to(device)
    num_params = count_parameters(model)
    print(num_params)

    nnplayer_regular = NNPlayer(optimal_nn_move_noise, model, 0)
    nnplayer_noise = NNPlayer(optimal_nn_move_noise, model, 0.2)

    if config.use_wandb:
        wandb_config = {
            "n_iter": config.n_iter,
            "n_games": config.n_games,
            "ntrain": config.ntrain,
            "learning_rate": config.learning_rate,
            "experiment_name": config.experiment_name,
        }
        wandb.init(project="connect_4", config=wandb_config)

    # Training loop
    noise_current = config.noise_initial
    for i in range(config.n_iter):
        noise_current = max(noise_current * config.noise_decay, config.noise_min)
        nnplayer_regular.eval_model_battle(n=config.eval_games, opp=opp, first=False)
        noise = np.random.uniform(0, noise_current)
        print(f"Iteration {i+1}/{config.n_iter}, Noise: {noise:.3f}")
        nnplayer_regular.simulate_noisy_game(n=config.n_games, noise_level=noise)
        nnplayer_regular.train_model(
            ntrain=config.ntrain,
            last_n_games=config.last_n_games,
            save_every_n_games=config.save_every_n_games,
        )
    if config.use_wandb:
        wandb.finish()

    # Save move history data for transformer training
    filename = nnplayer_regular.save_move_history()
    print(f"Saved move history to {filename}")
