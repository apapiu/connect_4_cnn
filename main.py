from models import Model, device, count_parameters
import numpy as np
import wandb    
from players import RandomPlayer, NNPlayer, optimal_nn_move_noise, random_move, TransformerPlayer
from config import config, transformer_config
from transformer import Transformer
from tqdm import tqdm
import pandas as pd
from game import Game


def play_vs(player_1, player_2, n_games=100):
    winner_eval = []

    for i in tqdm(range(n_games)):
        _, _, winner, _ = Game(player_1, player_2).simulate()

        winner_eval.append(winner)

    print(pd.Series(winner_eval).value_counts(normalize=True))

    return pd.Series(winner_eval).value_counts(normalize=True)


# Initialize transformer with config
transformer = Transformer(transformer_config)
transformer.to(device)

if __name__ == "__main__":
    random_player_plus = RandomPlayer(random_move, True)
    random_player_reg = RandomPlayer(random_move, False)

    # Configuration
    opp = random_player_plus

    model = Model().to(device)
    num_params = count_parameters(model)
    print(num_params)

    nnplayer_regular = NNPlayer(optimal_nn_move_noise, model, 0, transformer)
    nnplayer_noise = NNPlayer(optimal_nn_move_noise, model, 0.2, transformer)
    transformer_player = TransformerPlayer(model=transformer)

    if config.use_wandb:
        wandb_config = {
            "n_iter": config.n_iter,
            "n_games": config.n_games,
            "ntrain": config.ntrain,
            "learning_rate": config.learning_rate,
            "experiment_name": config.experiment_name,
        }
        wandb.init(project="connect_4_transformer", config=wandb_config)

    # Training loop
    noise_current = config.noise_initial
    for i in range(config.n_iter):
        noise_current = max(noise_current * config.noise_decay, config.noise_min)
        nnplayer_regular.eval_model_battle(n=config.eval_games, opp=opp, first=False)
        print("EVALS Transformer player:")
        result = play_vs(random_player_plus, transformer_player, n_games=100)
        noise = np.random.uniform(noise_current, noise_current)
        print(f"Iteration {i+1}/{config.n_iter}, Noise: {noise:.3f}")
        nnplayer_regular.simulate_noisy_game(n=config.n_games, noise_level=noise)
        nnplayer_regular.train_model(
            ntrain=config.ntrain,
            last_n_games=config.last_n_games,
            save_every_n_games=config.save_every_n_games,
        )
        nnplayer_regular.train_transformer_model()
    if config.use_wandb:
        wandb.finish()

    # Save move history data for transformer training
    filename = nnplayer_regular.save_move_history()
    print(f"Saved move history to {filename}")
