from typing import Protocol, cast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from models import device
from board import Board
from transformer import Transformer, get_probabilities
from torch.utils.data import DataLoader, TensorDataset

use_wandb = False


class Player(Protocol):
    # makes move inplace on board and returns the column played:
    def make_move(self, board: Board, player: int, move_history: list[int] | None = None) -> int: ...


class RandomPlayer:
    def __init__(self, move_function, plus):
        self.move_function = move_function
        self.plus = plus
        self.name = "random_player_2ply" if self.plus else "random_player"

    def make_move(self, board, player, move_history = None):
        return self.move_function(board, player, self.plus)


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

def optimal_transformer_move(board: Board, model: nn.Module, move_history: list[int], player:int):
    possible_moves = board.get_possible_moves()
    
    # Get probabilities for all moves
    preds = get_probabilities(model, move_history, device)
    
    # Mask out impossible moves by setting them to -inf
    masked_preds = preds.clone()
    for i in range(len(masked_preds)):
        if i not in possible_moves:
            masked_preds[i] = float('-inf')
    
    # Get the move with highest probability among possible moves
    best_move = masked_preds.argmax().item()
    
    board.make_move_inplace(cast(int, best_move), player)
    return int(best_move)



def optimal_nn_move(board: Board, model: nn.Module, player: int) -> int:
    preds = get_nn_preds(board, model, player)
    best_move = preds.idxmax()
    board.make_move_inplace(cast(int, best_move), player)
    return int(best_move)


def optimal_nn_move_noise(
    board: Board, player: int, model: nn.Module, std_noise: int = 0
) -> int:
    preds = get_nn_preds(board, model, player)
    preds = preds + np.random.normal(0, std_noise, len(preds))
    best_move = preds.idxmax()
    board.make_move_inplace(best_move, player)
    return best_move


def random_move(board: Board, player: int, use_2ply_check: bool) -> int:
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
    return rand_move

class TransformerPlayer:
    def __init__(self, model, move_function=optimal_transformer_move):
        self.model = model
        self.move_function = optimal_transformer_move
    
    def make_move(self, board, player, move_history):
        self.move_function(board, self.model, move_history, player)

class NNPlayer:
    random_player_plus = RandomPlayer(random_move, True)

    def __init__(self, move_function, model, noise, transformer = None):
        RandomPlayer(random_move, True)
        self.move_function = move_function
        self.noise = noise
        self.model = model
        self.transformer : Transformer | None = transformer
        self.games = 0
        self.states = []
        self.winners = []
        self.winner_eval = []
        self.move_history = []

    def make_move(self, board, player, move_history=None):
        return self.move_function(board, player, self.model, self.noise)

    def simulate_random_games(self, n=500):
        from game import Game
        for _ in tqdm(range(n)):
            game = Game(self.random_player_plus, self.random_player_plus)
            states, winners, winner, move_history = game.simulate()
            self.states.append(states)
            self.winners.append(winners)
            # Store padded move history for transformer training
            self.move_history.append(game.get_padded_move_history())

    def simulate_noisy_game(self, n=100, noise_level=0.2):
        from game import Game
        for _ in tqdm(range(n)):
            game = Game(
                NNPlayer(optimal_nn_move_noise, self.model, noise_level),
                NNPlayer(optimal_nn_move_noise, self.model, noise_level),
            )
            states, winners, winner, move_history = game.simulate()
            self.states.append(states)
            self.winners.append(winners)
            # Store padded move history for transformer training
            self.move_history.append(game.get_padded_move_history())

        self.games += n

        print(f"Model has been seen {self.games} self-play games")

    def eval_model_battle(self, opp, n=30, first=True):
        from game import Game
        results = []
        for _ in tqdm(range(n)):
            game = Game(self, opp) if first else Game(opp, self)
            states, winners, winner, _ = game.simulate()
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

    def train_transformer_model(self, ntrain=1000, last_n_games=15000):

        if not self.transformer:
            raise ValueError("Transformer needs to be defined for training!")

        seq_len = 41

        move_hist = self.move_history[-last_n_games:]

        moves_away = [np.arange(i.shape[0], 0, -1) for i in self.states[-last_n_games:]]

        # we will use these as reward in the loss function:
        # weight reward close to the end point of the game more:
        winners  = self.winners[-last_n_games:]

        winners_new = []
        for i in range(len(winners)):
            sample_weights = 1 / moves_away[i]

            # this makes the reward always positive -- we are ust doing weighted log likelihood:
            curr_winners = winners[i].copy()#np.abs(winners[i].copy())

            winners_new.append(curr_winners*sample_weights)    

        rewards = [winner[1:] for winner in winners_new]


        padded_rewards = []
        for reward in rewards:
            padding_needed = seq_len - len(reward)
            padded_reward = np.pad(reward, (0, padding_needed), constant_values=0)
            padded_rewards.append(padded_reward)

        rewards = padded_rewards

        move_hist = np.array(move_hist)
        data = move_hist

        X = data[:, :-1]  # All columns except last
        y = data[:, 1:]   # All columns except first

        choices = np.random.choice(np.arange(X.shape[0]), ntrain)

        # Convert to tensors
        X_tensor = torch.LongTensor(X[choices])
        y_tensor = torch.LongTensor(y[choices])

        rewards_tensor = torch.Tensor(np.vstack(rewards)[choices])
        pos_enc = torch.arange(seq_len).unsqueeze(0).repeat(X.shape[0], 1)[choices]
    
        # Create dataset with position encodings
        dataset = TensorDataset(X_tensor, pos_enc, rewards_tensor, y_tensor)
        
        # Create dataloader with shuffling
        data_loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

        for batch_idx, batch in enumerate(data_loader):
   
            self.transformer.train()
            loss = self.transformer.train_step_full_precision(batch, device)
            print(f'Transformer loss is {loss}')


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

    def save_move_history(self, filename: str = None) -> str:
        """Save move history data for transformer training."""
        import json
        from datetime import datetime
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"move_history_{timestamp}.json"
        
        # Pad winners to length 41 (similar to train_transformer_model)
        seq_len = 41
        padded_winners = []
        for winner in self.winners:
            # Skip first element (like in train_transformer_model: winner[1:])
            reward = winner[1:] if len(winner) > 1 else winner
            padding_needed = seq_len - len(reward)
            padded_reward = np.pad(reward, (0, padding_needed), constant_values=1)
            padded_winners.append(padded_reward.tolist())
        
        data = {
            "move_history": self.move_history,
            "winners": padded_winners
        }
        
        with open(filename, "w") as f:
            json.dump(data, f)
        
        return filename
