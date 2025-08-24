import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from simple_loader import load_connect4_data
import pandas as pd
from einops import rearrange
from config import transformer_config


def play_vs(player_1, player_2, n_games=100):
    """Play games between two players and return win statistics."""
    # Local imports to avoid circular dependency
    from game import Game
    
    winner_eval = []

    for i in tqdm(range(n_games)):
        _, _, winner, _ = Game(player_1, player_2).simulate()
        winner_eval.append(winner)

    print(pd.Series(winner_eval).value_counts(normalize=True))
    return pd.Series(winner_eval).value_counts(normalize=True)


scaler = GradScaler()

def get_probabilities(model, input_sequence, device):
    """
    Get probability distribution for next token given input sequence.
    
    Args:
        model: Trained transformer model
        input_sequence: List or tensor of move tokens
        device: Device to run inference on
        
    Returns:
        probabilities: Softmax probabilities over vocabulary
    """
    model.eval()
    
    with torch.no_grad():
        # Convert to tensor if needed
        if not isinstance(input_sequence, torch.Tensor):
            input_sequence = torch.tensor(input_sequence, dtype=torch.long)
        
        # Add batch dimension if needed
        if input_sequence.dim() == 1:
            input_sequence = input_sequence.unsqueeze(0)
        
        # Create position encoding
        seq_len = input_sequence.shape[1]
        pos_enc = torch.arange(seq_len).unsqueeze(0).repeat(input_sequence.shape[0], 1)
        
        # Move to device
        input_sequence = input_sequence.to(device)
        pos_enc = pos_enc.to(device)
        
        # Forward pass
        logits = model.forward(input_sequence, pos_enc)
        
        # Get probabilities for the last token (next move prediction)
        last_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        probabilities = torch.softmax(last_token_logits, dim=-1)
        
        return probabilities.squeeze()  # Remove batch dimension if single sequence

def pytorch_model_summary(model):
    summary = {}
    for name, param in model.named_parameters():
        layer_type = name.split('.')[0]
        param_size = list(param.size())
        num_params = param.numel()
        if layer_type not in summary:
            summary[layer_type] = {'param_size': param_size, 'num_params': num_params}
        else:
            summary[layer_type]['num_params'] += num_params
    return summary

class SequenceGenerator:
    def __init__(self, token_ids, seq_length, batch_size):
        self.token_ids = token_ids
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.n_tokens = len(token_ids)
        self.indices = np.arange(0, self.n_tokens - seq_length)

    def __iter__(self):
        np.random.shuffle(self.indices)
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i+self.batch_size]
            X_batch = np.zeros((len(batch_indices), self.seq_length), dtype=np.int16)
            y_batch = np.zeros((len(batch_indices), self.seq_length), dtype=np.int16)

            for j, idx in enumerate(batch_indices):
                X_batch[j] = self.token_ids[idx:idx+self.seq_length]
                y_batch[j] = self.token_ids[idx+1:idx+self.seq_length+1]

            pos_enc_batch = np.tile(np.arange(0, self.seq_length), (len(batch_indices), 1))
            yield torch.LongTensor(X_batch), torch.LongTensor(pos_enc_batch), torch.LongTensor(y_batch)




################################################################################
# model blocks:
################################################################################


class MHAttention(nn.Module):
    def __init__(self, is_causal=True, dropout_level=0.1, n_heads=4):
        super().__init__()
        self.is_causal = is_causal
        self.dropout_level = dropout_level
        self.n_heads = n_heads

    def forward(self, q, k, v, attn_mask=None):

        assert q.size(-1) == k.size(-1)
        assert k.size(-2) == v.size(-2)

        q, k, v = [rearrange(x, 'bs n (d h) -> bs h n d', h=self.n_heads) for x in [q,k,v]]
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        out = nn.functional.scaled_dot_product_attention(q, k, v,
                                                          attn_mask=attn_mask,
                                                          is_causal=self.is_causal,
                                                          dropout_p=self.dropout_level if self.training else 0)

        out = rearrange(out, 'bs h n d -> bs n (d h)', h=self.n_heads)

        return out

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, mlp_multiplier):
        super(AttentionBlock, self).__init__()
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_multiplier * embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_multiplier * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.attn = MHAttention(is_causal=True, dropout_level=dropout, n_heads=n_heads)

    def forward(self, x):
        # Pre-norm pattern: LayerNorm -> Attention -> Residual
        norm_x = self.ln1(x)
        q = self.q_linear(norm_x)
        k = self.k_linear(norm_x)
        v = self.v_linear(norm_x)
        attn = self.attn(q, k, v)
        x = x + self.dropout(attn)
        
        # Pre-norm pattern: LayerNorm -> MLP -> Residual
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.seq_length, config.embed_dim)
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(config.embed_dim, config.n_heads, config.dropout, config.mlp_multiplier) 
            for _ in range(config.attention_layers)
        ])
        self.fc = nn.Linear(config.embed_dim, config.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=config.lr, eps=config.epsilon)

    def forward(self, x, pos_enc):
        x = self.embedding(x) + self.pos_embedding(pos_enc)
        for block in self.attention_blocks:
            x = block(x)
        x = self.fc(x)
        return x

    def train_step_full_precision(self, batch, device):
        use_reward = True

        x, pos_enc, rewards_tensor, y = batch
        x, pos_enc, rewards_tensor, y = x.to(device), pos_enc.to(device), rewards_tensor.to(device), y.to(device)
        y_pred = self.forward(x, pos_enc)
        
        # Compute negative log likelihood
        log_probs = F.log_softmax(y_pred, dim=-1)
        
        # Flatten tensors
        log_probs_flat = log_probs.view(-1, log_probs.size(-1))  # (batch*seq, vocab_size)
        y_flat = y.view(-1)  # (batch*seq,)
        rewards_flat = rewards_tensor.view(-1)  # (batch*seq,)
        
        # Get log probabilities for the correct actions
        selected_log_probs = log_probs_flat.gather(1, y_flat.unsqueeze(1)).squeeze(1)  # (batch*seq,)
        
        # Clamp log probabilities to prevent -inf (common in RL)
        selected_log_probs = torch.clamp(selected_log_probs, min=-10)  # Prevent log(p) < -10

        rewards_flat = torch.clamp(rewards_flat, min=-0.5)
        
        # Reward-weighted negative log likelihood: -reward * log(p(action))
        if use_reward:
            weighted_loss = -rewards_flat * selected_log_probs
        else:
            weighted_loss = - selected_log_probs
        
        # Take mean over all elements
        loss = weighted_loss.mean()
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()


    def train_step(self, batch, device, scaler):
        x, pos_enc, rewards_tensor, y = batch
        x, pos_enc, rewards_tensor, y = x.to(device), pos_enc.to(device), rewards_tensor.to(device), y.to(device)

        self.optimizer.zero_grad()

        # Using autocast for the forward pass
        with autocast():
            y_pred = self.forward(x, pos_enc)
            
            # Compute reward-weighted negative log likelihood
            log_probs = F.log_softmax(y_pred, dim=-1)
            log_probs_flat = log_probs.view(-1, log_probs.size(-1))
            y_flat = y.view(-1)
            rewards_flat = rewards_tensor.view(-1)
            selected_log_probs = log_probs_flat.gather(1, y_flat.unsqueeze(1)).squeeze(1)
            # Clamp log probabilities to prevent -inf
            selected_log_probs = torch.clamp(selected_log_probs, min=-10)
            weighted_loss = -rewards_flat * selected_log_probs
            loss = weighted_loss.mean()

        # Using GradScaler to scale the loss and perform the backward pass
        scaler.scale(loss).backward()

        # Unscale the gradients and update the weights
        scaler.step(self.optimizer)

        # Update the scale for the next iteration
        scaler.update()

        return loss.item()



if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Connect4 Data - simple loading
    json_file = "move_history_20250824_140307.json"
    train_loader = load_connect4_data(json_file, shuffle=True)

    # Initialize model with transformer config
    model = Transformer(transformer_config)
    summary = pytorch_model_summary(model)
    print(summary)
    model.to(device)

    # Create players for evaluation
    from players import RandomPlayer, TransformerPlayer, random_move
    random_player_reg = RandomPlayer(random_move, False)
    random_player_plus = RandomPlayer(random_move, True)
    transformer_player = TransformerPlayer(model=model)

    # Training loop 
    n_epochs = 2
    batches_per_epoch = len(train_loader)
    
    for epoch in range(n_epochs):
        for batch_idx, batch in enumerate(tqdm(train_loader)):            
            # Train
            loss = model.train_step_full_precision(batch, device)
            
            # Print loss occasionally
            if batch_idx % 50 == 0:
                print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {loss:.4f}")
                print("PLAY VS:")
                play_vs(random_player_plus, transformer_player)
                print("EVAL END")
    
    print("Training completed!")
    
    # Example usage
    print("\n--- Example: Predicting next moves ---")
    example_sequence = [2, 4, 3, 4, 2, 5]  # Example Connect 4 moves
    

# Get full probability distribution
# example_sequence = [0,4,1,4,2,4,0,4,10]
# all_probas = get_probabilities(model, example_sequence, device)
# [f"{p:.4f}" for p in all_probas[:7]]

# print("\nFull probability distribution:")
# for move in range(7):
#     print(f"  Column {move}: {all_probas[move]:.4f}")