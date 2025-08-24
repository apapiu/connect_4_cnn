"""
Simple Connect 4 data loader - just load JSON as numpy, input=row, output=row shifted by 1, shuffled.
"""

import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_connect4_data(json_file, shuffle=True):
    """
    Load Connect 4 JSON data as numpy arrays.
    
    Args:
        json_file: Path to JSON file
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader with (input_sequences, pos_encodings, rewards, target_sequences)
    """
    # Load JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract move history and winners from dictionary format
    move_history = np.array(data["move_history"])
    winners = np.array(data["winners"])
    
    # Input = each row, Output = row shifted by 1
    X = move_history[:, :-1]  # All columns except last
    y = move_history[:, 1:]   # All columns except first
    
    print(f"Loaded {len(move_history)} sequences")
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Rewards shape: {winners.shape}")
    
    # Convert to tensors (ensure proper dtype and contiguity)
    X_tensor = torch.from_numpy(X).long().contiguous()
    y_tensor = torch.from_numpy(y).long().contiguous()
    rewards_tensor = torch.from_numpy(winners).long().contiguous()
    
    # Create position encodings  
    seq_len = X.shape[1]
    pos_enc = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(X.shape[0], 1).contiguous()
    
    # Create dataset with position encodings and rewards
    dataset = TensorDataset(X_tensor, pos_enc, rewards_tensor, y_tensor)
    
    # Create dataloader with shuffling
    dataloader = DataLoader(dataset, batch_size=256, shuffle=shuffle, drop_last=True)
    
    return dataloader


if __name__ == "__main__":
    # Test it
    loader = load_connect4_data("move_history_20250824_115249.json")
    
    for i, (x, pos_enc, rewards, y) in enumerate(loader):
        print(f"Batch {i+1}: X shape {x.shape}, pos_enc shape {pos_enc.shape}, rewards shape {rewards.shape}, y shape {y.shape}")
        print(f"Sample X: {x[0]}...")  
        print(f"Sample pos_enc: {pos_enc[0]}...")  # First 10 elements
        print(f"Sample rewards: {rewards[0]}...")  # First 10 elements
        print(f"Sample y: {y[0]}...")  # First 10 elements
        if i >= 2:
            break
