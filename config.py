"""Configuration settings for Connect 4 training."""

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for Connect 4 training."""
    
    # Training parameters
    n_iter: int = 40
    warm_start: bool = False
    n_games: int = 1000
    eval_games: int = 50
    last_n_games: int = 50000
    save_every_n_games: int = 10000
    
    # Model parameters
    learning_rate: float = 0.003
    
    # Noise parameters for exploration
    noise_initial: float = 0.2
    noise_decay: float = 0.99
    noise_min: float = 0.15
    
    # Experiment settings
    experiment_name: str = "simple_15k_pytorch"
    use_wandb: bool = False
    
    @property
    def ntrain(self) -> int:
        """Derived parameter: number of training steps."""
        return self.n_games * 50


@dataclass
class TransformerConfig:
    """Configuration for Transformer model."""
    
    # Transformer hyperparameters
    vocab_size: int = 12  # Connect4 columns: 0-6, plus padding tokens 10,11
    embed_dim: int = 128  # Smaller for Connect4
    seq_length: int = 41  # JSON sequences are 42 long, so input is 41
    n_heads: int = 4  # Multi-head attention
    attention_layers: int = 6  # Transformer layers
    dropout: float = 0.3  # Regularization
    mlp_multiplier: int = 2  # MLP expansion factor
    lr: float = 3e-4  # Learning rate for transformer
    epsilon: float = 1e-7  # Adam epsilon
    batch_size: int = 256  # Batch size


# Default configuration instances
config = Config()
transformer_config = TransformerConfig()
