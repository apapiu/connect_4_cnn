"""Configuration settings for Connect 4 training."""

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for Connect 4 training."""
    
    # Training parameters
    n_iter: int = 10
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


# Default configuration instance
config = Config()
