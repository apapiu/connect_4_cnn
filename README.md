# Self-Play Reinforcement Model for Connect 4

A simple CNN that learns Connect-4 through self play.

The model predicts the probability of winning given the current state 
it is in based on previously played games. That's it, no tree search, no policies.

All the training is through self-play, no external data or players are used.

The two key elements of making this work are exploration (driven here by 
adding noise to the predictions) and sampling to eliminate correlation between 
training samples.

### Exploration:

The exploration part is developed by adding noise to the neural net predictions:

    preds = preds + np.random.normal(0, std_noise, len(preds))

That's it. This leads the model to not replay the same games over and over
while also allowing it to play relatively good moves. 

### Sampling:

### Model Architecture:

A simple 3 layer convolutional neural network does the trick here. 

### Evaluation:

A negamax player:
- checks for a winning move and takes it
- blocks the winning move of an opponent
- avoids moves that allow the opponent to win in the next round

So basically the only way to win against a negamax player is 
by forcing a win.

### Setup:

#### Using uv (Recommended)

1. Install uv if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create and activate a virtual environment with dependencies:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv sync
   ```

3. Run the training:
   ```bash
   uv run python nnplayer.py
   ```

#### Alternative Setup (pip)

If you prefer using pip:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python nnplayer.py
```

### Play against the model

### Gradio Setup

### Speeding up the model



![img.png](img.png)

