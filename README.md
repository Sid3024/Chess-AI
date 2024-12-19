# Chess Transformer AI
## Intro
This project implements a Vision Transformer-based chess engine inspired by Google DeepMind's no-search architecture, with enhancements from Leela Chess Zero. The model processes 84 input tokens, representing board states, special rules (castling, en passant), and game phases. Key innovations include the "smolgen" module for board-specific attention and improved tokenization for symmetry exploitation. Trained on 140M high-quality datapoints, the engine delivers strong chess performance with domain-specific optimizations.

## Model
### Architecture
The model architecture extends and slightly modifies the Vision Transformer-based architecture used by Google DeepMind to create the first chess engine without explicit search ([DeepMind paper](https://arxiv.org/pdf/2402.04494)). The transformer architecture has 12 layers, 8 attention heads per layer and an embedding dimension of 256.

It also implements a modified version of the "smolgen" module introduced by Leela Chess Zero transformer (https://lczero.org/blog/2024/02/transformer-progress/), which is a domain specific enhancement. This module is used to address the fact that the interaction between 2 pieces is not just a function of their respective tokens but the entire board state. For example, squares that are far apart should have a weaker signal in a closed position and a stronger signal in an open position. To address this, we pass the CLS token through a linear layer to output a vector with 4096 dimensions, which is reshaped to (64, 64). This acts as a second attention map that is added to the original one before softmaxxing.

### Input Tokens
- **78 tokens** in total:
  - The first token is a **CLS token** for aggregating information from attention blocks.
  - After passing through the transformer, the **CLS token** is processed by a linear layer (**fc1**) to output a vector, which undergoes **softmax** to produce a policy vector across all **1968 UCI moves** (possible moves, not just legal moves).

- **64 tokens** represent the chessboard squares. Each square can be in one of **13 possible states**:
  - 1 for **empty**.
  - 6 for **white pieces**.
  - 6 for **black pieces**.
  
  These states have **learned embeddings**. Additionally, we use **learned absolute positional embeddings**.
- **Next token**: a binary token indicating whose turn it is.

- **Next 4 tokens**: binary tokens representing castling rights.

- **Next 9 tokens**: binary tokens encoding **en passant** rights:
  - The first token indicates whether **en passant** is available.
  - The remaining 8 tokens encode which file the **en passant** square is on.


### Tokenization format
The approach to tokenization in Google deepminds paper for the 64 square tokens was to take them in order from the 8th row (black king's row) to the 1st row (white king's row) for all positions, and a binary turn token to indicate whose turn it is. However Leela chess zero utilised board flipping, where when it was black's turn they listed the square tokens from the 1st row to the 8th, and labelled the pieces as "player" and "enemy" pieces rather than "white" or "black" pieces. This takes advantage of the symmetry of chess by always having the board state input from the perspective of the current player to simplify the task for the model.

We tried both tokenization methods, and Leela Chess Zero's method yielded a performance that was 20% better than that of Google Deepmind.


## Dataset
Dataset was generated from **game shards** on [lichess.com](https://lichess.com). The moves played in each game are considered the **target move**. We only use datapoints from the winning player (or both players in case of a draw) where the **average rating** of the players is greater than 2000, and we only include games with time controls of at least 10 minutes per person to ensure data quality.

The dataset consisted of a total of 140 million datapoints stored in SQL.


## Training
The model was trained for a total of 720k steps (6 epochs). The learning rate was manually decreased every 2 epochs, from 4e-4 to 1e-4 to 3e-5. It was also further decreased by a factor of 10 for the parameters of the output head.













