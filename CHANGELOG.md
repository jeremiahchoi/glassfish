### v0.1.0
- **Feature Engineering:** Used a 19-channel input tensor ($19 \times 8 \times 8$) encoding piece positions and game-state metadata (castling, turn, EP).
- **Data Pipeline:** Made a streaming parser using zstandard to perform stochastic sampling from a truncated 18GB Lichess dataset.
- **Preprocessing:** Implemented Tanh Normalization on regression targets ($y = \tanh(cp/400)$) to ensure gradient stability.
- **Architecture:** Constructed a 10-block Residual Network (ResNet) to address the vanishing gradient problem.- **Acceleration:** Integrated MPS (Metal Performance Shaders) for GPU-accelerated training on Apple Silicon.
- **Optimization:** Utilized Adam Optimizer and MSE Loss to approximate the Stockfish evaluation function.

Notes: Need to train first model, and create heatmaps visualizer.