### v0.1.2
- Trained the model on 300k positions with LR 0.001 and then again 1M positions with LR 0.0001
- Got great results:
| Session | Learning Rate | Start Loss (Epoch 1) | End Loss (Epoch 10) | Positions |
| ;--- | ;--- | ;--- | ;--- | ;--- |
| v0.1.1 | 0.001 | 0.14990 | 0.04546 | 300k | 
| v0.1.2 | 0.0001 | 0.12538 | 0.05571 | 1.0M |

### v0.1.0
- Used a 19-channel input tensor ($19 \times 8 \times 8$) encoding piece positions and game-state metadata (castling, turn, EP).
- Made a streaming parser using zstandard to perform stochastic sampling from a truncated 18GB Lichess dataset.
- Implemented Tanh Normalization on regression targets ($y = \tanh(cp/400)$) to ensure gradient stability.
- Constructed a 10-block Residual Network (ResNet) to address the vanishing gradient problem.- 
- Integrated MPS (Metal Performance Shaders) for GPU-accelerated training on Apple Silicon.
- Utilized Adam Optimizer and MSE Loss to approximate the Stockfish evaluation function.

Notes: Need to train first model, and create heatmaps visualizer.