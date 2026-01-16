import zstandard as zstd
import json
import io
import chess
import torch
import numpy as np
from utils import board_to_19_tensor

def stream_and_generate(input_zst, output_pt, max_samples=300000, min_depth=22):
    """
    Processes the partial .zst file. 
    Handles the abrupt end (EOF) gracefully.
    """
    inputs = []
    labels = []
    count = 0
    
    dctx = zstd.ZstdDecompressor()
    
    print(f"Reading from {input_zst}...")
    
    try:
        with open(input_zst, 'rb') as fh:
            # stream_reader is vital for partial files
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                
                for line in text_stream:
                    if count >= max_samples:
                        break
                    
                    try:
                        data = json.loads(line)
                        
                        # 1. Quality Filter: Only Depth 22+
                        # We take the first evaluation (usually the highest depth)
                        best_eval = data['evals'][0]
                        if best_eval['depth'] < min_depth:
                            continue
                        
                        # 2. Extract Label (Centipawns or Mate)
                        pv = best_eval['pvs'][0]
                        cp = pv.get('cp')
                        mate = pv.get('mate')
                        
                        if cp is not None:
                            # Normalize: Tanh(cp/400) provides a smooth [-1, 1] gradient
                            val = np.tanh(cp / 400.0)
                        elif mate is not None:
                            val = 1.0 if mate > 0 else -1.0
                        else:
                            continue

                        # 3. Convert FEN to 19-Channel Tensor
                        board = chess.Board(data['fen'])
                        tensor = board_to_19_tensor(board) # Returns [19, 8, 8]
                        
                        inputs.append(tensor)
                        labels.append(torch.tensor([val], dtype=torch.float32))
                        
                        count += 1
                        if count % 10000 == 0:
                            print(f"Processed {count} positions...")
                            
                    except (json.JSONDecodeError, KeyError, IndexError):
                        # This happens at the very last line of a partial download
                        continue

    except (zstd.ZstdError, EOFError):
        print("\nReached the end of the partial download successfully.")

    if count > 0:
        print(f"Saving {count} positions to {output_pt}...")
        # Save as a dictionary for the DataLoader
        torch.save({
            'inputs': torch.stack(inputs), 
            'labels': torch.stack(labels)
        }, output_pt)
        print("Done!")
    else:
        print("Error: No valid positions found. Check your min_depth or file path.")

if __name__ == "__main__":
    # Data downloaded with:
    # curl -o lichess_evals_partial.jsonl.zst https://database.lichess.org/lichess_db_eval.jsonl.zst
    stream_and_generate("data/raw/lichess_evals_partial.jsonl.zst", "data/processed/dataset1_62M.pt")