import zstandard as zstd
import json
import io
import chess
import torch
import numpy as np
from tqdm import tqdm
from utils import board_to_19_tensor

def stream_and_generate(input_zst, output_pt, max_samples=350000, skip_samples=300000, min_depth=22):
    inputs, labels = [], []
    count = 0
    skipped = 0
    
    dctx = zstd.ZstdDecompressor()
    print(f"🚀 Initializing stream. Skipping first {skip_samples} and collecting {max_samples} new positions...")
    
    pbar = tqdm(total=max_samples, desc="Collecting New Data")

    try:
        with open(input_zst, 'rb') as fh:
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                
                for line in text_stream:
                    try:
                        data = json.loads(line)
                        if not data.get('evals') or data['evals'][0]['depth'] < min_depth:
                            continue
                        
                        # SKIP LOGIC
                        if skipped < skip_samples:
                            skipped += 1
                            continue
                        
                        if count >= max_samples:
                            break
                        
                        best_pv = data['evals'][0]['pvs'][0]
                        cp, mate = best_pv.get('cp'), best_pv.get('mate')
                        val = np.tanh(cp/400.0) if cp is not None else (1.0 if mate > 0 else -1.0)

                        board = chess.Board(data['fen'])
                        inputs.append(board_to_19_tensor(board))
                        labels.append(torch.tensor([val], dtype=torch.float32))
                        
                        count += 1
                        pbar.update(1)
                            
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    except (zstd.ZstdError, EOFError):
        print("\n📥 End of file reached.")
    finally:
        pbar.close()

    if count > 0:
        print(f"💾 Saving {count} NEW positions to {output_pt}...")
        torch.save({'inputs': torch.stack(inputs), 'labels': torch.stack(labels)}, output_pt)
        print("✅ Done!")

if __name__ == "__main__":
    stream_and_generate(
        input_zst="data/raw/lichess_evals_425M.jsonl.zst", 
        output_pt="data/processed/dataset2.pt", 
        max_samples=1000000, 
        skip_samples=300000
    )