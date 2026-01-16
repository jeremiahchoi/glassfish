import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import NN
from tqdm import tqdm

DATASET_PATH = "data/processed/dataset1_62M.pt"
MODEL_NAME = "glassfish_v1.pth"

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        # Load the pre-processed tensors
        data = torch.load(data_path, map_location="cpu")
        self.inputs = data['inputs']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def train():
    # 1. Device Selection (Optimized for MacBook Air)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("🐢 Using CPU (Expect slow training)")

    # 2. Model, Optimizer, Loss
    model = NN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 3. Data Loader
    print(f"📦 Loading dataset from {DATASET_PATH}...")
    dataset = ChessDataset(DATASET_PATH)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 4. Training Loop
    print(f"🔥 Starting training for 10 epochs...")
    model.train()
    
    for epoch in range(10):
        running_loss = 0.0
        # Wrap the loader in tqdm for a pro progress bar
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        
        for i, (inputs, labels) in loop:
            # Move data to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Optimization steps
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar description with current loss
            if i % 10 == 0:
                loop.set_description(f"Epoch [{epoch+1}/10]")
                loop.set_postfix(loss=loss.item())
            
        avg_loss = running_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} Complete. Average Loss: {avg_loss:.5f}")
    
    # 5. Save Model
    torch.save(model.state_dict(), MODEL_NAME)
    print(f"💾 Model saved as {MODEL_NAME}")

if __name__ == "__main__":
    train()