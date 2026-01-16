import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import NN
from tqdm import tqdm
import os

# Configuration
DATASET_PATH = "data/processed/dataset2_1000k.pt"
MODEL_SAVE_NAME = "models/glassfish_v1_2.pth"    
LOAD_MODEL_PATH = "models/glassfish_v1_2.pth"  

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path, map_location="cpu")
        # Removing ghost dimension if present from previous generator issue
        self.inputs = data['inputs'].squeeze(1) if data['inputs'].dim() == 5 else data['inputs']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def train(load_path=None):
    # 1. Device Selection
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training on: {device}")

    # 2. Model Initialization
    model = NN().to(device)
    
    # LOAD EXISTING WEIGHTS IF PATH PROVIDED
    if load_path and os.path.exists(load_path):
        print(f"🧠 Loading existing weights from {load_path}...")
        model.load_state_dict(torch.load(load_path, map_location=device))
        current_lr = 0.0001 # Start with a smaller LR for fine-tuning
    else:
        print("👶 Starting training from scratch...")
        current_lr = 0.001

    # 3. Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=current_lr)
    # Reduces LR by half if loss doesn't improve for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    criterion = nn.MSELoss()
    
    # 4. Data Loader
    dataset = ChessDataset(DATASET_PATH)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 5. Training Loop
    model.train()
    for epoch in range(10):
        running_loss = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        
        for i, (inputs, labels) in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 0:
                loop.set_description(f"Epoch [{epoch+1}/10]")
                loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
            
        avg_loss = running_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} | Avg Loss: {avg_loss:.5f} | LR: {optimizer.param_groups[0]['lr']}")
        
        # Step the scheduler based on average loss
        scheduler.step(avg_loss)
    
    # 6. Save Updated Model
    torch.save(model.state_dict(), MODEL_SAVE_NAME)
    print(f"💾 Evolution complete. Saved as {MODEL_SAVE_NAME}")

if __name__ == "__main__":
    # To start from v1, pass the path here. To start fresh, pass None.
    train(load_path=LOAD_MODEL_PATH)