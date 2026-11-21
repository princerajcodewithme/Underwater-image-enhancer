import torch
from torch.utils.data import DataLoader
from src.dataset import UIEBDataset
from src.models.uscenet_upgraded import USCENetUpgraded
from src.utils.losses import GrayWorldLoss
import torch.nn.functional as F

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Upgraded USCENet (ViT + Smart Edge + Species Head) on: {device}")
    
    dataset = UIEBDataset(raw_dir="data/UIEB/raw")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = USCENetUpgraded().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    gray_loss_fn = GrayWorldLoss().to(device)
    
    epochs = 15
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for raw, cand in dataloader:
            raw, cand = raw.to(device), cand.to(device)
            optimizer.zero_grad()
            
            out, prior, mu, logvar, _ = model(raw, cand)
            
            l_rec = F.l1_loss(out, cand)
            l_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            l_gray = gray_loss_fn(out)
            
            # Loss weighting: High priority on Color Balance and Structural L1
            loss = 5.0 * l_rec + 0.001 * l_kl + 8.0 * l_gray
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), "checkpoints/uscenet_upgraded.pth")
    print("[✓] Upgraded USCENet Model Saved Successfully!")

if __name__ == "__main__":
    train()