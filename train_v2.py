import torch
from torch.utils.data import DataLoader
from src.dataset import UIEBDataset
from src.models.uscenet_v2 import USCENetV2
from src.utils.losses import GrayWorldLoss
import torch.nn.functional as F

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training USCENet-V2 (MobileNet + ViT + Smart Edge) on {device} ---")
    
    dataset = UIEBDataset(raw_dir="data/UIEB/raw")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = USCENetV2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    gray_loss_fn = GrayWorldLoss().to(device)
    
    for epoch in range(15):
        model.train()
        total_loss = 0.0
        for raw, cand in dataloader:
            raw, cand = raw.to(device), cand.to(device)
            optimizer.zero_grad()
            
            out, prior, _ = model(raw, cand)
            
            l_rec = F.l1_loss(out, cand)
            l_gray = gray_loss_fn(out)
            
            # Loss weighting: Enhanced Color Balance & Edge Contrast
            loss = 4.0 * l_rec + 10.0 * l_gray
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/15] Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), "checkpoints/uscenet_v2.pth")
    print("[✓] Saved Model Checkpoint: checkpoints/uscenet_v2.pth")

if __name__ == "__main__":
    train()