import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.dataset import UIEBDataset
from src.models.uscenet_v2 import USCENetV2
from src.models.uscenet import USCENet

def test_and_compare():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results/v2_outputs", exist_ok=True)
    
    dataset = UIEBDataset(raw_dir="data/UIEB/raw")
    
    # 1. Load V2 Model
    model_v2 = USCENetV2().to(device)
    if os.path.exists("checkpoints/uscenet_v2.pth"):
        model_v2.load_state_dict(torch.load("checkpoints/uscenet_v2.pth", map_location=device))
        print("[✓] Loaded USCENet-V2 Model Checkpoint")
    else:
        print("[!] Running with initialized parameters...")
    model_v2.eval()

    # 2. Load Baseline Model
    model_baseline = USCENet().to(device)
    if os.path.exists("checkpoints/uscenet_baseline.pth"):
        model_baseline.load_state_dict(torch.load("checkpoints/uscenet_baseline.pth", map_location=device))
        print("[✓] Loaded Baseline Model Checkpoint")
    model_baseline.eval()

    raw_dir = "data/UIEB/raw"
    image_names = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print("\n--- RUNNING HIGH-QUALITY EVALUATION & VISUAL COMPARISON ---\n")

    with torch.no_grad():
        for i in range(min(10, len(dataset))):
            raw, _ = dataset[i]
            filename = image_names[i] if i < len(image_names) else f"sample_{i+1}.png"
            raw_tensor = raw.unsqueeze(0).to(device)
            
            # Baseline output
            baseline_out, _, _, _ = model_baseline(raw_tensor)
            
            # Upgraded V2 output (No Detection Head)
            final_out, prior, smart_edge = model_v2(raw_tensor)
            
            raw_img = transforms.ToPILImage()(raw)
            baseline_img = transforms.ToPILImage()(baseline_out.squeeze(0).cpu())
            v2_img = transforms.ToPILImage()(final_out.squeeze(0).cpu())
            edge_img = transforms.ToPILImage()(smart_edge.squeeze(0).cpu())
            
            # 4-Panel Side-by-Side Comparison
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            axes[0].imshow(raw_img)
            axes[0].set_title(f"Raw Input ({filename})", fontsize=10, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(baseline_img)
            axes[1].set_title("Baseline (Bluish Tint)", fontsize=10, fontweight='bold')
            axes[1].axis('off')
            
            axes[2].imshow(edge_img, cmap='gray')
            axes[2].set_title("Smart Edge (Filtered Noise)", fontsize=10, fontweight='bold')
            axes[2].axis('off')

            axes[3].imshow(v2_img)
            axes[3].set_title("USCENet-V2 (High Quality)", fontsize=10, fontweight='bold')
            axes[3].axis('off')
            
            plt.tight_layout()
            save_path = f"results/v2_outputs/compare_{filename}"
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            plt.close()
            print(f"Processed [{i+1}/10]: {filename} -> Saved in results/v2_outputs/")

    print(f"\n[✓] All High-Quality Outputs Saved in: results/v2_outputs/")

if __name__ == "__main__":
    test_and_compare()