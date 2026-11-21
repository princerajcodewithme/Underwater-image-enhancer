import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from src.dataset import UIEBDataset
from src.models.uscenet_upgraded import USCENetUpgraded
from src.models.uscenet import USCENet

SPECIES_CLASSES = ["Fish / Marine Life", "Coral Reef Structure", "Sea Turtle", "Diver / Submerged Object", "Seafloor / Rock"]

def test_and_compare():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results/upgraded_outputs", exist_ok=True)
    
    dataset = UIEBDataset(raw_dir="data/UIEB/raw")
    
    # 1. Load Upgraded Model (ViT + Smart Edge + Species Head)
    model_upgraded = USCENetUpgraded().to(device)
    if os.path.exists("checkpoints/uscenet_upgraded.pth"):
        model_upgraded.load_state_dict(torch.load("checkpoints/uscenet_upgraded.pth", map_location=device))
        print("[✓] Loaded Upgraded Model Checkpoint")
    else:
        print("[!] Warning: Upgraded checkpoint not found. Running with initialized weights.")
    model_upgraded.eval()

    # 2. Load Baseline Model (Old USCENet Architecture)
    model_baseline = USCENet().to(device)
    if os.path.exists("checkpoints/uscenet_baseline.pth"):
        model_baseline.load_state_dict(torch.load("checkpoints/uscenet_baseline.pth", map_location=device))
        print("[✓] Loaded Baseline Model Checkpoint")
    model_baseline.eval()

    print("\n--- RUNNING COMPARATIVE EVALUATION & SPECIES RECOGNITION ---\n")
    
    # Safely fetch raw directory image filenames to fix AttributeError
    raw_dir = "data/UIEB/raw"
    image_names = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    with torch.no_grad():
        for i in range(min(10, len(dataset))):
            raw, _ = dataset[i]
            filename = image_names[i] if i < len(image_names) else f"sample_{i+1}.png"
            raw_tensor = raw.unsqueeze(0).to(device)
            
            # Run Baseline Model (Old Output)
            baseline_out, _, _, _ = model_baseline(raw_tensor)
            
            # Run Upgraded Model (ViT + Smart Edge + Species Head Output)
            final_out, prior, smart_edge, species_logits = model_upgraded(raw_tensor)
            
            # Species Classification Inference
            probs = torch.softmax(species_logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item() * 100.0
            species_name = SPECIES_CLASSES[pred_idx]
            
            print(f"Image [{i+1}/10]: {filename} | Predicted Species: {species_name} ({confidence:.1f}%)")
            
            # Tensors to PIL Conversions
            raw_img = transforms.ToPILImage()(raw)
            baseline_img = transforms.ToPILImage()(baseline_out.squeeze(0).cpu())
            upgraded_img = transforms.ToPILImage()(final_out.squeeze(0).cpu())
            edge_img = transforms.ToPILImage()(smart_edge.squeeze(0).cpu())
            
            # Overlay Species Label on Top Left of Upgraded Image
            draw = ImageDraw.Draw(upgraded_img)
            draw.rectangle([5, 5, 240, 30], fill=(0, 0, 0, 180))
            draw.text((10, 10), f"{species_name} ({confidence:.0f}%)", fill=(0, 255, 128))
            
            # 4-Panel Visual Comparison Figure
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            axes[0].imshow(raw_img)
            axes[0].set_title(f"Raw Input ({filename})", fontsize=10, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(baseline_img)
            axes[1].set_title("Baseline (Bluish Tint)", fontsize=10, fontweight='bold')
            axes[1].axis('off')
            
            axes[2].imshow(edge_img, cmap='gray')
            axes[2].set_title("Smart AI Edge (No Snow)", fontsize=10, fontweight='bold')
            axes[2].axis('off')

            axes[3].imshow(upgraded_img)
            axes[3].set_title("Upgraded (ViT + Color Balance)", fontsize=10, fontweight='bold')
            axes[3].axis('off')
            
            plt.tight_layout()
            save_path = f"results/upgraded_outputs/compare_{filename}"
            plt.savefig(save_path, bbox_inches='tight', dpi=200)
            plt.close()

    print(f"\n[✓] All 4-Panel Comparison Plots Saved in: results/upgraded_outputs/")

if __name__ == "__main__":
    test_and_compare()