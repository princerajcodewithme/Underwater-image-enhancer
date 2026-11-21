import os
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from src.models.uscenet import USCENet
from src.utils.losses import GrayWorldLoss

# Basic PSNR Calculation
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return 100.0
    return 20.0 * np.log10(1.0 / np.sqrt(mse))

def test_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing baseline model on device: {device}")

    # Load Baseline Model Weights
    model = USCENet().to(device)
    checkpoint_path = "checkpoints/uscenet_baseline.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found. Please train the baseline model first.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    gray_loss_fn = GrayWorldLoss().to(device)

    # Setup Paths
    raw_images_dir = "data/UIEB/raw"
    output_results_dir = "results/baseline_outputs"
    os.makedirs(output_results_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(raw_images_dir, "*.png")) + glob.glob(os.path.join(raw_images_dir, "*.jpg")))[:10]  # First 10 images for quick test

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    to_pil = transforms.ToPILImage()

    print("\n--- BASELINE MODEL RESULTS & METRICS ---")
    psnr_scores = []
    gray_losses = []

    with torch.no_grad():
        for idx, path in enumerate(img_paths):
            raw_pil = Image.open(path).convert('RGB')
            raw_tensor = transform(raw_pil).unsqueeze(0).to(device)

            # Inference (No candidate needed during testing)
            enhanced_tensor, prior_tensor, _, _ = model(raw_tensor)

            # Calculate metrics relative to input/prior
            psnr = calculate_psnr(raw_tensor, enhanced_tensor)
            gw_loss = gray_loss_fn(enhanced_tensor).item()

            psnr_scores.append(psnr)
            gray_losses.append(gw_loss)

            # Save Visual Comparisons (Input vs Prior vs Enhanced Output)
            raw_out = to_pil(raw_tensor.squeeze(0).cpu())
            prior_out = to_pil(prior_tensor.squeeze(0).cpu())
            enh_out = to_pil(enhanced_tensor.squeeze(0).cpu())

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(raw_out)
            axes[0].set_title("Input (Raw Underwater)")
            axes[0].axis("off")

            axes[1].imshow(prior_out)
            axes[1].set_title("SACESN Prior (Red+Structure)")
            axes[1].axis("off")

            axes[2].imshow(enh_out)
            axes[2].set_title("USCENet Final Output")
            axes[2].axis("off")

            save_path = os.path.join(output_results_dir, f"sample_{idx+1}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

            print(f"Image [{idx+1}/{len(img_paths)}]: {os.path.basename(path)} | PSNR: {psnr:.2f} dB | Gray-World Color Cast Loss: {gw_loss:.5f}")

    print("\n--- FINAL AVERAGE PERFORMANCE SUMMARY ---")
    print(f"Average PSNR (Relative Reconstruction): {np.mean(psnr_scores):.2f} dB")
    print(f"Average Color Cast Metric (Gray-World Loss): {np.mean(gray_losses):.5f}")
    print(f"Saved visual comparisons in: {output_results_dir}/")

if __name__ == "__main__":
    test_and_evaluate()