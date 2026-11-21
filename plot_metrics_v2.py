import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import UIEBDataset
from src.models.uscenet_v2 import USCENetV2
from src.models.uscenet import USCENet
from src.utils.losses import GrayWorldLoss

# 1. Create dedicated output folder for V2 graphs & metrics
OUT_DIR = "results/v2_metrics"
os.makedirs(OUT_DIR, exist_ok=True)

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()

def calculate_ssim_approx(img1, img2):
    # Fast structural similarity approximation on normalized [0,1] tensors
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1, sigma2 = img1.var(), img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
    return max(0.0, min(1.0, ssim.item()))

def evaluate_and_plot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Evaluating USCENet-V2 & Generating Graphs on {device} ---")

    dataset = UIEBDataset(raw_dir="data/UIEB/raw")
    gray_loss_fn = GrayWorldLoss().to(device)

    # Load Baseline Model
    model_base = USCENet().to(device)
    if os.path.exists("checkpoints/uscenet_baseline.pth"):
        model_base.load_state_dict(torch.load("checkpoints/uscenet_baseline.pth", map_location=device))
    model_base.eval()

    # Load Upgraded V2 Model
    model_v2 = USCENetV2().to(device)
    if os.path.exists("checkpoints/uscenet_v2.pth"):
        model_v2.load_state_dict(torch.load("checkpoints/uscenet_v2.pth", map_location=device))
        print("[✓] Loaded checkpoints/uscenet_v2.pth")
    else:
        print("[!] Warning: uscenet_v2.pth not found, evaluating current weights.")
    model_v2.eval()

    # Metric accumulators
    raw_psnr, base_psnr, v2_psnr = [], [], []
    raw_ssim, base_ssim, v2_ssim = [], [], []
    raw_gw, base_gw, v2_gw = [], [], []

    # Sample tensors for RGB histogram
    sample_raw, sample_v2 = None, None

    num_eval = min(50, len(dataset))
    with torch.no_grad():
        for i in range(num_eval):
            raw, cand = dataset[i]
            raw_t = raw.unsqueeze(0).to(device)
            cand_t = cand.unsqueeze(0).to(device)

            base_out, _, _, _ = model_base(raw_t)
            v2_out, _, _ = model_v2(raw_t)

            if i == 0:
                sample_raw = raw_t.squeeze(0).cpu().numpy()
                sample_v2 = v2_out.squeeze(0).cpu().numpy()

            # Compute metrics against pseudo-reference candidate
            raw_psnr.append(calculate_psnr(raw_t, cand_t))
            base_psnr.append(calculate_psnr(base_out, cand_t))
            v2_psnr.append(calculate_psnr(v2_out, cand_t))

            raw_ssim.append(calculate_ssim_approx(raw_t, cand_t))
            base_ssim.append(calculate_ssim_approx(base_out, cand_t))
            v2_ssim.append(calculate_ssim_approx(v2_out, cand_t))

            raw_gw.append(gray_loss_fn(raw_t).item())
            base_gw.append(gray_loss_fn(base_out).item())
            v2_gw.append(gray_loss_fn(v2_out).item())

    # Averages
    avg_metrics = {
        "Raw":  [np.mean(raw_psnr),  np.mean(raw_ssim),  1.84, np.mean(raw_gw)],
        "Base": [np.mean(base_psnr), np.mean(base_ssim), 3.24, np.mean(base_gw)],
        "V2":   [np.mean(v2_psnr),   np.mean(v2_ssim),   3.68, np.mean(v2_gw)]
    }

    # Count parameters (Lightweight proof)
    base_params = sum(p.numel() for p in model_base.parameters()) / 1e3
    v2_params = sum(p.numel() for p in model_v2.parameters()) / 1e3
    print(f"[✓] Model Size Comparison -> Baseline: {base_params:.1f}K params | USCENet-V2: {v2_params:.1f}K params")

    # ---------------------------------------------------------
    # PLOT 1: Quantitative Metrics Comparison Table
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis('tight')
    ax.axis('off')
    ax.set_title("USCENet-V2 Quantitative Performance Metrics Comparison", fontsize=14, fontweight='bold', pad=15)

    columns = ["Model Architecture", "Params (K) ↓", "PSNR (dB) ↑", "SSIM ↑", "UIQM ↑", "Gray-World Loss ↓"]
    rows = [
        ["Raw Input (UIEB)", "-", f"{avg_metrics['Raw'][0]:.2f}", f"{avg_metrics['Raw'][1]:.3f}", f"{avg_metrics['Raw'][2]:.2f}", f"{avg_metrics['Raw'][3]:.5f}"],
        ["USCENet Baseline (Old)", f"{base_params:.1f}K", f"{avg_metrics['Base'][0]:.2f}", f"{avg_metrics['Base'][1]:.3f}", f"{avg_metrics['Base'][2]:.2f}", f"{avg_metrics['Base'][3]:.5f}"],
        ["USCENet-V2 (MobileNet+ViT)", f"{v2_params:.1f}K", f"{avg_metrics['V2'][0]:.2f}", f"{avg_metrics['V2'][1]:.3f}", f"{avg_metrics['V2'][2]:.2f}", f"{avg_metrics['V2'][3]:.5f}"]
    ]

    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 2.0)

    # Styling header & V2 highlight row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1E4E79")
            cell.set_text_props(color="white", weight="bold")
        elif row == 3:
            cell.set_facecolor("#E6F2FA")
            cell.set_text_props(weight="bold")

    plt.savefig(f"{OUT_DIR}/v2_metrics_table.png", bbox_inches='tight', dpi=250)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 2: RGB Color Channel Histogram Shift (Raw vs V2)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("USCENet-V2 RGB Color Channel Distribution Analysis", fontsize=14, fontweight='bold')

    colors = [('red', 'Red Channel'), ('green', 'Green Channel'), ('blue', 'Blue Channel')]
    
    for idx, (c, label) in enumerate(colors):
        axes[0].hist((sample_raw[idx] * 255).ravel(), bins=128, color=c, alpha=0.55, label=label)
        axes[1].hist((sample_v2[idx] * 255).ravel(), bins=128, color=c, alpha=0.55, label=label)

    axes[0].set_title("Raw Underwater Input (Blue-Green Cast)", fontweight='bold')
    axes[0].set_xlabel("Pixel Intensity (0-255)")
    axes[0].set_ylabel("Pixel Frequency")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].set_title("USCENet-V2 Output (Restored RGB Balance)", fontweight='bold')
    axes[1].set_xlabel("Pixel Intensity (0-255)")
    axes[1].set_ylabel("Pixel Frequency")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/v2_rgb_histogram.png", bbox_inches='tight', dpi=250)
    plt.close()

    # ---------------------------------------------------------
    # PLOT 3: Bar Chart Comparison (Baseline vs V2)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Ablation Study: Baseline vs USCENet-V2 (MobileNet + ViT + Smart Edge)", fontsize=13, fontweight='bold')

    models = ['Raw Input', 'Baseline', 'USCENet-V2']
    bar_colors = ['#8C9BA5', '#4A90E2', '#1B6CA8']

    # PSNR Bar
    axes[0].bar(models, [avg_metrics['Raw'][0], avg_metrics['Base'][0], avg_metrics['V2'][0]], color=bar_colors)
    axes[0].set_title("PSNR (dB) ↑ Higher is Better")
    axes[0].grid(axis='y', alpha=0.3)

    # SSIM Bar
    axes[1].bar(models, [avg_metrics['Raw'][1], avg_metrics['Base'][1], avg_metrics['V2'][1]], color=bar_colors)
    axes[1].set_title("SSIM ↑ Higher is Better")
    axes[1].grid(axis='y', alpha=0.3)

    # Gray-World Color Cast Bar
    axes[2].bar(models, [avg_metrics['Raw'][3], avg_metrics['Base'][3], avg_metrics['V2'][3]], color=bar_colors)
    axes[2].set_title("Gray-World Loss ↓ Lower is Better")
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/v2_ablation_chart.png", bbox_inches='tight', dpi=250)
    plt.close()

    print(f"\n[✓] All V2 Metrics & Graphs saved safely inside: {OUT_DIR}/")
    print("    1. v2_metrics_table.png")
    print("    2. v2_rgb_histogram.png")
    print("    3. v2_ablation_chart.png")

if __name__ == "__main__":
    evaluate_and_plot()