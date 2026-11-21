import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Depthwise Separable Conv (MobileNet-style Lightweight Block)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x

# 2. Smart AI Edge Detector (Filters particle/marine snow noise)
class SmartEdgeDetector(nn.Module):
    def __init__(self):
        super(SmartEdgeDetector, self).__init__()
        gaussian = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 16.0
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('gaussian', gaussian)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x):
        gray = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
        denoised_gray = F.conv2d(gray, self.gaussian, padding=1)
        grad_x = F.conv2d(denoised_gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(denoised_gray, self.sobel_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

# 3. Vision Transformer Block (Global Context & Lighting Balance)
class ViTGlobalAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(ViTGlobalAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.GELU(),
            nn.Linear(in_channels * 2, in_channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        flat_x = x.flatten(2).transpose(1, 2)
        norm_x = self.norm(flat_x)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)
        x_res = flat_x + attn_out
        x_out = x_res + self.ffn(self.norm(x_res))
        return x_out.transpose(1, 2).reshape(B, C, H, W)

# 4. Lightweight SACESN Sub-Network
class MobileNetSACESN(nn.Module):
    def __init__(self):
        super(MobileNetSACESN, self).__init__()
        self.red_gain = nn.Parameter(torch.tensor(3.5))
        self.red_bias = nn.Parameter(torch.tensor(0.15))
        self.smart_edge = SmartEdgeDetector()
        
        self.conv_rgb = DepthwiseSeparableConv(3, 16)
        self.conv_hsi = DepthwiseSeparableConv(3, 16)
        self.conv_lab = DepthwiseSeparableConv(3, 16)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(48, 12, 1), nn.ReLU(),
            nn.Conv2d(12, 48, 1), nn.Sigmoid()
        )
        self.fusion_conv = nn.Conv2d(48, 3, 3, padding=1)

    def forward(self, x):
        x_comp = x.clone()
        r, g, b = x_comp[:, 0:1, :, :], x_comp[:, 1:2, :, :], x_comp[:, 2:3, :, :]
        
        # Color Balance & Red Gain Optimization
        r_boosted = r + 0.85 * (g - r) * (1.0 - r)
        x_comp[:, 0:1, :, :] = torch.clamp(r_boosted * self.red_gain + self.red_bias, 0.0, 1.0)
        x_comp[:, 1:2, :, :] = torch.clamp(g * 0.70, 0.0, 1.0)
        x_comp[:, 2:3, :, :] = torch.clamp(b * 0.65, 0.0, 1.0)
        
        f_rgb = self.conv_rgb(x_comp)
        f_hsi = self.conv_hsi(x_comp)
        f_lab = self.conv_lab(x_comp)
        
        f_cat = torch.cat([f_rgb, f_hsi, f_lab], dim=1)
        f_att = f_cat * self.se(f_cat)
        f_color = self.fusion_conv(f_att)
        
        S = self.smart_edge(x)
        f_str = f_color * (1.0 + 0.45 * S)
        return torch.clamp(f_str, 0.0, 1.0), S

# 5. PAdaIN Feature Normalization
class PAdaIN(nn.Module):
    def __init__(self):
        super(PAdaIN, self).__init__()

    def forward(self, x, mu_target, sigma_target):
        mu_x = torch.mean(x, dim=[2, 3], keepdim=True)
        sigma_x = torch.std(x, dim=[2, 3], keepdim=True) + 1e-6
        normalized = (x - mu_x) / sigma_x
        scale = sigma_target.unsqueeze(-1).unsqueeze(-1)
        shift = mu_target.unsqueeze(-1).unsqueeze(-1)
        return normalized * (1.0 + 0.1 * scale) + 0.1 * shift

# 6. Final USCENet-V2 Architecture
class USCENetV2(nn.Module):
    def __init__(self, latent_dim=64):
        super(USCENetV2, self).__init__()
        self.sacesn = MobileNetSACESN()
        self.padain = PAdaIN()
        
        self.enc1 = DepthwiseSeparableConv(6, 32)
        self.enc2 = DepthwiseSeparableConv(32, 64, stride=2)
        self.enc3 = DepthwiseSeparableConv(64, 128, stride=2)
        
        self.vit = ViTGlobalAttention(128)
        
        self.fc_mu = nn.Conv2d(128, latent_dim, 1)
        self.fc_logvar = nn.Conv2d(128, latent_dim, 1)
        
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(latent_dim, 64, 4, stride=2, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(64 + 64, 32, 4, stride=2, padding=1), nn.ReLU())
        self.out_conv = nn.Sequential(nn.Conv2d(32 + 32, 3, 3, padding=1), nn.Sigmoid())

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, raw, candidate=None):
        prior, smart_edge = self.sacesn(raw)
        inp = torch.cat([prior, candidate], dim=1) if candidate is not None else torch.cat([prior, raw], dim=1)
        
        e1 = self.enc1(inp)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        e3_vit = self.vit(e3)
        
        mu, logvar = self.fc_mu(e3_vit), self.fc_logvar(e3_vit)
        mu_vec = torch.mean(mu, dim=[2, 3])
        logvar_vec = torch.mean(logvar, dim=[2, 3])
        
        z = self.reparameterize(mu, logvar)
        z = self.padain(z, mu_vec, torch.exp(0.5 * logvar_vec))
        
        d1 = self.dec1(z)
        d2 = self.dec2(torch.cat([d1, e2], 1))
        out = self.out_conv(torch.cat([d2, e1], 1))
        
        final_out = torch.clamp(prior + (out - 0.5) * 0.45, 0.0, 1.0)
        return final_out, prior, smart_edge