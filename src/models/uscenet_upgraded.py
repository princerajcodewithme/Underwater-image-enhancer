import torch
import torch.nn as nn
import torch.nn.functional as F

class SmartEdgeDetector(nn.Module):
    def __init__(self):
        super(SmartEdgeDetector, self).__init__()
        # Low-pass Gaussian blur to suppress micro-particulate waste noise
        gaussian = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 16.0
        # Macro Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('gaussian', gaussian)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x):
        gray = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
        # Step 1: Filter out particle noise
        denoised_gray = F.conv2d(gray, self.gaussian, padding=1)
        # Step 2: Extract sharp species boundaries
        grad_x = F.conv2d(denoised_gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(denoised_gray, self.sobel_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

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
        attn_out, attn_weights = self.mha(norm_x, norm_x, norm_x)
        x_res = flat_x + attn_out
        x_out = x_res + self.ffn(self.norm(x_res))
        return x_out.transpose(1, 2).reshape(B, C, H, W), attn_weights

class UpgradedSACESN(nn.Module):
    def __init__(self):
        super(UpgradedSACESN, self).__init__()
        self.red_gain = nn.Parameter(torch.tensor(3.2))
        self.red_bias = nn.Parameter(torch.tensor(0.12))
        self.smart_edge = SmartEdgeDetector()
        
        self.conv_rgb = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.conv_hsi = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.conv_lab = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(48, 12, 1), nn.ReLU(),
            nn.Conv2d(12, 48, 1), nn.Sigmoid()
        )
        self.fusion_conv = nn.Conv2d(48, 3, 3, padding=1)

    def forward(self, x):
        x_comp = x.clone()
        r, g, b = x_comp[:, 0:1, :, :], x_comp[:, 1:2, :, :], x_comp[:, 2:3, :, :]
        
        # Strong Red Boost & Blue/Green Attenuation (Fixing Bluish Tint)
        r_boosted = r + 0.8 * (g - r) * (1.0 - r)
        x_comp[:, 0:1, :, :] = torch.clamp(r_boosted * self.red_gain + self.red_bias, 0.0, 1.0)
        x_comp[:, 1:2, :, :] = torch.clamp(g * 0.75, 0.0, 1.0) # Reduce Green
        x_comp[:, 2:3, :, :] = torch.clamp(b * 0.70, 0.0, 1.0) # Reduce Blue
        
        f_rgb = self.conv_rgb(x_comp)
        f_hsi = self.conv_hsi(x_comp)
        f_lab = self.conv_lab(x_comp)
        
        f_cat = torch.cat([f_rgb, f_hsi, f_lab], dim=1)
        f_att = f_cat * self.se(f_cat)
        f_color = self.fusion_conv(f_att)
        
        S = self.smart_edge(x)
        f_str = f_color * (1.0 + 0.4 * S)
        return torch.clamp(f_str, 0.0, 1.0), S

class USCENetUpgraded(nn.Module):
    def __init__(self, latent_dim=64, num_classes=5):
        super(USCENetUpgraded, self).__init__()
        self.sacesn = UpgradedSACESN()
        
        self.enc1 = nn.Sequential(nn.Conv2d(6, 32, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU())
        
        self.vit = ViTGlobalAttention(128)
        
        self.fc_mu = nn.Conv2d(128, latent_dim, 1)
        self.fc_logvar = nn.Conv2d(128, latent_dim, 1)
        
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(latent_dim, 64, 4, stride=2, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(64 + 64, 32, 4, stride=2, padding=1), nn.ReLU())
        self.out_conv = nn.Sequential(nn.Conv2d(32 + 32, 3, 3, padding=1), nn.Sigmoid())
        
        # Species Classifier Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, raw, candidate=None):
        prior, smart_edge = self.sacesn(raw)
        inp = torch.cat([prior, candidate], dim=1) if candidate is not None else torch.cat([prior, raw], dim=1)
        
        e1 = self.enc1(inp)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        e3_vit, attn_map = self.vit(e3)
        
        mu, logvar = self.fc_mu(e3_vit), self.fc_logvar(e3_vit)
        z = self.reparameterize(mu, logvar)
        
        d1 = self.dec1(z)
        d2 = self.dec2(torch.cat([d1, e2], 1))
        out = self.out_conv(torch.cat([d2, e1], 1))
        
        final_out = torch.clamp(prior + (out - 0.5) * 0.4, 0.0, 1.0)
        species_logits = self.classifier(e3_vit)
        
        return final_out, prior, smart_edge, species_logits