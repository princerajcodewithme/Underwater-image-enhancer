import torch
import torch.nn as nn
import torch.nn.functional as F

class SACESN(nn.Module):
    def __init__(self):
        super(SACESN, self).__init__()
        self.red_gain = nn.Parameter(torch.tensor(2.5))
        self.red_bias = nn.Parameter(torch.tensor(0.1))
        
        self.conv_rgb = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.conv_hsi = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.conv_lab = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(48, 12, 1),
            nn.ReLU(),
            nn.Conv2d(12, 48, 1),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Conv2d(48, 3, 3, padding=1)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def extract_structure(self, x):
        gray = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    def forward(self, x):
        x_comp = x.clone()
        
        # Explicit Red Channel Restoration based on Green/Blue attenuation
        r = x_comp[:, 0:1, :, :]
        g = x_comp[:, 1:2, :, :]
        r_compensated = r + 0.6 * (g - r) * (1.0 - r)
        x_comp[:, 0:1, :, :] = torch.clamp(r_compensated * self.red_gain + self.red_bias, 0.0, 1.0)
        
        f_rgb = self.conv_rgb(x_comp)
        f_hsi = self.conv_hsi(x_comp)
        f_lab = self.conv_lab(x_comp)
        
        f_cat = torch.cat([f_rgb, f_hsi, f_lab], dim=1)
        f_att = f_cat * self.se(f_cat)
        f_color = self.fusion_conv(f_att)
        
        S = self.extract_structure(x)
        f_str = f_color * (1.0 + 0.3 * S)
        return torch.clamp(f_str, 0.0, 1.0)

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

class USCENet(nn.Module):
    def __init__(self, latent_dim=64):
        super(USCENet, self).__init__()
        self.sacesn = SACESN()
        self.padain = PAdaIN()
        
        self.enc1 = nn.Sequential(nn.Conv2d(6, 32, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU())
        
        self.fc_mu = nn.Conv2d(128, latent_dim, 1)
        self.fc_logvar = nn.Conv2d(128, latent_dim, 1)
        
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(latent_dim, 64, 4, stride=2, padding=1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(64 + 64, 32, 4, stride=2, padding=1), nn.ReLU())
        self.out_conv = nn.Sequential(nn.Conv2d(32 + 32, 3, 3, padding=1), nn.Sigmoid())

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, raw, candidate=None):
        prior = self.sacesn(raw)
        
        if candidate is not None:
            inp = torch.cat([prior, candidate], dim=1)
        else:
            inp = torch.cat([prior, raw], dim=1)
            
        e1 = self.enc1(inp)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        mu = self.fc_mu(e3)
        logvar = self.fc_logvar(e3)
        
        mu_vec = torch.mean(mu, dim=[2, 3])
        logvar_vec = torch.mean(logvar, dim=[2, 3])
        
        z = self.reparameterize(mu, logvar)
        z = self.padain(z, mu_vec, torch.exp(0.5 * logvar_vec))
        
        d1 = self.dec1(z)
        d2 = self.dec2(torch.cat([d1, e2], 1))
        out = self.out_conv(torch.cat([d2, e1], 1))
        
        final_out = torch.clamp(prior + (out - 0.5) * 0.5, 0.0, 1.0)
        return final_out, prior, mu_vec, logvar_vec