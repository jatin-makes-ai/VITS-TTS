import torch
import sys
import os

# Ensure project structure is recognized
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vits.generator import Generator
from models.vits.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator
# Assuming these are in your models/vits/ folder from previous sessions
# from models.vits.encoders import TextEncoder, PosteriorEncoder 

def test_full_vits_limit(batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Stress Testing Full VITS | Batch Size: {batch_size} ---")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 1. Initialize the Full Ensemble
    net_g = Generator().to(device)
    net_mpd = MultiPeriodDiscriminator().to(device)
    net_msd = MultiScaleDiscriminator().to(device)
    
    # 2. Dummy Data (16,384 samples ~ 0.7s at 22kHz)
    z = torch.randn(batch_size, 192, 64).to(device).requires_grad_(True)
    y = torch.randn(batch_size, 1, 16384).to(device)

    try:
        # --- STEP 1: GENERATOR + FLOW FORWARD ---
        y_hat = net_g(z)
        
        # --- STEP 2: DISCRIMINATOR EVALUATION ---
        # MPD
        res_mpd_r, res_mpd_g, _, _ = net_mpd(y, y_hat)
        # MSD
        res_msd_r, res_msd_g, _, _ = net_msd(y, y_hat)
        
        # --- STEP 3: SIMULATED BACKWARD PASS ---
        # Generator Loss (Simplified)
        loss_g = 0
        for g in res_mpd_g + res_msd_g:
            loss_g += torch.mean((1 - g)**2)
        
        print(f"Running Full Generator Backward (BS={batch_size})...")
        loss_g.backward(retain_graph=True)
        
        # Discriminator Loss (Simplified)
        loss_d = 0
        for r, g in zip(res_mpd_r + res_msd_r, res_mpd_g + res_msd_g):
            loss_d += torch.mean((1 - r)**2) + torch.mean(g**2)
            
        print(f"Running Full Discriminator Backward (BS={batch_size})...")
        loss_d.backward()

        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        total_mem = 4290 # Your GTX 1650 Limit
        
        print(f"\nPEAK VRAM: {peak_mem:.2f} MB / {total_mem} MB")
        print(f"UTILIZATION: {(peak_mem/total_mem)*100:.1f}%")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[!] OOM at Batch Size {batch_size}. Try reducing to {batch_size // 2}.")
        else:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_full_vits_limit(batch_size=4)