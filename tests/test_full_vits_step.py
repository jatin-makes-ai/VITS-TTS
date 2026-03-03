import torch
import sys
import os

# Ensure paths are correct for uv run
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vits.generator import Generator
from models.vits.discriminators import MultiPeriodDiscriminator

def test_full_gan_step():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Full GAN Step Test: {torch.cuda.get_device_name(0)} ---")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 1. Initialize Both Models
    net_g = Generator().to(device)
    net_p = MultiPeriodDiscriminator().to(device)
    
    # 2. Dummy Inputs
    # z: Latent from Encoder [Batch, Dim, Length]
    # y: Ground Truth Audio [Batch, 1, Samples]
    z = torch.randn(1, 192, 64).to(device).requires_grad_(True) # 64 frames ~ 16k samples
    y = torch.randn(1, 1, 16384).to(device) 

    print(f"VRAM after loading G + MPD: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

    try:
        # --- GENERATOR STEP ---
        y_hat = net_g(z) # Generate audio
        # MPD evaluates the generated audio
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = net_p(y, y_hat)
        
        # Simulate Generator Loss (Adversarial + Feature Matching)
        loss_g = 0
        for g in y_d_gs:
            loss_g += torch.mean((1 - g)**2)
        
        print("Running Generator Backward...")
        loss_g.backward(retain_graph=True) # Retain graph because we'll reuse y_hat for Disc
        
        # --- DISCRIMINATOR STEP ---
        # Note: In real training, we'd detach y_hat here, but let's test peak load
        loss_d = 0
        for r, g in zip(y_d_rs, y_d_gs):
            loss_d += torch.mean((1 - r)**2) + torch.mean(g**2)
            
        print("Running Discriminator Backward...")
        loss_d.backward()

        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"\nPEAK VRAM CONSUMED: {peak_mem:.2f} MB")
        
        # Safety check for 4GB (approx 4096MB)
        if peak_mem < 3800:
            print(">>> STATUS: READY FOR TRAINING.")
        else:
            print(">>> STATUS: BORDERLINE. Use AMP and Batch Size 1.")

    except RuntimeError as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_full_gan_step()