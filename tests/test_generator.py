import torch
import sys
import os

# Add the project root to path so we can import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vits.generator import Generator

def test_generator_performance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Generator Test on: {torch.cuda.get_device_name(0)} ---")
    
    # Reset VRAM stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 1. Initialize Generator
    # Settings match standard VITS (upsampling 256x total)
    model = Generator(
        initial_channels=192, 
        resblock_kernel_sizes=[3, 7, 11],
        upsample_rates=[8, 8, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 4, 4]
    ).to(device)
    
    # 2. Prepare Dummy Input
    # [Batch, Channels, Length] -> 50 mel frames ~ 0.5s of audio
    z = torch.randn(1, 192, 50).to(device).requires_grad_(True)
    
    print(f"Memory after model load: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

    try:
        # 3. Forward Pass
        print("Starting Forward Pass...")
        waveform = model(z)
        print(f"Output Waveform Shape: {waveform.shape}")
        forward_mem = torch.cuda.memory_allocated() / 1e6
        print(f"Memory after Forward: {forward_mem:.2f} MB")

        # 4. Backward Pass (The Stress Test)
        print("Starting Backward Pass simulation...")
        # Create a dummy loss
        loss = waveform.sum()
        loss.backward()
        
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"Peak VRAM during Backprop: {peak_mem:.2f} MB")
        print("\n--- TEST SUCCESSFUL ---")
        
        if peak_mem < 3500:
            print("Status: SUCCESS. You have enough headroom for Batch Size 1.")
        else:
            print("Status: WARNING. VRAM usage is very high. Consider Automatic Mixed Precision (AMP).")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n--- TEST FAILED: OOM ---")
            print("Your GTX 1650 cannot handle this config in FP32. We will need to use AMP (Half Precision).")
        else:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_generator_performance()