import torch
from models.vits.posterior_encoder import PosteriorEncoder

def test_posterior():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # VITS Standard Hyperparameters
    in_channels = 513      # Standard for 1024-point FFT spectrograms
    out_channels = 192     # Latent dimension (must match Text Encoder)
    hidden_channels = 192
    kernel_size = 5
    dilation_rate = 1
    n_layers = 16          # Posterior encoders are typically deeper

    # 1. Initialize Encoder
    encoder = PosteriorEncoder(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        n_layers=n_layers
    ).to(device)

    # 2. Create Dummy Spectrogram Data
    # Batch of 4, 513 frequency bins, 200 time steps (~2.3 seconds of audio)
    batch_size = 4
    spec_channels = 513
    time_steps = 200
    
    x = torch.randn(batch_size, spec_channels, time_steps).to(device)
    x_mask = torch.ones(batch_size, 1, time_steps).to(device)

    print("🚀 Running Posterior Forward Pass...")
    
    # 3. Forward Pass
    with torch.no_grad():
        m, logs = encoder(x, x_mask)

    # 4. Verify Shapes
    print(f"Input Spectrogram Shape: {x.shape}") # [4, 513, 200]
    print(f"Mean (m) Shape: {m.shape}")          # [4, 192, 200]
    print(f"Logs Shape: {logs.shape}")            # [4, 192, 200]
    
    # 5. Check Memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e6
        print(f"VRAM Allocated: {allocated:.2f} MB")

    print("✅ Posterior Encoder test passed!")

if __name__ == "__main__":
    test_posterior()