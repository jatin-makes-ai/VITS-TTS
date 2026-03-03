import torch
from models.vits.text_encoder import TextEncoder

def test_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Encoder with VITS defaults
    # out_channels=192 is the standard latent dimension for VITS
    encoder = TextEncoder(
        out_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6
    ).to(device)

    # 2. Create Dummy Data
    # Batch of 4, Max sequence length of 50 phonemes
    batch_size = 4
    seq_len = 50
    x = torch.randint(0, 100, (batch_size, seq_len)).to(device)
    x_lengths = torch.LongTensor([50, 40, 30, 20]).to(device) # Variable lengths

    print("🚀 Running Forward Pass...")
    
    # 3. Forward Pass
    with torch.no_grad():
        x_out, m, logs, mask = encoder(x, x_lengths)

    # 4. Verify Shapes
    print(f"Input Shape: {x.shape}")
    print(f"Output Hidden Shape: {x_out.shape}")  # Expect [4, 192, 50]
    print(f"Mean (m) Shape: {m.shape}")           # Expect [4, 192, 50]
    print(f"Logs Shape: {logs.shape}")             # Expect [4, 192, 50]
    
    # 5. Check Memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        print(f"Memory Allocated: {allocated:.2f} MB")
        print(f"Memory Reserved: {reserved:.2f} MB")

    print("✅ Text Encoder test passed!")

if __name__ == "__main__":
    test_encoder()