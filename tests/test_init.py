import torch
from models.vits.models import SynthesizerTrn

def test_init():
    # Standard settings for your 4GB VRAM
    model = SynthesizerTrn(
        n_vocab=100,
        spec_channels=513,
        segment_size=16384,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[8, 8, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 4, 4],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Synthesizer successfully initialized on", device, "!")


def test_forward():
    """Run one forward pass to verify the full VITS pipeline (text -> alignment -> waveform segment)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SynthesizerTrn(
        n_vocab=100,
        spec_channels=513,
        segment_size=16384,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[8, 8, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 4, 4],
    ).to(device)
    model.eval()
    b, T_text, T_spec = 2, 50, 200
    x = torch.randint(0, 99, (b, T_text), device=device)
    x_lengths = torch.tensor([T_text, 40], device=device, dtype=torch.long)
    y = torch.randn(b, 513, T_spec, device=device)
    y_lengths = torch.tensor([T_spec, 150], device=device, dtype=torch.long)
    with torch.no_grad():
        o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = model(x, x_lengths, y, y_lengths)
    assert o.shape[0] == b and o.shape[1] == 1 and o.shape[2] == 16384, (o.shape,)
    print("Forward pass OK. Output waveform segment shape:", o.shape)


if __name__ == "__main__":
    test_init()
    print("Running forward pass...")
    test_forward()