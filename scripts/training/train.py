import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

from models.vits.models import SynthesizerTrn
from models.vits.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator
from models.vits.losses import generator_loss, discriminator_loss, feature_loss, kl_loss, mel_loss
from scripts.preprocess.audio_to_mel import spectrogram_torch, get_mel_spectrogram
from scripts.training.dataset import VITSDataset, collate_fn
from utils.text_utils import symbols


def slice_audio_segments(y, y_lengths, ids_slice, segment_size, hop_length):
    """
    Slice real audio to match generator segments.
    y: [B, T_samples], ids_slice: [B] in STFT frames, segment_size in samples.
    Returns [B, 1, segment_size].
    """
    b, t = y.size()
    device = y.device
    segments = torch.zeros(b, segment_size, device=device, dtype=y.dtype)
    for i in range(b):
        start_frame = ids_slice[i].item()
        start = start_frame * hop_length
        end = min(start + segment_size, y_lengths[i].item())
        if end > start:
            segments[i, : end - start] = y[i, start:end]
    return segments.unsqueeze(1)


def train_one_epoch(net_g, net_mpd, net_msd, train_loader, optim_g, optim_d, device, hps, epoch):
    net_g.train()
    net_mpd.train()
    net_msd.train()

    hop_length = hps["data"]["hop_length"]
    segment_size = hps["train"]["segment_size"]
    scaler = GradScaler('cuda', enabled=False)
    total_loss_g, total_loss_d, total_loss_kl, total_loss_mel, n_batches = 0.0, 0.0, 0.0, 0.0, 0

    for batch_idx, (texts, text_lengths, audios, audio_lengths, speaker_ids) in enumerate(train_loader):
        texts = texts.to(device)
        text_lengths = text_lengths.to(device)
        audios = audios.to(device)
        audio_lengths = audio_lengths.to(device)

        # Compute linear spectrograms for posterior encoder
        # audios: [B, T] -> spec: [B, spec_channels, T_spec]
        spec = spectrogram_torch(
            audios,
            n_fft=hps["data"]["n_fft"],
            hop_size=hop_length,
            win_size=hps["data"]["win_length"],
            center=False,
        )
        # Use actual spectrogram time dimension \u2014 STFT with center=False produces
        # floor((T - n_fft) / hop) + 1 frames, which is <= audio_lengths // hop_length
        spec_lengths = torch.tensor(
            [spec.size(2)] * spec.size(0), dtype=torch.long, device=device
        )

        # --- 1. DISCRIMINATOR STEP ---
        optim_d.zero_grad()
        with autocast('cuda', enabled=False):
            # Forward VITS
            y_hat, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                texts, text_lengths, spec, spec_lengths
            )

            # Slice real audio to match generated audio length
            y_seg = slice_audio_segments(audios, audio_lengths, ids_slice, segment_size, hop_length)

            # MPD + MSD Forward
            y_d_rs, y_d_gs, _, _ = net_mpd(y_seg, y_hat.detach())
            y_d_rs_s, y_d_gs_s, _, _ = net_msd(y_seg, y_hat.detach())

            loss_d = discriminator_loss(y_d_rs + y_d_rs_s, y_d_gs + y_d_gs_s)

        loss_d.backward()
        clip_grad_norm_(list(net_mpd.parameters()) + list(net_msd.parameters()), 5.0)
        optim_d.step()

        # --- 2. GENERATOR STEP ---
        optim_g.zero_grad()
        with autocast('cuda', enabled=False):
            # Re-run Discriminators on y_hat (no detach this time)
            y_d_rs, y_d_gs, fmap_rs, fmap_gs = net_mpd(y_seg, y_hat)
            y_d_rs_s, y_d_gs_s, fmap_rs_s, fmap_gs_s = net_msd(y_seg, y_hat)

            # KL regularization
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask) * 1.0

            # GAN + Feature Matching Losses
            loss_fm = feature_loss(fmap_rs, fmap_gs) + feature_loss(fmap_rs_s, fmap_gs_s)
            loss_gen = generator_loss(y_d_gs + y_d_gs_s)

            # Mel reconstruction loss: compute mel for both real slice and generated
            mel_real = spectrogram_torch(
                y_seg.squeeze(1), hps["data"]["n_fft"],
                hop_length, hps["data"]["win_length"], center=False
            )
            mel_fake = spectrogram_torch(
                y_hat.squeeze(1).detach(), hps["data"]["n_fft"],
                hop_length, hps["data"]["win_length"], center=False
            )
            min_t = min(mel_real.size(2), mel_fake.size(2))
            loss_mel = mel_loss(mel_real[..., :min_t], mel_fake[..., :min_t]) * 45.0

            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

        loss_gen_all.backward()
        clip_grad_norm_(net_g.parameters(), 5.0)
        optim_g.step()

        g_val = loss_gen_all.item()
        d_val = loss_d.item()
        if not (torch.isnan(loss_gen_all) or torch.isnan(loss_d)):
            total_loss_g += g_val
            total_loss_d += d_val
            total_loss_kl += loss_kl.item()
            total_loss_mel += loss_mel.item()
            n_batches += 1

        if batch_idx % 5 == 0:
            if torch.isnan(loss_gen_all) or torch.isnan(loss_d):
                print(f"  [Epoch {epoch}] Batch {batch_idx} | ⚠️  NaN detected")
            else:
                print(
                    f"  [Epoch {epoch}] Batch {batch_idx} | "
                    f"G: {g_val:.3f} | D: {d_val:.3f} | "
                    f"KL: {loss_kl.item():.3f} | Mel: {loss_mel.item():.3f}"
                )

    if n_batches > 0:
        return {
            "loss_g": total_loss_g / n_batches,
            "loss_d": total_loss_d / n_batches,
            "loss_kl": total_loss_kl / n_batches,
            "loss_mel": total_loss_mel / n_batches,
        }
    return None


def main():
    parser = argparse.ArgumentParser(description="Train VITS TTS model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 1)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs (default: 10)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  VITS-TTS Training | Device: {device} | Epochs: {args.epochs}")
    print(f"{'='*60}\n")

    # Load hyperparameters
    with open(os.path.join("configs", "config.json"), "r", encoding="utf-8") as f:
        hps = json.load(f)

    # Initialize Models
    net_g = SynthesizerTrn(
        n_vocab=len(symbols),
        spec_channels=513,
        segment_size=hps["train"]["segment_size"],
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

    net_mpd = MultiPeriodDiscriminator().to(device)
    net_msd = MultiScaleDiscriminator().to(device)

    # Optimizers
    optim_g = torch.optim.AdamW(net_g.parameters(), lr=2e-4, betas=(0.8, 0.99), eps=1e-9)
    optim_d = torch.optim.AdamW(
        list(net_mpd.parameters()) + list(net_msd.parameters()),
        lr=2e-4, betas=(0.8, 0.99), eps=1e-9,
    )

    # LR schedulers — exponential decay, standard for VITS
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=0.999)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=0.999)

    start_epoch = 1

    # Resume from checkpoint if requested
    if args.resume and os.path.isfile(args.resume):
        print(f"📂 Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        net_g.load_state_dict(ckpt["net_g"])
        net_mpd.load_state_dict(ckpt["net_mpd"])
        net_msd.load_state_dict(ckpt["net_msd"])
        optim_g.load_state_dict(ckpt["optim_g"])
        optim_d.load_state_dict(ckpt["optim_d"])
        scheduler_g.load_state_dict(ckpt["scheduler_g"])
        scheduler_d.load_state_dict(ckpt["scheduler_d"])
        start_epoch = ckpt["epoch"] + 1
        print(f"   Resumed at epoch {start_epoch}\n")

    # Dataset & DataLoader
    manifest_path = os.path.join("data", "processed", "train_list.txt")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run the preprocess scripts first.")

    train_dataset = VITSDataset(manifest_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    print(f"📊 Dataset: {len(train_dataset)} samples | Batch size: {hps['train']['batch_size']}")
    print(f"📋 Batches per epoch: {len(train_loader)}\n")

    os.makedirs("checkpoints", exist_ok=True)

    # --- Main Training Loop ---
    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        metrics = train_one_epoch(
            net_g, net_mpd, net_msd,
            train_loader, optim_g, optim_d,
            device, hps, epoch
        )
        scheduler_g.step()
        scheduler_d.step()
        elapsed = time.time() - t0

        if metrics:
            print(
                f"\n── Epoch {epoch:4d} complete ({elapsed:.1f}s) ──"
                f" G: {metrics['loss_g']:.4f}"
                f" | D: {metrics['loss_d']:.4f}"
                f" | KL: {metrics['loss_kl']:.4f}"
                f" | Mel: {metrics['loss_mel']:.4f}"
                f" | LR: {scheduler_g.get_last_lr()[0]:.2e}\n"
            )

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == (start_epoch + args.epochs - 1):
            ckpt_path = os.path.join("checkpoints", f"vits_epoch_{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "net_g": net_g.state_dict(),
                "net_mpd": net_mpd.state_dict(),
                "net_msd": net_msd.state_dict(),
                "optim_g": optim_g.state_dict(),
                "optim_d": optim_d.state_dict(),
                "scheduler_g": scheduler_g.state_dict(),
                "scheduler_d": scheduler_d.state_dict(),
                "hps": hps,
            }, ckpt_path)
            print(f"💾 Checkpoint saved → {ckpt_path}")

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()