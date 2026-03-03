import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

# Model Imports
from models.vits.generator import Generator
from models.vits.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator
from models.vits.losses import generator_loss, discriminator_loss, feature_loss, kl_loss
# Assuming you've built the data loading logic in data_utils.py
# from data_utils import TextAudioLoader, TextAudioCollate

def train_one_epoch(net_g, net_d, train_loader, optim_g, optim_d, scaler, device):
    net_g.train()
    net_d.train()

    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
        x, x_lengths = x.to(device), x_lengths.to(device)
        spec, spec_lengths = spec.to(device), spec_lengths.to(device)
        y, y_lengths = y.to(device), y_lengths.to(device)

        # --- 1. DISCRIMINATOR STEP ---
        optim_d.zero_grad()
        with autocast(enabled=False): # Keep FP32 for stability if possible
            # Forward Generator (Inference mode for Disc step)
            y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths)
            
            # Slice real audio to match generated audio length
            # y_mel_slice = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.data.segment_size)
            
            # MPD + MSD Forward
            y_d_rs, y_d_gs, _, _ = net_d[0](y, y_hat.detach())
            y_d_rs_s, y_d_gs_s, _, _ = net_d[1](y, y_hat.detach())
            
            loss_d = discriminator_loss(y_d_rs + y_d_rs_s, y_d_gs + y_d_gs_s)

        loss_d.backward()
        clip_grad_norm_(net_d.parameters(), 5.0)
        optim_d.step()

        # --- 2. GENERATOR STEP ---
        optim_g.zero_grad()
        with autocast(enabled=False):
            # Re-run Discriminators on y_hat (no detach this time)
            y_d_rs, y_d_gs, fmap_rs, fmap_gs = net_d[0](y, y_hat)
            y_d_rs_s, y_d_gs_s, fmap_rs_s, fmap_gs_s = net_d[1](y, y_hat)
            
            # Reconstruction Losses
            loss_mel = F.l1_loss(spec, spec) * 45.0 # Placeholder for actual Mel loss
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * 1.0
            
            # GAN + Feature Matching Losses
            loss_fm = feature_loss(fmap_rs, fmap_gs) + feature_loss(fmap_rs_s, fmap_gs_s)
            loss_gen = generator_loss(y_d_gs + y_d_gs_s)
            
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

        loss_gen_all.backward()
        clip_grad_norm_(net_g.parameters(), 5.0)
        optim_g.step()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} | Loss G: {loss_gen_all.item():.4f} | Loss D: {loss_d.item():.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting VITS training on {device} (GTX 1650)")

    # Initialize Models (Use your previously validated configs)
    net_g = Generator().to(device)
    net_mpd = MultiPeriodDiscriminator().to(device)
    net_msd = MultiScaleDiscriminator().to(device)
    net_d = torch.nn.ModuleList([net_mpd, net_msd])

    # Optimizers
    optim_g = torch.optim.AdamW(net_g.parameters(), lr=2e-4, betas=(0.8, 0.99), eps=1e-9)
    optim_d = torch.optim.AdamW(net_d.parameters(), lr=2e-4, betas=(0.8, 0.99), eps=1e-9)
    
    scaler = GradScaler(enabled=False) # Start with FP32

    # TODO: Load LJSpeech Dataset here
    # train_dataset = TextAudioLoader("path/to/ljspeech/metadata.csv", hps.data)
    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=TextAudioCollate())

    print("Model initialized. Waiting for dataset path...")

if __name__ == "__main__":
    main()