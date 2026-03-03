import torch
import torch.nn.functional as F

def feature_loss(fmap_r, fmap_g):
    """Feature Matching Loss: Ensures synthetic features match real ones."""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """Standard Hinge/LSGAN loss for the Discriminator."""
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
    return loss

def generator_loss(disc_outputs):
    """Adversarial loss for the Generator."""
    loss = 0
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        loss += l
    return loss

def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """KL Divergence for VAE regularization."""
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    # Clamp to prevent exp(-2*logs_p) from exploding when logs_p is very negative
    logs_p_clamped = torch.clamp(logs_p, min=-10, max=10)
    kl = logs_p_clamped - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p_clamped)
    kl = torch.sum(kl * z_mask)
    return kl / torch.sum(z_mask)


def mel_loss(y_mel, y_hat_mel):
    """L1 loss on log-mel spectrograms. Key stabilizer for VITS training."""
    return F.l1_loss(y_mel, y_hat_mel)