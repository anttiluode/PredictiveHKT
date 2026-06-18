"""
predictive-hkt5.py  —  the Theta-Gated Chandelier (the timed veto)
==================================================================
v4 added the CHANDELIER (axo-axonic) gate on the Koopman modes -- the AIS veto.
But it was a constant clamp. In the brain it is not constant: it BREATHES with
theta, and a slower circuit drives that breathing.

THE CIRCUIT THIS BUILDS (Qi et al. 2024, the CA3 findings, made into one loop):
  - chandelier cells fire RHYTHMICALLY around the THETA PEAK (veto strong);
  - they are SILENCED at the THETA TROUGH by GABAergic MEDIAL-SEPTAL neurons
    firing there -> the trough is a disinhibition WINDOW where pyramidal cells
    (the read) are allowed to broadcast;
  - chandelier cells go SILENT during sharp-wave RIPPLES (replay runs
    disinhibited).

So three organs now sit on one clock:
  - BASKET / gamma  -> the perisomatic gain window     (v3, the basket gate)
  - CHANDELIER / AIS -> the trigger-zone veto on modes  (v4, the chandelier gate)
  - MEDIAL SEPTUM / theta -> the slow clock that breathes the veto  (v5, here)

WHAT v5 ADDS:
  - ThetaClock: a free-running theta oscillator (the medial-septum analogue).
  - the chandelier's effective strength = base_strength * theta_drive, where
    theta_drive = 1 at the peak (full veto) and 0 at the trough (released window).
  - DREAM MODE = ripple/replay: the chandelier is released (drive -> 0), so the
    free-running physics replays disinhibited, exactly as ChCs fall silent in SWR.

HONEST SCOPE (see the_theta_gated_chandelier.md for the ledger):
  - the webcam loop runs at ~30 fps; theta (4-8 Hz) is only ~4-8 frames per cycle,
    so this is a STRUCTURAL demo of theta gating, not a faithful theta rhythm;
  - the ChC-theta phase relationship is implemented as a clean cosine; the real
    one has scatter;
  - the medial septum is a bare oscillator here, not a modeled septal circuit;
  - HKT is continuous and has no spikes, so this stays a veto-at-the-read analogue;
  - not run on hardware here -- written against v4; needs the GPU + webcam + VAE.

Do not hype. Do not lie. Just show.
PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
"""
import os
import threading
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import time

# ===============================
# Environment & PyTorch Setup (Triton-Free)
# ===============================
os.environ["DIFFUSERS_NO_IP_ADAPTER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)

from diffusers import AutoencoderKLTemporalDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
# THE MEDIAL-SEPTUM CLOCK  —  a free-running theta oscillator
# =====================================================================
class ThetaClock:
    """
    Medial-septum analogue. A free-running theta oscillator that breathes the
    chandelier veto. MS GABAergic cells fire at the theta TROUGH and silence the
    ChC there (the disinhibition window); the ChC vetoes hardest at the PEAK.

      drive(phase) = 0.5*(1 + cos(2*pi*phase))   # 1 at peak (phase 0), 0 at trough

    During ripple/replay (dream mode) the ChC goes silent regardless of phase, so
    `released=True` forces the drive to 0.
    """
    def __init__(self, freq_hz=6.0, fps=30.0):
        self.freq_hz = float(freq_hz)
        self.fps = float(fps)
        self.phase = 0.0

    def step(self, released=False):
        self.phase = (self.phase + self.freq_hz / max(self.fps, 1e-6)) % 1.0
        if released:                               # SWR / replay: ChC silent
            return 0.0, self.phase
        drive = 0.5 * (1.0 + np.cos(2.0 * np.pi * self.phase))
        return float(drive), self.phase


# =====================================================================
# THE CHANDELIER (AIS) GATE  —  the automatic Isolate Mode
# =====================================================================
class ChandelierGate(nn.Module):
    """
    Axo-axonic (chandelier) gate on the Koopman modes = the AIS veto. Unchanged
    from v4; the only new thing in v5 is that its `strength` is driven by the
    ThetaClock each frame instead of being constant.

    Modes: 'off' | 'homeostatic' (default, divisive shunt on excessive energy) |
           'rotation' (suppress static cluster, pass rotators) |
           'learned' (trained per-mode gate, straight-through; adds params).
    """
    def __init__(self, top_k=16, setpoint=10.0, strength=0.6,
                 omega_thresh=0.20, omega_temp=0.10):
        super().__init__()
        self.top_k = int(top_k)
        self.setpoint = float(setpoint)
        self.strength = float(strength)
        self.omega_thresh = float(omega_thresh)
        self.omega_temp = float(omega_temp)
        self.controller = nn.Sequential(
            nn.Linear(2 * self.top_k, 2 * self.top_k), nn.ReLU(),
            nn.Linear(2 * self.top_k, self.top_k), nn.Sigmoid(),
        )
        self.last_veto = 0.0

    def forward(self, mag_feat, phase_feat, mode='off', strength=None):
        if mode == 'off' or mag_feat is None:
            self.last_veto = 0.0
            return mag_feat, phase_feat

        s = self.strength if strength is None else float(strength)
        e0 = mag_feat.sum(dim=-1, keepdim=True) + 1e-6

        if mode == 'homeostatic':
            over = torch.relu(mag_feat.sum(dim=-1, keepdim=True) - self.setpoint)
            gate = 1.0 / (1.0 + s * over)
            mag_feat = mag_feat * gate
            phase_feat = phase_feat * gate

        elif mode == 'rotation':
            omega = phase_feat.abs()
            keep = torch.sigmoid((omega - self.omega_thresh) / self.omega_temp)
            keep = 1.0 - s * (1.0 - keep)
            mag_feat = mag_feat * keep
            phase_feat = phase_feat * keep

        elif mode == 'learned':
            ctrl_in = torch.cat([mag_feat, phase_feat], dim=-1).detach()
            soft = self.controller(ctrl_in)
            hard = (soft > 0.5).float()
            gate = hard + soft - soft.detach()
            gate = 1.0 - s * (1.0 - gate)
            mag_feat = mag_feat * gate
            phase_feat = phase_feat * gate

        e1 = mag_feat.sum(dim=-1, keepdim=True)
        self.last_veto = float((1.0 - (e1 / e0)).mean().clamp(0, 1).item())
        return mag_feat, phase_feat


# =====================================================================
# STAGE 1 & 2: Spatial Compression, Inhibitory Gating & Koopman Dynamics
# =====================================================================
class DynamicKoopmanEncoder(nn.Module):
    def __init__(self, delay=80, embed_dim=128, top_k=16):
        super(DynamicKoopmanEncoder, self).__init__()
        self.delay = delay
        self.top_k = top_k
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        # STAGE 1: Basket-Cell Gate (perisomatic common-mode filter)
        self.basket_cell_gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

        self.spatial_dim = 4 * 64 * 64
        self.bottleneck = nn.Linear(self.spatial_dim, 256)
        self.proj = nn.Linear(256 * (delay + 1), embed_dim)

        # STAGE 2b: Chandelier-Cell Gate (axo-axonic, on the AIS = the Koopman modes)
        self.chandelier = ChandelierGate(top_k=top_k)

    def forward(self, x, latent_buffer, dream_steps=1, isolate_mode=-1,
                chandelier_mode='off', chandelier_strength=None):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        spatial_latent = self.conv4(h)

        gate = self.basket_cell_gate(spatial_latent)
        gated_spatial_latent = spatial_latent * gate

        flat_latent = gated_spatial_latent.view(gated_spatial_latent.size(0), -1)
        compressed_latent = self.bottleneck(flat_latent)

        if len(latent_buffer) < self.delay:
            latent_buffer.append(compressed_latent.detach())
            return gated_spatial_latent, latent_buffer, None, None, None

        active_window = latent_buffer + [compressed_latent]
        next_buffer = latent_buffer[1:] + [compressed_latent.detach()]

        with torch.amp.autocast('cuda', enabled=False):
            x_pad = torch.stack(active_window, dim=1).float()
            x_pad = torch.nan_to_num(x_pad, nan=0.0, posinf=1e4, neginf=-1e4)

            x_embed = x_pad.view(x_pad.size(0), -1)
            x_embed = self.proj(x_embed).unsqueeze(1)

            X1 = x_pad[:, :-1, :]
            X2 = x_pad[:, 1:, :]

            eye = torch.eye(256, device=x.device, dtype=torch.float32).unsqueeze(0)
            X1T_X1 = torch.bmm(X1.transpose(1, 2), X1) + 1e-3 * eye

            inv_X1T_X1 = torch.linalg.pinv(X1T_X1)

            A = torch.bmm(torch.bmm(X2.transpose(1, 2), X1), inv_X1T_X1)
            A = torch.nan_to_num(A, nan=0.0, posinf=1e4, neginf=-1e4)

            if dream_steps > 1:
                A_base = A.clone()
                for _ in range(dream_steps - 1):
                    A = torch.bmm(A, A_base)
                    A = A * 0.99

            eigvals, _ = torch.linalg.eig(A)
            mag = torch.abs(eigvals)
            mag = torch.clamp(mag, max=1.0)

            _, idx = torch.topk(mag, self.top_k, dim=-1)

            top_eig = torch.gather(eigvals, 1, idx)
            mag_feat = torch.abs(top_eig)
            phase_feat = torch.angle(top_eig)

            # ---- CHANDELIER (AIS) GATE, theta-breathed via chandelier_strength ----
            mag_feat, phase_feat = self.chandelier(
                mag_feat, phase_feat, mode=chandelier_mode, strength=chandelier_strength
            )

            # PROBE 2: Modal Isolation (debug override, after the gate)
            if isolate_mode != -1 and 0 <= isolate_mode < self.top_k:
                mask = torch.zeros_like(mag_feat)
                mask[:, isolate_mode] = 1.0
                mag_feat = mag_feat * mask
                phase_feat = phase_feat * mask

            dynamic_latent = torch.cat([mag_feat, phase_feat], dim=-1)

        dynamic_latent = dynamic_latent.to(spatial_latent.dtype)
        return gated_spatial_latent, next_buffer, dynamic_latent, mag_feat, phase_feat


# =====================================================================
# STAGE 3: Semantic Reconstruction & Transformer Masking
# =====================================================================
class DynamicKoopmanDecoder(nn.Module):
    def __init__(self, top_k=16):
        super(DynamicKoopmanDecoder, self).__init__()
        koop_feat_dim = 2 * top_k

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=koop_feat_dim, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.mask_generator = nn.Sequential(
            nn.Linear(koop_feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4 * 64 * 64),
            nn.Sigmoid()
        )

        self.conv_trans1 = nn.ConvTranspose2d(4, 256, kernel_size=3, stride=1, padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv_trans3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv_trans4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, spatial_latent, dynamic_latent):
        mask_out = None
        if dynamic_latent is not None:
            trans_out = self.transformer(dynamic_latent.unsqueeze(1))
            mask = self.mask_generator(trans_out.squeeze(1))
            mask_out = mask.view(-1, 4, 64, 64)
            x = spatial_latent * mask_out
        else:
            x = spatial_latent

        x = self.relu(self.conv_trans1(x))
        x = self.relu(self.conv_trans2(x))
        x = self.relu(self.conv_trans3(x))
        recon = torch.sigmoid(self.conv_trans4(x))
        return recon, mask_out


# ===============================
# Predictive VAETrainer
# ===============================
class PredictiveVAETrainer:
    def __init__(self, encoder, decoder, teacher_vae):
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_vae = teacher_vae
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-4)
        self.loss_fn = nn.MSELoss()
        self.scaler = torch.amp.GradScaler('cuda')

    def train_on_transition(self, frame_T, frame_T_plus_1, latent_buffer,
                            chandelier_mode='off', chandelier_strength=None):
        self.encoder.train()
        self.decoder.train()
        self.optimizer.zero_grad()

        with torch.no_grad():
            target_latent = self.teacher_vae.encode(frame_T_plus_1.half()).latent_dist.sample().float()
            decoded = self.teacher_vae.decode(target_latent.half(), num_frames=1).sample
            target_image = ((decoded / 2 + 0.5).clamp(0, 1)).float()

        with torch.amp.autocast('cuda'):
            # the chandelier breathes with theta during learning too -- the network
            # learns across the theta cycle (encoding happens through theta).
            spatial_latent_T, updated_buffer, dynamic_latent_T, _, _ = self.encoder(
                frame_T, latent_buffer, dream_steps=1, isolate_mode=-1,
                chandelier_mode=chandelier_mode, chandelier_strength=chandelier_strength
            )
            pred_image_T_plus_1, _ = self.decoder(spatial_latent_T, dynamic_latent_T)

            loss = self.loss_fn(pred_image_T_plus_1, target_image)

        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item(), updated_buffer


# ===============================
# LatentVideoFilter: GUI & Loop
# ===============================
class LatentVideoFilter:
    def __init__(self, master):
        self.master = master
        self.master.title("Predictive HKT v5 - Theta-gated Chandelier")
        self.device = device

        print("Loading SVD Teacher VAE (Triton-Free)...")
        self.teacher_vae = AutoencoderKLTemporalDecoder.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            subfolder="vae",
            torch_dtype=torch.float16
        ).to(self.device)
        self.teacher_vae.eval()

        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.cap = None

        # the medial-septum theta clock
        self.theta_clock = ThetaClock(freq_hz=6.0, fps=30.0)

        # chandelier settings cached from the GUI thread, read by the training thread
        self._ch_mode = 'off'
        self._ch_base = 0.6         # base strength from the spinbox
        self._ch_eff = 0.6          # theta-modulated effective strength (what we pass)
        self._theta_drive = 1.0
        self._theta_phase = 0.0

        self.setup_gui()

        self.adaptive_encoder = DynamicKoopmanEncoder().to(self.device)
        self.adaptive_decoder = DynamicKoopmanDecoder().to(self.device)
        self.adaptive_trainer = PredictiveVAETrainer(self.adaptive_encoder, self.adaptive_decoder, self.teacher_vae)

        self.inference_buffer = []
        self.train_buffer = []

        self.teach_mode = False
        self.dream_mode = False
        self.dream_steps = 1
        self.frozen_tensor = None
        self.frozen_buffer = None

        self.latest_frame = None
        self.frame_T = None
        self.frame_lock = threading.Lock()

        self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
        self.training_thread.start()

        self.update_video()

    def setup_gui(self):
        control_frame = tk.Frame(self.master)
        control_frame.pack(side='top', fill='x', padx=10, pady=5)

        self.teach_button = tk.Button(control_frame, text="Start Teach Mode", command=self.toggle_teach_mode)
        self.teach_button.pack(side='left', padx=5)

        self.dream_button = tk.Button(control_frame, text="Start Dream Mode", command=self.toggle_dream_mode)
        self.dream_button.pack(side='left', padx=5)

        self.save_button = tk.Button(control_frame, text="Save", command=self.save_model)
        self.save_button.pack(side='left', padx=5)
        self.load_button = tk.Button(control_frame, text="Load", command=self.load_model)
        self.load_button.pack(side='left', padx=5)

        # PROBE CONTROLS
        probe_frame = tk.Frame(self.master, relief='groove', borderwidth=2)
        probe_frame.pack(side='top', fill='x', padx=10, pady=5)
        tk.Label(probe_frame, text="PROBES:", font=("Arial", 10, "bold")).pack(side='left', padx=5)

        self.probe1_var = tk.BooleanVar(value=False)
        tk.Checkbutton(probe_frame, text="1. Eigen Constellation", variable=self.probe1_var).pack(side='left', padx=5)

        tk.Label(probe_frame, text=" | 2. Isolate (-1=All):").pack(side='left', padx=5)
        self.probe2_var = tk.IntVar(value=-1)
        tk.Spinbox(probe_frame, from_=-1, to=15, textvariable=self.probe2_var, width=4).pack(side='left')

        self.probe3_var = tk.BooleanVar(value=False)
        tk.Checkbutton(probe_frame, text=" | 3. Gaze (Mask)", variable=self.probe3_var).pack(side='left', padx=5)

        # PROBE 4: Chandelier (AIS) gate
        tk.Label(probe_frame, text=" | 4. Chandelier:").pack(side='left', padx=5)
        self.probe4_var = tk.StringVar(value='off')
        tk.OptionMenu(probe_frame, self.probe4_var, 'off', 'homeostatic', 'rotation', 'learned').pack(side='left')
        tk.Label(probe_frame, text="str:").pack(side='left')
        self.probe4_strength = tk.DoubleVar(value=0.6)
        tk.Spinbox(probe_frame, from_=0.0, to=3.0, increment=0.1, textvariable=self.probe4_strength, width=4).pack(side='left')

        # PROBE 5: Theta gate (medial septum breathes the chandelier)
        self.probe5_var = tk.BooleanVar(value=False)
        tk.Checkbutton(probe_frame, text=" | 5. \u03b8-gate", variable=self.probe5_var).pack(side='left', padx=5)
        tk.Label(probe_frame, text="Hz:").pack(side='left')
        self.probe5_hz = tk.DoubleVar(value=6.0)
        tk.Spinbox(probe_frame, from_=1.0, to=12.0, increment=0.5, textvariable=self.probe5_hz, width=4).pack(side='left')

        self.video_label = tk.Label(self.master)
        self.video_label.pack(padx=10, pady=10)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(self.master, textvariable=self.status_var, relief='sunken', anchor='w')
        self.status_label.pack(side='bottom', fill='x')

    def toggle_teach_mode(self):
        self.teach_mode = not self.teach_mode
        if self.teach_mode:
            self.teach_button.config(text="Stop Teach Mode")
            self.status_var.set("Teach mode active: Learning physics.")
            with self.frame_lock:
                self.frame_T = self.latest_frame.copy() if self.latest_frame is not None else None
        else:
            self.teach_button.config(text="Start Teach Mode")
            self.status_var.set("Teach mode paused.")

    def toggle_dream_mode(self):
        self.dream_mode = not self.dream_mode
        if self.dream_mode:
            self.dream_button.config(text="Stop Dream Mode")
            self.status_var.set("Dream Mode: ripple/replay (chandelier released).")
            with self.frame_lock:
                if self.latest_frame is not None:
                    image = Image.fromarray(cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB))
                    self.frozen_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    self.frozen_buffer = [b.clone() for b in self.inference_buffer]
                    self.dream_steps = 1
        else:
            self.dream_button.config(text="Start Dream Mode")
            self.status_var.set("Reconnected to Webcam.")
            self.frozen_tensor = None
            self.frozen_buffer = None
            self.inference_buffer = []

    def save_model(self):
        filename = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch", "*.pth")])
        if filename:
            torch.save({
                'encoder': self.adaptive_encoder.state_dict(),
                'decoder': self.adaptive_decoder.state_dict(),
            }, filename)

    def load_model(self):
        filename = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth")])
        if filename:
            checkpoint = torch.load(filename, map_location=self.device)
            self.adaptive_encoder.load_state_dict(checkpoint['encoder'])
            self.adaptive_decoder.load_state_dict(checkpoint['decoder'])

    def training_loop(self):
        while True:
            if self.teach_mode and not self.dream_mode and self.latest_frame is not None:
                with self.frame_lock:
                    frame_T_plus_1 = self.latest_frame.copy()
                if self.frame_T is not None:
                    try:
                        img_T = Image.fromarray(cv2.cvtColor(self.frame_T, cv2.COLOR_BGR2RGB))
                        img_T1 = Image.fromarray(cv2.cvtColor(frame_T_plus_1, cv2.COLOR_BGR2RGB))
                        t_T = self.transform(img_T).unsqueeze(0).to(self.device)
                        t_T1 = self.transform(img_T1).unsqueeze(0).to(self.device)

                        loss, self.train_buffer = self.adaptive_trainer.train_on_transition(
                            t_T, t_T1, self.train_buffer,
                            chandelier_mode=self._ch_mode, chandelier_strength=self._ch_eff
                        )
                        self.status_var.set(
                            f"Training | Loss {loss:.4f} | ChC {self._ch_mode} | "
                            f"\u03b8 {self._theta_phase:.2f} drive {self._theta_drive:.2f}"
                        )
                    except Exception:
                        pass
                self.frame_T = frame_T_plus_1.copy()
            time.sleep(0.1)

    def update_video(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)

        # cache the chandelier controls + advance the theta clock (GUI thread)
        self._ch_mode = self.probe4_var.get()
        try:
            self._ch_base = float(self.probe4_strength.get())
        except Exception:
            self._ch_base = 0.6

        self.theta_clock.freq_hz = float(self.probe5_hz.get())
        # dream mode = ripple: the chandelier is released regardless of phase
        drive, phase = self.theta_clock.step(released=self.dream_mode)
        self._theta_drive, self._theta_phase = drive, phase
        if self.probe5_var.get():
            self._ch_eff = self._ch_base * drive          # veto breathes with theta
        else:
            self._ch_eff = self._ch_base                  # constant clamp (v4 behaviour)

        if self.cap is not None and self.cap.isOpened():
            try:
                if not self.dream_mode:
                    ret, real_frame = self.cap.read()
                    if ret:
                        with self.frame_lock:
                            self.latest_frame = real_frame.copy()
                        image = Image.fromarray(cv2.cvtColor(real_frame, cv2.COLOR_BGR2RGB))
                        active_tensor = self.transform(image).unsqueeze(0).to(self.device)
                        active_buffer = self.inference_buffer
                        current_steps = 1
                else:
                    active_tensor = self.frozen_tensor
                    active_buffer = self.frozen_buffer
                    current_steps = self.dream_steps
                    self.dream_steps += 1

                if active_tensor is not None:
                    isolate_val = self.probe2_var.get()
                    with torch.no_grad():
                        spatial_latent, updated_buffer, dynamic_latent, mag_f, phase_f = self.adaptive_encoder(
                            active_tensor, active_buffer, dream_steps=current_steps,
                            isolate_mode=isolate_val,
                            chandelier_mode=self._ch_mode, chandelier_strength=self._ch_eff
                        )
                        recon, mask_out = self.adaptive_decoder(spatial_latent, dynamic_latent)

                    if not self.dream_mode:
                        self.inference_buffer = updated_buffer

                    recon_np = recon.cpu().squeeze(0).permute(1, 2, 0).numpy()
                    recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)
                    display_frame = cv2.cvtColor(recon_np, cv2.COLOR_RGB2BGR)

                    # PROBE 3: Transformer's Gaze
                    if self.probe3_var.get() and mask_out is not None:
                        mask_np = mask_out.mean(dim=1).cpu().squeeze(0).numpy()
                        mask_np = cv2.resize(mask_np, (512, 512), interpolation=cv2.INTER_NEAREST)
                        mask_bgr = cv2.applyColorMap((mask_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        display_frame = cv2.addWeighted(display_frame, 0.4, mask_bgr, 0.6, 0)

                    # PROBE 1: Eigen Constellation
                    if self.probe1_var.get() and mag_f is not None and phase_f is not None:
                        cx, cy, r = 420, 90, 70
                        cv2.circle(display_frame, (cx, cy), r, (100, 100, 100), 1)
                        cv2.line(display_frame, (cx - r, cy), (cx + r, cy), (100, 100, 100), 1)
                        cv2.line(display_frame, (cx, cy - r), (cx, cy + r), (100, 100, 100), 1)

                        mags = mag_f.cpu().squeeze(0).numpy()
                        phases = phase_f.cpu().squeeze(0).numpy()
                        for i in range(len(mags)):
                            px = int(cx + mags[i] * r * np.cos(phases[i]))
                            py = int(cy + mags[i] * r * np.sin(phases[i]))
                            if mags[i] < 0.05:
                                color = (60, 60, 200)        # chandelier-vetoed -> dim red
                            elif i == isolate_val:
                                color = (0, 255, 0)
                            else:
                                color = (0, 150, 255)
                            cv2.circle(display_frame, (px, py), 4, color, -1)

                        veto = getattr(self.adaptive_encoder.chandelier, 'last_veto', 0.0)
                        cv2.putText(display_frame, f"ChC[{self._ch_mode}] veto {veto*100:.0f}%",
                                    (cx - r, cy + r + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (60, 200, 130) if self._ch_mode != 'off' else (120, 120, 120), 1)

                    # theta dial: a small breathing bar (medial-septum clock)
                    if self.probe5_var.get():
                        bx, by, bw = 360, 200, 120
                        cv2.putText(display_frame, "MS theta", (bx, by - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 180, 90), 1)
                        cv2.rectangle(display_frame, (bx, by), (bx + bw, by + 10), (70, 70, 70), 1)
                        cv2.rectangle(display_frame, (bx, by),
                                      (bx + int(bw * self._theta_drive), by + 10), (90, 200, 230), -1)
                        tag = "PEAK: veto ON" if self._theta_drive > 0.6 else \
                              ("TROUGH: window" if self._theta_drive < 0.4 else "...")
                        if self.dream_mode:
                            tag = "RIPPLE: released (replay)"
                        cv2.putText(display_frame, tag, (bx, by + 26),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 200, 230), 1)

                    image_pil = Image.fromarray(display_frame)
                    photo = ImageTk.PhotoImage(image=image_pil)
                    self.video_label.config(image=photo)
                    self.video_label.image = photo

            except Exception as e:
                self.status_var.set(f"Runtime Error: {e}")

            self.master.after(30, self.update_video)

    def run(self):
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.master.mainloop()

    def on_closing(self):
        if self.cap is not None:
            self.cap.release()
        self.master.destroy()


def main():
    root = tk.Tk()
    app = LatentVideoFilter(root)
    app.run()


if __name__ == "__main__":
    main()
