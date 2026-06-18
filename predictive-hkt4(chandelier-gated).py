"""
predictive-hkt4.py  —  HKT with the Chandelier (AIS) gate on the Koopman modes
==============================================================================
v3 had a BASKET-CELL gate (perisomatic, common-mode whitening) at the spatial
latent. This adds the other inhibitory organ in the right place: the CHANDELIER
cell (axo-axonic) on the AXON INITIAL SEGMENT.

WHY THE MODES ARE THE RIGHT SPOT (and not the spatial latent or the attention):
In this framework the AIS *is* the Koopman read (HKT README; membrane_to_qualia_
synthesis). A chandelier cell sits on the AIS. So its engine analogue lands
exactly on the Koopman modes -- after the operator A is fit, before the
transformer reads its spectrum. You already had a HAND-OPERATED version of this:
Probe 2 (Isolate Mode) zeros every eigenvalue but one before the readout. The
chandelier gate just automates that probe into a rule.

GROUNDED IN (kept honest, see the_chandelier_gate.md for the ledger):
  Qi, Zhao, Tian, Lu, He, Tai (2024) "Specific and Plastic: Chandelier Cell-to-
  Axon Initial Segment Connections..." Neurosci. Bull. 40(11):1774-1788.
  - ChCs target the AIS exclusively (distal AIS, aligned with low-threshold
    Nav1.6) and fire mainly when network excitability is EXCESSIVE -> the
    'homeostatic' mode: a divisive shunt that engages with total mode energy
    (prevent-runaway / gain control), the paper's central functional finding.
  - the ChC effect is a SHUNT and its SIGN IS CONTESTED (depolarising vs
    hyperpolarising depending on Cl-); a shunt reduces firing regardless of
    polarity -> the gate is DIVISIVE (multiplicative), the sign-robust choice.
  - during arousal ChCs suppress spontaneous/background activity to raise SNR
    -> the 'rotation' mode passes moving modes, shunts the static cluster.
  - 'learned' is the trained controller (fixed to be differentiable with a
    straight-through estimator). It adds trained, DRIFTING params onto the
    otherwise trainless core -- labelled, optional, for comparison only.

HONEST SCOPE: HKT is continuous and has no spikes, so this is a functional
analogue of an AIS veto (veto-at-the-read), not a model of axo-axonic spike
gating. The homeostatic setpoint/strength are uncalibrated knobs -- tune them on
your hardware. The trainless modes (off/homeostatic/rotation) add no parameters;
only 'learned' touches the trainless-core claim.

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
# THE CHANDELIER (AIS) GATE  —  the automatic Isolate Mode
# =====================================================================
class ChandelierGate(nn.Module):
    """
    Axo-axonic (chandelier) gate on the Koopman modes = the AIS veto.

    Modes (selectable live):
      'off'         passthrough (back-compatible; behaves like v3).
      'homeostatic' DEFAULT, the paper-faithful one: a DIVISIVE shunt that engages
                    with total mode energy above a setpoint -- ChCs "fire when the
                    network is excessive". Uniform clamp (one ChC -> many cells).
      'rotation'    suppress the STATIC cluster (|phase|~0, the right-edge of the
                    eigen-constellation), pass the rotators -- the SNR / suppress-
                    spontaneous role. Soft, so you watch the static dots dim.
      'learned'     a trained per-mode gate with a straight-through estimator (the
                    differentiable fix of the naive (g>0.5) gate). LABELLED: this
                    adds trained, drifting parameters onto the trainless core.

    Reports self.last_veto (0..1): fraction of mode energy it removed this frame,
    for the constellation readout.
    """
    def __init__(self, top_k=16, setpoint=10.0, strength=0.6,
                 omega_thresh=0.20, omega_temp=0.10):
        super().__init__()
        self.top_k = int(top_k)
        self.setpoint = float(setpoint)     # total-|lambda| above which it clamps (tune!)
        self.strength = float(strength)     # clamp / veto strength
        self.omega_thresh = float(omega_thresh)
        self.omega_temp = float(omega_temp)
        # learned controller: only used in 'learned' mode. Reads the constellation.
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
            # ChC fires when EXCESSIVE: divisive shunt scaling with energy over setpoint.
            over = torch.relu(mag_feat.sum(dim=-1, keepdim=True) - self.setpoint)
            gate = 1.0 / (1.0 + s * over)                 # (B,1) uniform clamp, in (0,1]
            mag_feat = mag_feat * gate
            phase_feat = phase_feat * gate

        elif mode == 'rotation':
            # suppress static (|phase|~0), pass rotators. Soft gate; s scales veto depth.
            omega = phase_feat.abs()
            keep = torch.sigmoid((omega - self.omega_thresh) / self.omega_temp)
            keep = 1.0 - s * (1.0 - keep)
            mag_feat = mag_feat * keep
            phase_feat = phase_feat * keep

        elif mode == 'learned':
            ctrl_in = torch.cat([mag_feat, phase_feat], dim=-1).detach()
            soft = self.controller(ctrl_in)               # (B, top_k) in (0,1)
            hard = (soft > 0.5).float()
            gate = hard + soft - soft.detach()            # straight-through estimator
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

        # STAGE 1: Basket-Cell Gate (perisomatic common-mode filter) -- unchanged
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

        # Warm up phase
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

            # Predictive Physics (Matrix Exponentiation)
            if dream_steps > 1:
                A_base = A.clone()
                for _ in range(dream_steps - 1):
                    A = torch.bmm(A, A_base)
                    A = A * 0.99  # Physical Friction

            eigvals, _ = torch.linalg.eig(A)
            mag = torch.abs(eigvals)
            mag = torch.clamp(mag, max=1.0)

            _, idx = torch.topk(mag, self.top_k, dim=-1)

            top_eig = torch.gather(eigvals, 1, idx)
            mag_feat = torch.abs(top_eig)
            phase_feat = torch.angle(top_eig)

            # ---- CHANDELIER (AIS) GATE: the automatic Isolate Mode ----
            # the AIS = the Koopman read, so the axo-axonic veto lives here.
            mag_feat, phase_feat = self.chandelier(
                mag_feat, phase_feat, mode=chandelier_mode, strength=chandelier_strength
            )

            # PROBE 2: Modal Isolation (debug override, applied AFTER the gate)
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
            # the chandelier gate is part of the circuit during learning too, so the
            # network learns WITH the veto on (and the 'learned' controller can train).
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
        self.master.title("Predictive HKT v4 - Basket + Chandelier")
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

        # chandelier settings cached from the GUI thread, read by the training thread
        self._ch_mode = 'off'
        self._ch_str = 0.6

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
        # Top Controls
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
        tk.Label(probe_frame, text="BIOLOGICAL PROBES:", font=("Arial", 10, "bold")).pack(side='left', padx=5)

        self.probe1_var = tk.BooleanVar(value=False)
        tk.Checkbutton(probe_frame, text="1. Eigen Constellation", variable=self.probe1_var).pack(side='left', padx=5)

        tk.Label(probe_frame, text=" |  2. Isolate Mode (-1=All):").pack(side='left', padx=5)
        self.probe2_var = tk.IntVar(value=-1)
        tk.Spinbox(probe_frame, from_=-1, to=15, textvariable=self.probe2_var, width=5).pack(side='left')

        self.probe3_var = tk.BooleanVar(value=False)
        tk.Checkbutton(probe_frame, text=" |  3. Transformer's Gaze (Mask)", variable=self.probe3_var).pack(side='left', padx=5)

        # PROBE 4: Chandelier (AIS) gate on the Koopman modes
        tk.Label(probe_frame, text=" |  4. Chandelier (AIS):").pack(side='left', padx=5)
        self.probe4_var = tk.StringVar(value='off')
        tk.OptionMenu(probe_frame, self.probe4_var, 'off', 'homeostatic', 'rotation', 'learned').pack(side='left')
        tk.Label(probe_frame, text="strength:").pack(side='left')
        self.probe4_strength = tk.DoubleVar(value=0.6)
        tk.Spinbox(probe_frame, from_=0.0, to=3.0, increment=0.1, textvariable=self.probe4_strength, width=4).pack(side='left')

        # Video Display
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
            self.status_var.set("Dream Mode Active: Physics Extrapolation.")
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
                            chandelier_mode=self._ch_mode, chandelier_strength=self._ch_str
                        )
                        self.status_var.set(f"Training Active | Loss: {loss:.4f} | ChC: {self._ch_mode}")
                    except Exception:
                        pass
                self.frame_T = frame_T_plus_1.copy()
            time.sleep(0.1)

    def update_video(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)

        # cache the chandelier controls in the GUI thread for the training thread to read
        self._ch_mode = self.probe4_var.get()
        try:
            self._ch_str = float(self.probe4_strength.get())
        except Exception:
            self._ch_str = 0.6

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
                            chandelier_mode=self._ch_mode, chandelier_strength=self._ch_str
                        )
                        recon, mask_out = self.adaptive_decoder(spatial_latent, dynamic_latent)

                    if not self.dream_mode:
                        self.inference_buffer = updated_buffer

                    recon_np = recon.cpu().squeeze(0).permute(1, 2, 0).numpy()
                    recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)
                    display_frame = cv2.cvtColor(recon_np, cv2.COLOR_RGB2BGR)

                    # PROBE 3: Transformer's Gaze (Heatmap Overlay)
                    if self.probe3_var.get() and mask_out is not None:
                        mask_np = mask_out.mean(dim=1).cpu().squeeze(0).numpy()
                        mask_np = cv2.resize(mask_np, (512, 512), interpolation=cv2.INTER_NEAREST)
                        mask_bgr = cv2.applyColorMap((mask_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        display_frame = cv2.addWeighted(display_frame, 0.4, mask_bgr, 0.6, 0)

                    # PROBE 1: Eigen Constellation (Radar Plot)
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
                            # a mode the chandelier has shunted near zero collapses to the
                            # centre and is drawn dim red -- you watch the veto engage.
                            if mags[i] < 0.05:
                                color = (60, 60, 200)
                            elif i == isolate_val:
                                color = (0, 255, 0)
                            else:
                                color = (0, 150, 255)
                            cv2.circle(display_frame, (px, py), 4, color, -1)

                        veto = getattr(self.adaptive_encoder.chandelier, 'last_veto', 0.0)
                        cv2.putText(display_frame, f"ChC[{self._ch_mode}] veto {veto*100:.0f}%",
                                    (cx - r, cy + r + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (60, 200, 130) if self._ch_mode != 'off' else (120, 120, 120), 1)

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
