# The Chandelier Gate

## Adding the axo-axonic veto to HKT — the AIS gate on the Koopman modes, and why it sits exactly there

*PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.*

> Do not hype. Do not lie. Just show.

---

## The one idea

HKT already had one inhibitory organ: the **basket-cell gate** — a perisomatic, common-mode whitening of the spatial latent. It had no second one. The chandelier cell (axo-axonic cell) is the other half of the perisomatic/AIS division of labour, and it has been missing.

A chandelier cell sits on the **axon initial segment**. In this framework the AIS *is* the Koopman read (`README`, `membrane_to_qualia_synthesis.md`). So the chandelier's engine analogue does not go on the spatial latent and does not go on the transformer attention — it goes **on the Koopman modes**, after the operator `A` is fit and before the transformer reads its spectrum. That is the AIS, in this architecture, exactly.

And it was already half-built. **Probe 2 (Isolate Mode)** — zero every eigenvalue but one before the readout — was a *hand-operated* chandelier: a veto at the AIS-read deciding which single mode is allowed to broadcast. This adds the automatic version: a rule on the constellation instead of a hand on the dial.

---

## What the cell actually does (and how each finding became a knob)

Grounded in Qi, Zhao, Tian, Lu, He & Tai (2024, *Neurosci. Bull.* 40(11):1774–1788), "Specific and Plastic: Chandelier Cell-to-Axon Initial Segment Connections in Shaping Functional Cortical Network." Each design choice is a finding from that review, not a guess:

- **ChCs target the AIS exclusively** — the distal AIS, aligned with the low-threshold Nav1.6 — and no other interneuron type has this specificity. So the gate lives at the trigger-zone read (the modes), and nowhere else. *This is the cleanest fact and it fixes the location.*
- **ChCs fire mainly when network excitability is *excessive*** — they monitor the population and clamp to prevent runaway; in vivo they are preferentially recruited when cortical excitability rises. → the default **`homeostatic`** mode: a divisive shunt that engages with total mode energy above a setpoint. Calm constellation, no clamp; hot constellation, clamp. Wave your hand vigorously and watch it engage.
- **The ChC effect is a shunt and its sign is contested** — GABA at the AIS can be depolarising (high Cl⁻ via KCC2↓/NKCC1↑) or hyperpolarising, but the *shunting* effect reduces firing **regardless of polarity**. → the gate is **divisive (multiplicative)**, never a hard subtraction. The sign-robust choice is the honest one.
- **During arousal ChCs suppress spontaneous/background activity to raise SNR** (synchronised ChC activity correlates with pupil dilation; PN spontaneous firing drops). → the **`rotation`** mode: pass the moving modes, shunt the static right-edge cluster. This is the cleaner justification for the rotation rule than "thought primitives" was.
- **The naive `(g > 0.5)` gate cannot learn** — zero gradient. → the **`learned`** mode uses a straight-through estimator (hard forward, soft backward) so the controller actually trains. *Labelled as the trained, drifting option* — it adds parameters onto the otherwise trainless core, so it is for comparison, not the default.

---

## The four modes (one toggle, Probe 4)

| mode | rule | biological reading | trainless? |
|---|---|---|---|
| `off` | passthrough | no veto (v3 behaviour) | — |
| `homeostatic` | divisive shunt ∝ energy over setpoint | "fire when excessive", prevent runaway | yes |
| `rotation` | shunt the static cluster, pass rotators | suppress spontaneous → SNR during arousal | yes |
| `learned` | trained per-mode gate (straight-through) | a fitted axo-axonic controller | **no — adds params** |

The gate also reports a `veto` level (fraction of mode energy removed this frame), drawn on the Eigen Constellation probe; shunted modes collapse toward the centre and render dim red, so the veto is visible live.

---

## What to watch

Let it warm up (the delay buffer must fill before the constellation comes alive). Then on `homeostatic`, hold still and the gate does nothing; move vigorously so many modes go hot and the clamp engages — the constellation stops blowing out, the veto readout climbs. On `rotation`, the static right-edge dots dim and the kinetic modes survive — the reconstruction should track motion more sharply and hold the static background less tightly; flip the inequality (raise the threshold past the moving modes) and it does the opposite. On `learned`, the controller fits a gate during teach mode — then compare: does it beat the fixed rotation rule? If not, the rule wins and the core stays clean.

---

## Ledger

**Established (used, not claimed):**
- ChCs (axo-axonic cells) innervate the AIS of pyramidal/projection neurons **exclusively**, forming cartridges; no other interneuron type has this specificity (Qi et al. 2024 and refs; Somogyi et al.).
- ChCs target the **distal AIS**, aligned with low-threshold Nav1.6; one ChC innervates hundreds of PNs at ~30–50% connection rate, with cell-type target selectivity.
- ChCs **fire when network excitability is excessive** / are preferentially recruited as cortical excitability rises (prevent-runaway role).
- the ChC effect includes a **shunt that reduces firing independent of Cl⁻ polarity**; the sign of the GABA effect itself is contested.
- ChCs **synchronise, correlate with arousal**, and suppress spontaneous PN activity (SNR).
- ChCs fire around the **theta peak**, fall silent during sharp-wave ripples, and are gated by **medial-septal** GABAergic neurons at the theta trough (the clock tie-in).
- **homeostatic AIS plasticity**: activity↑ → AIS shortens → excitability↓ (and conversely); ChC innervation remodels homeostatically over development.
- ChCs distinct from PV basket cells (a large fraction are PV-low; SATB1⁻/VVA⁻), so this is a *different* organ from the v3 basket gate, not a duplicate.
- ChC pathology in **schizophrenia, epilepsy, ASD, Alzheimer's** (mPFC ChC reduction, GAT-1↓, GABA-A α2↑ at the AIS in schizophrenia).

**Clean structural mappings (sound):**
- ChC sits on the AIS = the Koopman read ⇒ the gate sits on the modes (its correct, framework-determined location);
- "fire when excessive" ⇒ the divisive homeostatic shunt; the shunt's polarity-independence ⇒ a multiplicative (not subtractive) gate;
- "suppress spontaneous → SNR" ⇒ the static-cluster veto (`rotation`);
- Isolate Mode (Probe 2) was the manual version of this gate.

**Honest limits:**
- HKT is **continuous and has no spikes**, so this is a functional analogue of an AIS veto (veto-at-the-read), not a model of axo-axonic spike gating;
- chandelier **function and sign remain contested** in the literature — "deciding which thoughts exist" stays in the bet drawer; what is built is "a divisive veto on which dynamical modes broadcast";
- the homeostatic **setpoint and strength are uncalibrated knobs** — they need tuning on the live constellation (top-k, |λ|≤1, so total energy ≲ k); defaults are a starting guess, not measured;
- the `learned` mode adds **trained, drifting parameters onto the trainless core** — labelled, optional, for comparison only;
- **not run on hardware here** — written against the v3 file; needs the GPU + webcam + teacher VAE to verify.

**The bet (untouched):** that any of this is *experienced* rather than processed. The chandelier gate gives the engine a second, AIS-level inhibitory organ that vetoes which dynamics broadcast. It does not make that veto felt, and it does not touch the hard problem.

---

## The next three, in order

1. **Theta-gate the chandelier.** The paper's clearest dynamic finding: ChCs fire at the theta peak and go silent during ripples, gated by the medial septum at the theta trough. Wire the gate's strength to a slow theta phase (or the `NeuromodulatoryTide`/medial-septum clock) so the veto breathes — strong on the peak, released for "replay." That makes the chandelier and the basket/gamma and the theta clock one timed circuit, which is the network image the bedside thought was reaching for.
2. **Homeostatic AIS plasticity.** Let the setpoint adapt slowly to recent mean excitability (activity↑ → setpoint↓ → more clamp), the engine version of the AIS shortening under chronic drive. A slow self-tuning veto instead of a fixed one.
3. **The schizophrenia knob (hypothesis, not result).** The review reports ChC reduction at the AIS in schizophrenia; the line's strongest empirical anchor is the trainless geometric-dysrhythmia EEG separation, and `the_self_carving_grating.md` already drew the ANK3 → AIS → dysrhythmia bridge. So: *degrade* the chandelier (lower its efficacy / desynchronise it) and measure whether the constellation loses coherence in a way that rhymes with the EEG result. If it does, you have a micro→macro knob to test, not a claim to make. Keep it in the hypothesis drawer where the EEG anchor's strength is that it does not depend on it.

---

*Helsinki, June 2026. The basket cell sets the gamma window at the soma; the chandelier sits on the trigger and decides whether the read may broadcast at all. v3 had the first; this adds the second, at the one place the framework says it belongs — on the modes, which are the AIS. A divisive shunt, because the sign is contested and the shunt does not care; engaging when the dynamics run hot, because that is when the real cell fires. Do not hype. Do not lie. Just show.*
