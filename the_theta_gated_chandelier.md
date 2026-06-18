# The Theta-Gated Chandelier

## Version two: the AIS veto put on a clock — basket, chandelier, and medial septum as one timed circuit

*PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.*

> Do not hype. Do not lie. Just show.

---

## The one idea

v4 gave HKT the chandelier (axo-axonic) gate on the Koopman modes — the AIS veto. But it was a *constant* clamp, and the real cell is not constant. It **breathes with theta**, and a slower circuit drives the breathing. v2 builds that circuit, so three organs now sit on one clock:

- **basket / gamma** — the perisomatic gain window (v3, the basket gate);
- **chandelier / AIS** — the trigger-zone veto on the modes (v4, the chandelier gate);
- **medial septum / theta** — the slow clock that breathes the veto (v5, here).

That is the network image the whole thing was reaching for: a fast inhibitory veto at the trigger, opened and closed on a slow rhythm set from outside.

---

## The wiring, taken straight from the paper

Qi et al. (2024) report it almost ready to wire, for CA3:

- chandelier cells fire **rhythmically around the theta peak** — the veto is strong there;
- they are **silenced at the theta trough** by GABAergic **medial-septal** neurons firing at the trough — which **disinhibits** the pyramidal cells and contributes to the theta dipole. The trough is a **window**: the read is allowed to broadcast;
- chandelier cells **go silent during sharp-wave ripples** — replay runs disinhibited.

So the engine version:

```
theta_drive(phase) = 0.5*(1 + cos 2*pi*phase)     # 1 at peak (veto), 0 at trough (window)
chandelier_strength = base_strength * theta_drive  # the veto breathes
dream/ripple mode   -> theta_drive forced to 0     # ChC silent, replay disinhibited
```

The `ThetaClock` is the medial-septum analogue: a free-running theta oscillator whose phase sets the drive. Switch on **Probe 5 (θ-gate)** and the chandelier's clamp pulses at theta — full at the peak, released at the trough — and the veto% readout and the MS-theta bar breathe with it. Enter **Dream Mode** and the clock is treated as a ripple: the chandelier releases entirely and the free-running physics replays without the veto, exactly as ChCs fall silent in SWR.

This composes with everything v4 had: the homeostatic / rotation / learned modes all still apply; theta only modulates *how hard* they clamp this instant.

---

## Two clocks, two jobs (don't merge them)

The line already had a slow clock — the infraslow **Neuromodulatory Tide** (~0.07 Hz, the Priming-Tide repo). Keep them distinct:

- the **tide** is chemical readiness, 10–20 s — *whether* the substrate may resonate at all (β / expression gate);
- **theta** is the septal clock, 4–8 Hz — *when, within each cycle*, the read may broadcast (the chandelier window).

The honest next build is to **nest** them: theta inside the tide, so a fast veto-window breathes inside a slow chemical permission. That is the literal theta-in-tide of the membrane-to-qualia synthesis, with the chandelier as the thing being gated on both timescales.

---

## Ledger

**Established (used, not claimed):**
- CA3 chandelier cells fire rhythmically around the theta peak and are silent during sharp-wave ripples (Qi et al. 2024 and refs therein);
- a subset of GABAergic medial-septal neurons fire at the theta trough and inhibit CA3 chandelier cells there, disinhibiting pyramidal cells and contributing to the theta dipole;
- the medial septum is a theta pacemaker for the hippocampus (carried from the line's prior documents).

**Clean structural mappings (sound):**
- ThetaClock = the medial-septal theta pacemaker;
- chandelier strength ∝ theta peak, released at the trough = the measured ChC-theta phase relation;
- dream-mode release = ChC silence during ripples / replay;
- the resulting three-organ loop (basket-gamma window, chandelier-AIS veto, septal-theta clock) = the domain-specific inhibitory circuit of the cortical microcircuit, on one timer.

**Honest limits:**
- the webcam loop runs at ~30 fps; theta at 4–8 Hz is only ~4–8 frames per cycle, so this is a **structural demo** of theta gating, not a faithful theta rhythm — to study the rhythm itself the loop would need to run faster or theta would need to be slowed for visualisation;
- the ChC-theta phase relation is a **clean cosine**; the real one has scatter, and ChC firing is not a pure sinusoid of theta phase;
- the medial septum is a **bare free-running oscillator** here, not a modeled septal circuit (no cholinergic tone, no separate GABAergic/glutamatergic populations);
- HKT is **continuous and has no spikes**, so the chandelier stays a veto-at-the-read analogue, not axo-axonic spike gating;
- the homeostatic setpoint / strength remain uncalibrated knobs;
- **not run on hardware here** — written against v4; needs the GPU + webcam + teacher VAE to verify the breathing is visible and stable.

**The bet (untouched):** that any of this is *experienced* rather than processed. Putting the veto on a clock makes the circuit timed and inspectable; it does not make the timing felt, and it does not touch the hard problem.

---

## Where it goes next

1. **Nest theta in the infraslow tide** — the fast septal window breathing inside the slow chemical readiness, two clocks doing two jobs (whether × when).
2. **Homeostatic AIS plasticity** — let the chandelier setpoint adapt slowly to recent mean excitability (the AIS-shortening-under-drive result), so the veto self-tunes over minutes.
3. **The standing EEG hypothesis** (kept in the drawer) — degrade or desynchronise the chandelier and ask whether the constellation loses coherence in a way that rhymes with the trainless geometric-dysrhythmia result. A knob to test, not a claim.

---

*Helsinki, June 2026. The basket cell sets the gamma window at the soma; the chandelier vetoes at the trigger; and the medial septum, firing at the trough, opens the gate once a theta cycle so the read can speak. Three organs, one clock, and a veto that breathes instead of clamping. Do not hype. Do not lie. Just show.*
