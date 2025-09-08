# ai-sequencer.py — Generative Minimal Sequencer (VCV Rack via IAC) with 3 Channels + MIDI Clock
#
# Idea:
#  • Starts with very few notes (small motif)
#  • Slowly evolves: only small changes per cycle (Add/Rotate/Register-Shift)
#  • Long gates/ties (controllable), subtle phase shifts, minimal humanization
#  • No API access, purely local
#  • **NEW**: Third channel (Lead/Upper Voice) + per-lane gate variations
#
# Setup (macOS + VCV Rack):
#  1) Enable IAC Driver (Audio MIDI Setup → MIDI Studio → IAC Driver → "Enable Device")
#  2) VCV Rack: one instance of "Core → MIDI-CV" for Melody (Ch.1), Bass (Ch.2), Lead (Ch.3)
#     • Device: "IAC Driver Bus 1" (or equivalent)
#     • Channels: 1 / 2 / 3
#  3) Optional: "MIDI Clock" module – our script sends clock globally to the same port
#
# Installation:
#   python3 -m venv venv && source venv/bin/activate
#   pip install --upgrade pip
#   pip install mido python-rtmidi numpy openai
#
# Start:
#   python3 ai-sequencer.py

import time
import random
import math
import threading
import os
import json
import sys
import argparse
from typing import List, Dict, Any

import numpy as np
import mido
from mido import Message
import openai

def load_config_from_prompt(prompt_path: str) -> Dict[str, Any]:
    """
    Reads composition parameters from prompt.txt (JSON or simple Key:Value lines).
    Example prompt.txt:
        bpm: 120
        scale: dorian
        progression_degrees: [0, 3, 4, 5]
    Or as JSON:
        {"bpm": 120, "scale": "dorian", ...}
    """
    if not os.path.exists(prompt_path):
        print("[SequencerVCV] No prompt.txt found, using default configuration.")
        return {}
    with open(prompt_path, "r") as f:
        txt = f.read().strip()
        if not txt:
            print("[SequencerVCV] prompt.txt is empty, using default configuration.")
            return {}
        try:
            # Try JSON
            cfg = json.loads(txt)
            if isinstance(cfg, dict):
                return cfg
        except Exception:
            pass
        # Fallback: Key:Value Parsing
        cfg = {}
        for line in txt.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                try:
                    # Try to interpret value as int, float, or list
                    if v.startswith("[") and v.endswith("]"):
                        cfg[k] = json.loads(v)
                    elif v.isdigit():
                        cfg[k] = int(v)
                    else:
                        try:
                            cfg[k] = float(v)
                        except Exception:
                            cfg[k] = v
                except Exception:
                    cfg[k] = v
        return cfg

def generate_config_from_prompt(prompt_path: str) -> Dict[str, Any]:
    """
    Reads the prompt from prompt.txt and generates musical parameters using OpenAI GPT.
    """
    if not os.path.exists(prompt_path):
        print("[SequencerVCV] No prompt.txt found, using default configuration.")
        return {}
    with open(prompt_path, "r") as f:
        prompt = f.read().strip()
        if not prompt:
            print("[SequencerVCV] prompt.txt is empty, using default configuration.")
            return {}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set!")
    openai.api_key = api_key

    system_msg = (
        "You are an AI music assistant for a generative MIDI sequencer. "
        "Read the user's prompt and generate a JSON object with all necessary parameters for the sequencer script. "
        "Use meaningful values for bpm, scale, root, bars, steps_per_bar, progression_degrees, "
        "mel_base_motif_degrees, mel_max_register_shift, mel_evolve_every_bars, mel_tie_bias, mel_min_len_steps, mel_max_len_steps, mel_ghost_prob, "
        "bass_base_motif_degrees, bass_register_offset_oct, bass_max_register_shift, bass_evolve_every_bars, bass_tie_bias, bass_min_len_steps, bass_max_len_steps, "
        "lead_base_motif_degrees, lead_register_offset_oct, lead_max_register_shift, lead_evolve_every_bars, lead_tie_bias, lead_min_len_steps, lead_max_len_steps, "
        "chord_tone_bias, harmony_enable, clock_enable, swing, jitter_ms. "
        "Respond only with the JSON object, no explanations."
    )

    # Correct method for openai>=1.0.0
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )
    import json
    try:
        cfg = json.loads(response.choices[0].message.content)
        return cfg
    except Exception as e:
        print("[SequencerVCV] Error parsing AI response:", e)
        print("[SequencerVCV] AI response was:", response.choices[0].message.content)
        return {}

# ---------------- Configuration ----------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # Minimal defaults, everything else comes from prompt.txt
    "melody_channel": 1,
    "bass_channel": 2,
    "lead_channel": 3,
    "device": "IAC",  # Renamed from midi_port_hint
    "clock_enable": True,
}

CONFIG: Dict[str, Any] = DEFAULT_CONFIG.copy()

SCALES: Dict[str, List[int]] = {
    "major": [0,2,4,5,7,9,11],
    "minor": [0,2,3,5,7,8,10],
    "dorian": [0,2,3,5,7,9,10],
    "phrygian": [0,1,3,5,7,8,10],
    "lydian": [0,2,4,6,7,9,11],
    "mixolydian": [0,2,4,5,7,9,10],
    "locrian": [0,1,3,5,6,8,10],
    "whole_tone": [0,2,4,6,8,10],
    "octatonic": [0,1,3,4,6,7,9,10],
    "messiaen6": [0,2,4,5,6,8,10,11],
    "chromatic": list(range(12)),
    # Aphex-like extras
    "aphex_oct": [0,1,3,4,6,7,9,10],
    "aphex_tetra": [0,1,6,7],
    "aphex_frag": [0,1,2,5,7,8],
    "aphex_hybrid": [0,2,3,4,7,9,11],
    "aphex_sparse": [0,7,10],
}

def scale_len(scale_name: str) -> int:
    # Consider "mode" as an alias for "scale"
    return len(SCALES.get(scale_name, SCALES["dorian"]))

def chord_degrees(scale_name: str, root_degree: int, add_seventh: bool = True):
    """
    Returns chord-tone degrees (within the scale) for a diatonic triad/7th chord
    built on 'root_degree'. Operates in scale-steps, not semitones.
    """
    L = scale_len(scale_name)
    degrees = [ (root_degree + o) % L for o in (0, 2, 4) ]  # triad: 1-3-5
    if add_seventh:
        degrees.append((root_degree + 6) % L)              # add 7th: 1-3-5-7
    return degrees


# ---------------- Helper Functions ----------------

def list_outputs() -> List[str]:
    return mido.get_output_names()


def open_port(hint: str):
    names = list_outputs()
    if not names:
        raise RuntimeError("No MIDI Out found. IAC active?")
    for n in names:
        if hint.lower() in n.lower():
            return mido.open_output(n)
    return mido.open_output(names[0])


NOTE_NAME_TO_MIDI = {
    "C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63,
    "E": 64, "F": 65, "F#": 66, "Gb": 66, "G": 67, "G#": 68, "Ab": 68,
    "A": 69, "A#": 70, "Bb": 70, "B": 71
}

def parse_root_note(root):
    """
    Converts a note name (e.g. 'C', 'F#', 'A') or a number to a MIDI note number.
    Default: 57 (A3)
    """
    if isinstance(root, int):
        return root
    if isinstance(root, float):
        return int(root)
    if isinstance(root, str):
        r = root.strip().upper()
        # Optional: octave specification like 'C4'
        if len(r) > 1 and r[-1].isdigit():
            name = r[:-1]
            octave = int(r[-1])
            midi_base = NOTE_NAME_TO_MIDI.get(name, 60)
            return midi_base + (octave - 4) * 12
        return NOTE_NAME_TO_MIDI.get(r, 57)
    return 57

def scale_note(root: int, degree: int, scale_name: str) -> int:
    scale = SCALES.get(scale_name, SCALES["dorian"])  # Fallback dorian
    octave = degree // len(scale)
    idx = degree % len(scale)
    midi_root = parse_root_note(root)
    return midi_root + octave*12 + scale[idx]

# ---------------- State ----------------
class LaneState:
    def __init__(self, motif_degrees: List[int], max_register_shift: int):
        self.motif = motif_degrees.copy()
        self.register = 0
        self.rotation = 0
        self.max_reg = max_register_shift

    def evolve(self):
        r = random.random()
        if r < 0.45 and len(self.motif) > 1:
            self.rotation = (self.rotation + random.choice([1, -1])) % len(self.motif)
        elif r < 0.8:
            if len(self.motif) < 6 and random.random() < 0.6:
                anchor = random.choice(self.motif)
                delta = random.choice([-2, -1, 1, 2])
                self.motif.append(max(0, anchor + delta))
            elif len(self.motif) > 2:
                self.motif.pop(random.randrange(len(self.motif)))
        else:
            self.register = int(np.clip(self.register + random.choice([-1,1]), -self.max_reg, self.max_reg))

    def degrees_for_cycle(self) -> List[int]:
        if not self.motif:
            self.motif = [0]
        m = self.motif[self.rotation:] + self.motif[:self.rotation]
        return m + m

class GlobalState:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.mel = LaneState(cfg["mel_base_motif_degrees"], cfg["mel_max_register_shift"])
        self.bass = LaneState(cfg["bass_base_motif_degrees"], cfg["bass_max_register_shift"])
        self.lead = LaneState(cfg["lead_base_motif_degrees"], cfg["lead_max_register_shift"])
        self.iter = 0

# ---------------- Pattern Construction ----------------

def build_lane(cfg: Dict[str, Any], lane: LaneState, role: str) -> List[Dict[str,int]]:
    steps_total = cfg.get("steps_per_bar", 16) * cfg.get("bars", 8)
    steps_per_bar = cfg.get("steps_per_bar", 16)
    degrees_cycle = lane.degrees_for_cycle()
    bias_map = cfg.get("chord_tone_bias", {})
    if not isinstance(bias_map, dict):
        # If AI only provides a float, convert it
        bias_map = {"mel": bias_map, "bass": bias_map, "lead": bias_map}
    if role == "mel":
        Lmin = cfg.get("mel_min_len_steps", 2)
        Lmax = cfg.get("mel_max_len_steps", 16)
        tie_bias = cfg.get("mel_tie_bias", 0.7)
        vel0 = cfg.get("melody_velocity", 96)
        reg_off = 0
        ghost_prob = cfg.get("mel_ghost_prob", 0.04)
        gate_lo, gate_hi = cfg.get("mel_gate_var", (0.8, 1.0))
        chord_bias = bias_map.get("mel", 0.7)
    elif role == "bass":
        Lmin = cfg.get("bass_min_len_steps", 16)
        Lmax = cfg.get("bass_max_len_steps", 64)
        tie_bias = cfg.get("bass_tie_bias", 0.95)
        vel0 = cfg.get("bass_velocity", 100)
        reg_off = cfg.get("bass_register_offset_oct", -1) * 12
        ghost_prob = 0.0
        gate_lo, gate_hi = cfg.get("bass_gate_var", (0.9, 1.0))
        chord_bias = bias_map.get("bass", 1.0)
    else:  # lead
        Lmin = cfg.get("lead_min_len_steps", 1)
        Lmax = cfg.get("lead_max_len_steps", 8)
        tie_bias = cfg.get("lead_tie_bias", 0.45)
        vel0 = cfg.get("lead_velocity", 92)
        reg_off = cfg.get("lead_register_offset_oct", 1) * 12
        ghost_prob = cfg.get("lead_ghost_prob", 0.02)
        gate_lo, gate_hi = cfg.get("lead_gate_var", (0.4, 0.9))
        chord_bias = bias_map.get("lead", 0.6)

    events: List[Dict[str,int]] = []
    step_cursor = 0
    Lscale = scale_len(cfg.get("mode", cfg.get("scale", "dorian")))  # Prefer mode

    while step_cursor < steps_total:
        bar_idx = step_cursor // steps_per_bar
        if cfg.get("harmony_enable", True):
            block = cfg.get("chord_change_every_bars", 2)
            prog = cfg.get("progression_degrees", [0,5,3,4])
            chord_deg = prog[(bar_idx // max(1, block)) % len(prog)]
            chord_tones = chord_degrees(cfg.get("mode", cfg.get("scale", "dorian")), chord_deg, add_seventh=True)
        else:
            chord_tones = []

        if chord_tones and random.random() < chord_bias:
            if role == "bass":
                base_choice = random.choice([0,4])
                deg = (chord_deg + base_choice) % Lscale
            else:
                deg = random.choice(chord_tones)
        else:
            deg = random.choice(degrees_cycle)

        deg += lane.register * Lscale
        note = scale_note(cfg.get("root", 57), deg, cfg.get("mode", cfg.get("scale", "dorian"))) + reg_off

        if random.random() < tie_bias:
            length = random.randint(max(Lmin, 2), Lmax)
        else:
            length = random.randint(Lmin, max(Lmin+1, int(Lmax*0.5)))
        length = int(np.clip(length, 1, steps_total - step_cursor))

        vel = int(np.clip(vel0 + random.randint(-3, 3), 1, 127))
        gate = float(random.uniform(gate_lo, gate_hi))
        events.append({"step": step_cursor, "note": note, "len": length, "vel": vel, "gate": gate})

        if ghost_prob > 0 and random.random() < ghost_prob and step_cursor > 0:
            ghost_len = max(1, length // 4)
            ghost_vel = max(1, int(vel * 0.55))
            ghost_gate = max(0.2, gate * 0.6)
            events.append({"step": max(0, step_cursor-1), "note": note, "len": ghost_len, "vel": ghost_vel, "gate": ghost_gate})

        step_cursor += length

    events.sort(key=lambda e: e["step"]) 
    return events


def build_pattern(cfg: Dict[str, Any], st: GlobalState) -> Dict[str, Any]:
    patt = {
        "pattern": {
            "melody": build_lane(cfg, st.mel, "mel"),
            "bass": build_lane(cfg, st.bass, "bass"),
            "lead": build_lane(cfg, st.lead, "lead"),
        }
    }
    return patt

# ---------------- Clock Thread ----------------
class MidiClock:
    def __init__(self, port, cfg):
        self.port = port
        self.cfg = cfg
        self.running = False
        self.thread = None

    def start(self):
        if self.cfg.get("clock_enable", True) and not self.running:
            self.running = True
            self.port.send(Message('start'))  # 0xFA
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join()
            self.port.send(Message('stop'))   # 0xFC

    def _run(self):
        bpm = self.cfg["bpm"]
        interval = (60.0 / bpm) / 24.0  # 24 PPQN
        while self.running:
            self.port.send(Message('clock'))  # 0xF8
            time.sleep(interval)

# ---------------- Playback ----------------

def play_pattern(port, cfg: Dict[str, Any], patt: Dict[str, Any]):
    bpm = cfg["bpm"]
    step_dur = (60.0 / bpm) / 4.0  # 16th notes
    total_steps = cfg["steps_per_bar"] * cfg["bars"]
    swing = float(cfg.get("swing", 0.0))
    jitter = cfg.get("jitter_ms", 0) / 1000.0

    schedules = {name: [[] for _ in range(total_steps)] for name in ("melody","bass","lead")}
    for name in schedules.keys():
        for ev in patt["pattern"][name]:
            s = ev["step"] % total_steps
            schedules[name][s].append(ev)

    ch_map = {
        "melody": cfg["melody_channel"] - 1,
        "bass": cfg["bass_channel"] - 1,
        "lead": cfg["lead_channel"] - 1,
    }

    for s in range(total_steps):
        step_start = time.time()
        step_offset = swing * step_dur if (s % 2 == 1) else 0.0

        for name in ("melody","bass","lead"):
            for ev in schedules[name][s]:
                note = int(ev["note"])
                vel = int(ev["vel"])
                length_steps = int(ev["len"]) 
                gate_frac = float(ev.get("gate", 0.95))

                dur = step_dur * length_steps
                micro = random.uniform(-jitter, jitter)
                delay0 = max(0.0, step_offset + micro)

                now = time.time()
                wait = (delay0) - (now - step_start)
                if wait > 0:
                    time.sleep(wait)

                port.send(Message('note_on', channel=ch_map[name], note=note, velocity=vel))
                gate_time = max(0.03, dur * gate_frac)
                time.sleep(gate_time)
                port.send(Message('note_off', channel=ch_map[name], note=note, velocity=0))

        elapsed = time.time() - step_start
        rem = step_dur - elapsed
        if rem > 0:
            time.sleep(rem)

# ---------------- Runtime Logic ----------------

def all_notes_off(port, cfg: Dict[str, Any]):
    """
    Sends Note-Off for all notes on all used channels and a MIDI reset.
    """
    channels = [
        cfg["melody_channel"] - 1,
        cfg["bass_channel"] - 1,
        cfg["lead_channel"] - 1,
    ]
    # Turn off all notes in the range 0-127
    for ch in channels:
        for note in range(128):
            port.send(Message('note_off', channel=ch, note=note, velocity=0))
    # Send MIDI reset (General MIDI Reset: 0xFF is System Reset, but not all devices support this)
    port.send(Message('reset'))  # 0xFF


def print_sequence_parameters(cfg: Dict[str, Any]):
    """
    Prints the main musical parameters from the configuration.
    """
    # Use mode if available, otherwise scale
    mode = cfg.get("mode", None)
    scales = mode if mode else cfg.get("scale", "minor")
    bars = cfg.get("bars", 8)
    tempo = cfg.get("bpm", 70)
    root = cfg.get("root", "C")
    print(f"[Sequencer] Mode/Scale: {scales}")
    print(f"[Sequencer] Number of Bars: {bars}")
    print(f"[Sequencer] Tempo (BPM): {tempo}")
    print(f"[Sequencer] Root Key: {root}")

def main():
    parser = argparse.ArgumentParser(description="AI Sequencer")
    parser.add_argument("--device", type=str, help="MIDI device name hint (e.g. 'OXI', 'IAC', 'Arturia')")
    args = parser.parse_args()

    print("[SequencerVCV] Generating composition parameters from prompt.txt using AI …")
    user_cfg = generate_config_from_prompt("prompt.txt")
    CONFIG = DEFAULT_CONFIG.copy()
    CONFIG.update(user_cfg)

    # Ensure channel mapping: 1 = Bass, 2 = Melody, 3 = Lead
    CONFIG["bass_channel"] = 1
    CONFIG["melody_channel"] = 2
    CONFIG["lead_channel"] = 3

    # Use device from parameter if set
    if args.device:
        CONFIG["device"] = args.device

    # Display available devices
    print("[SequencerVCV] Available MIDI Out devices:")
    for idx, name in enumerate(list_outputs()):
        print(f"  [{idx}] {name}")
    # Optional: Device selection via input, only if no parameter is set
    if not args.device:
        user_hint = input("Device name (Enter for default/IAC): ").strip()
        if user_hint:
            CONFIG["device"] = user_hint
        else:
            CONFIG["device"] = "IAC"

    print(f"[SequencerVCV] Active configuration: {CONFIG}")

    print_sequence_parameters(CONFIG)

    print("[SequencerVCV] Opening MIDI Out …")
    out = open_port(CONFIG["device"])
    print(f"[SequencerVCV] Sending to: {out.name}")

    clock = MidiClock(out, CONFIG)
    clock.start()

    st = GlobalState(CONFIG)
    mel_evo = int(CONFIG.get("mel_evolve_every_bars", 8))
    bass_evo = int(CONFIG.get("bass_evolve_every_bars", 16))
    lead_evo = int(CONFIG.get("lead_evolve_every_bars", 12))
    bars_passed = 0

    try:
        while True:
            patt = build_pattern(CONFIG, st)
            play_pattern(out, CONFIG, patt)
            bars_passed += CONFIG.get("bars", 8)
            if mel_evo > 0 and (bars_passed % mel_evo == 0):
                st.mel.evolve()
            if bass_evo > 0 and (bars_passed % bass_evo == 0):
                st.bass.evolve()
            if lead_evo > 0 and (bars_passed % lead_evo == 0):
                st.lead.evolve()
    except KeyboardInterrupt:
        print("[SequencerVCV] Stopping …")
    finally:
        clock.stop()
        all_notes_off(out, CONFIG)

if __name__ == "__main__":
    main()

