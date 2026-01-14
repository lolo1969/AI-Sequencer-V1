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
import signal
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

import numpy as np
import mido
from mido import Message
from openai import OpenAI

# Constants
PPQN = 24  # Pulses per quarter note for MIDI clock
MIN_GATE_TIME = 0.03  # Minimum gate time in seconds
MAX_MOTIF_LENGTH = 6
MIN_MOTIF_LENGTH = 2
DEFAULT_BPM = 120
DEFAULT_VELOCITY = 96
VELOCITY_VARIATION = 10

# Lazy initialization of OpenAI client
_openai_client: Optional[OpenAI] = None

def get_openai_client() -> OpenAI:
    """Lazily initialize OpenAI client only when needed."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set!")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

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

    system_msg = (
        "You are an AI music assistant for a generative MIDI sequencer. "
        "Generate a JSON object with musical parameters that create coherent and musical sequences. "
        "Focus on diatonic harmony, smooth voice leading, and balanced rhythmic patterns. "
        "Ensure the bass provides a strong harmonic foundation, the melody is lyrical, and the lead adds expressive ornamentation. "
        "Use meaningful values for bpm, scale, root, bars, steps_per_bar, progression_degrees, "
        "mel_base_motif_degrees, mel_max_register_shift, mel_evolve_every_bars, mel_tie_bias, mel_min_len_steps, mel_max_len_steps, mel_ghost_prob, "
        "bass_base_motif_degrees, bass_register_offset_oct, bass_max_register_shift, bass_evolve_every_bars, bass_tie_bias, bass_min_len_steps, bass_max_len_steps, "
        "lead_base_motif_degrees, lead_register_offset_oct, lead_max_register_shift, lead_evolve_every_bars, lead_tie_bias, lead_min_len_steps, lead_max_len_steps, "
        "chord_tone_bias, harmony_enable, clock_enable, swing, jitter_ms. "
        "Respond only with the JSON object, no explanations."
    )

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=800
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[SequencerVCV] Error parsing AI JSON response: {e}")
        return {}
    except Exception as e:
        print(f"[SequencerVCV] Error communicating with OpenAI: {e}")
        return {}

# Predefined motifs for more musical starting points
DEFAULT_MOTIFS = {
    "melody": [0, 2, 4, 5],
    "bass": [0, -2, -4],
    "lead": [7, 9, 12],
}

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
    """Returns a list of available MIDI output port names."""
    try:
        return mido.get_output_names()
    except Exception as e:
        print(f"[SequencerVCV] Error listing MIDI outputs: {e}")
        return []


def open_port(hint: str) -> mido.ports.BaseOutput:
    """Opens a MIDI output port matching the hint string."""
    names = list_outputs()
    if not names:
        raise RuntimeError("No MIDI Out found. Is IAC Driver active?")
    for n in names:
        if hint.lower() in n.lower():
            print(f"[SequencerVCV] Found matching port: {n}")
            return mido.open_output(n)
    print(f"[SequencerVCV] No port matching '{hint}', using first available: {names[0]}")
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
    """Convert a scale degree to a MIDI note number."""
    scale = SCALES.get(scale_name, SCALES["dorian"])  # Fallback dorian
    octave = degree // len(scale)
    idx = degree % len(scale)
    midi_root = parse_root_note(root)
    return midi_root + octave*12 + scale[idx]


def clamp_midi_note(note: int) -> int:
    """Clamp a MIDI note to the valid range 0-127, with octave wrapping."""
    while note < 0:
        note += 12
    while note > 127:
        note -= 12
    return int(np.clip(note, 0, 127))

# ---------------- State ----------------
class LaneState:
    """Represents the state of a single melodic lane (melody, bass, or lead)."""
    
    def __init__(self, motif_degrees: List[int], max_register_shift: int):
        self.motif = motif_degrees.copy() if motif_degrees else DEFAULT_MOTIFS.get("melody", [0])
        self.register = 0
        self.rotation = 0
        self.max_reg = max(1, max_register_shift)  # Ensure at least 1

    def evolve(self) -> None:
        """Evolve the motif through rotation, addition/removal, or register shift."""
        r = random.random()
        if r < 0.45 and len(self.motif) > MIN_MOTIF_LENGTH:
            # Rotate motif
            self.rotation = (self.rotation + random.choice([1, -1])) % len(self.motif)
        elif r < 0.8:
            # Add or remove note
            if len(self.motif) < MAX_MOTIF_LENGTH and random.random() < 0.6:
                anchor = random.choice(self.motif)
                delta = random.choice([-2, -1, 1, 2])
                self.motif.append(max(0, anchor + delta))
            elif len(self.motif) > MIN_MOTIF_LENGTH:
                self.motif.pop(random.randrange(len(self.motif)))
        else:
            # Register shift
            self.register = int(np.clip(self.register + random.choice([-1, 1]), -self.max_reg, self.max_reg))

    def degrees_for_cycle(self) -> List[int]:
        """Returns the current motif degrees arranged for a full cycle."""
        if not self.motif:
            self.motif = [0]
        m = self.motif[self.rotation:] + self.motif[:self.rotation]
        return m + m  # Double for longer phrases

class GlobalState:
    """Manages the state of all three melodic lanes."""
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.mel = LaneState(
            cfg.get("mel_base_motif_degrees", DEFAULT_MOTIFS["melody"]),
            cfg.get("mel_max_register_shift", 1)
        )
        self.bass = LaneState(
            cfg.get("bass_base_motif_degrees", DEFAULT_MOTIFS["bass"]),
            cfg.get("bass_max_register_shift", 1)
        )
        self.lead = LaneState(
            cfg.get("lead_base_motif_degrees", DEFAULT_MOTIFS["lead"]),
            cfg.get("lead_max_register_shift", 1)
        )
        self.iter = 0


def get_scale_name(cfg: Dict[str, Any]) -> str:
    """Helper to get scale name from config, handling mode/scale aliases."""
    return cfg.get("mode", cfg.get("scale", "dorian"))

# ---------------- Pattern Construction ----------------

def build_lane(cfg: Dict[str, Any], lane: LaneState, role: str) -> List[Dict[str, Any]]:
    """Build a sequence of note events for a single lane."""
    steps_total = cfg.get("steps_per_bar", 16) * cfg.get("bars", 8)
    steps_per_bar = cfg.get("steps_per_bar", 16)
    degrees_cycle = lane.degrees_for_cycle()
    scale_name = get_scale_name(cfg)
    
    # Handle chord_tone_bias as dict or float
    bias_map = cfg.get("chord_tone_bias", {})
    if not isinstance(bias_map, dict):
        bias_map = {"mel": bias_map, "bass": bias_map, "lead": bias_map}

    # Lane-specific parameters with sensible defaults
    lane_params = {
        "mel": {
            "Lmin": cfg.get("mel_min_len_steps", 2),
            "Lmax": cfg.get("mel_max_len_steps", 16),
            "tie_bias": cfg.get("mel_tie_bias", 0.7),
            "vel0": cfg.get("melody_velocity", DEFAULT_VELOCITY),
            "reg_off": 0,
            "ghost_prob": cfg.get("mel_ghost_prob", 0.04),
            "gate_var": cfg.get("mel_gate_var", (0.8, 1.0)),
            "chord_bias": bias_map.get("mel", 0.7),
        },
        "bass": {
            "Lmin": cfg.get("bass_min_len_steps", 16),
            "Lmax": cfg.get("bass_max_len_steps", 64),
            "tie_bias": cfg.get("bass_tie_bias", 0.95),
            "vel0": cfg.get("bass_velocity", 100),
            "reg_off": cfg.get("bass_register_offset_oct", -1) * 12,
            "ghost_prob": 0.0,
            "gate_var": cfg.get("bass_gate_var", (0.9, 1.0)),
            "chord_bias": bias_map.get("bass", 1.0),
        },
        "lead": {
            "Lmin": cfg.get("lead_min_len_steps", 1),
            "Lmax": cfg.get("lead_max_len_steps", 8),
            "tie_bias": cfg.get("lead_tie_bias", 0.45),
            "vel0": cfg.get("lead_velocity", 92),
            "reg_off": cfg.get("lead_register_offset_oct", 1) * 12,
            "ghost_prob": cfg.get("lead_ghost_prob", 0.02),
            "gate_var": cfg.get("lead_gate_var", (0.4, 0.9)),
            "chord_bias": bias_map.get("lead", 0.6),
        },
    }
    
    params = lane_params.get(role, lane_params["mel"])
    Lmin, Lmax = params["Lmin"], params["Lmax"]
    tie_bias, vel0 = params["tie_bias"], params["vel0"]
    reg_off, ghost_prob = params["reg_off"], params["ghost_prob"]
    gate_lo, gate_hi = params["gate_var"]
    chord_bias = params["chord_bias"]

    events: List[Dict[str, Any]] = []
    step_cursor = 0
    Lscale = scale_len(scale_name)

    while step_cursor < steps_total:
        bar_idx = step_cursor // steps_per_bar
        
        # Determine chord tones for current position
        if cfg.get("harmony_enable", True):
            block = max(1, cfg.get("chord_change_every_bars", 2))
            prog = cfg.get("progression_degrees", [0, 5, 3, 4])
            chord_deg = prog[(bar_idx // block) % len(prog)]
            chord_tones = chord_degrees(scale_name, chord_deg, add_seventh=True)
        else:
            chord_tones = []

        # Select degree based on chord bias
        if chord_tones and random.random() < chord_bias:
            if role == "bass":
                base_choice = random.choice([0, 4])  # Root or fifth
                deg = (chord_deg + base_choice) % Lscale
            else:
                deg = random.choice(chord_tones)
        else:
            deg = random.choice(degrees_cycle)

        deg += lane.register * Lscale
        note = scale_note(cfg.get("root", 57), deg, scale_name) + reg_off
        note = clamp_midi_note(note)  # Ensure valid MIDI range

        # Calculate note length - ensure valid range for randint
        Lmin_eff = max(1, Lmin)
        Lmax_eff = max(Lmin_eff + 1, Lmax)  # Ensure Lmax > Lmin
        
        if random.random() < tie_bias:
            length = random.randint(Lmin_eff, Lmax_eff)
        else:
            length = random.randint(Lmin_eff, max(Lmin_eff + 1, int(Lmax_eff * 0.5)))
        length = int(np.clip(length, 1, steps_total - step_cursor))

        # Add velocity variation
        vel = int(np.clip(vel0 + random.randint(-VELOCITY_VARIATION, VELOCITY_VARIATION), 1, 127))
        gate = float(random.uniform(gate_lo, gate_hi))
        
        events.append({
            "step": step_cursor,
            "note": note,
            "len": length,
            "vel": vel,
            "gate": gate
        })

        # Add ghost notes for texture
        if ghost_prob > 0 and random.random() < ghost_prob and step_cursor > 0:
            ghost_len = max(1, length // 4)
            ghost_vel = max(1, int(vel * 0.55))
            ghost_gate = max(0.2, gate * 0.6)
            events.append({
                "step": max(0, step_cursor - 1),
                "note": note,
                "len": ghost_len,
                "vel": ghost_vel,
                "gate": ghost_gate
            })

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
    """MIDI clock generator running in a separate high-priority thread."""
    
    def __init__(self, port: mido.ports.BaseOutput, cfg: Dict[str, Any]):
        self.port = port
        self.cfg = cfg
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the MIDI clock if enabled in config."""
        if self.cfg.get("clock_enable", True) and not self.running:
            self.running = True
            try:
                self.port.send(Message('start'))  # 0xFA
            except Exception as e:
                print(f"[MidiClock] Error sending start: {e}")
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        """Stop the MIDI clock."""
        if not self.running:
            return
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        try:
            self.port.send(Message('stop'))  # 0xFC
        except Exception as e:
            print(f"[MidiClock] Error sending stop: {e}")

    def _run(self) -> None:
        """Clock thread main loop - optimized for stability."""
        bpm = self.cfg.get("bpm", DEFAULT_BPM)
        interval = (60.0 / bpm) / PPQN  # Time between clock ticks
        
        # Use perf_counter for high resolution
        next_tick = time.perf_counter()
        
        while self.running:
            try:
                self.port.send(Message('clock'))  # 0xF8
            except Exception:
                break
            
            # Schedule next tick (absolute time to prevent drift)
            next_tick += interval
            
            # Calculate how long to sleep
            sleep_time = next_tick - time.perf_counter()
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -0.1:
                # We're way behind (>100ms), reset
                next_tick = time.perf_counter()

# ---------------- Playback ----------------

class NoteTracker:
    """Tracks active notes to ensure proper note-off on shutdown."""
    
    def __init__(self):
        self.active_notes: Dict[int, set] = {}  # channel -> set of notes
        self._lock = threading.Lock()
    
    def note_on(self, channel: int, note: int) -> None:
        with self._lock:
            if channel not in self.active_notes:
                self.active_notes[channel] = set()
            self.active_notes[channel].add(note)
    
    def note_off(self, channel: int, note: int) -> None:
        with self._lock:
            if channel in self.active_notes:
                self.active_notes[channel].discard(note)
    
    def get_all_active(self) -> List[tuple]:
        """Returns list of (channel, note) tuples for all active notes."""
        with self._lock:
            result = []
            for ch, notes in self.active_notes.items():
                for note in notes:
                    result.append((ch, note))
            return result


# Global note tracker for cleanup
_note_tracker = NoteTracker()

# Global shutdown event for clean termination
_shutdown_event = threading.Event()


def interruptible_sleep(duration: float) -> bool:
    """Sleep that can be interrupted by shutdown event.
    
    Returns True if sleep completed normally, False if interrupted.
    """
    if duration <= 0:
        return True
    # Use Event.wait() which can be interrupted
    return not _shutdown_event.wait(timeout=duration)


def play_pattern(port: mido.ports.BaseOutput, cfg: Dict[str, Any], patt: Dict[str, Any]) -> bool:
    """Play a complete pattern through the MIDI port.
    
    Returns True if completed normally, False if interrupted by shutdown.
    """
    bpm = cfg.get("bpm", DEFAULT_BPM)
    step_dur = (60.0 / bpm) / 4.0  # 16th notes
    total_steps = cfg.get("steps_per_bar", 16) * cfg.get("bars", 8)
    swing = float(cfg.get("swing", 0.0))
    jitter = cfg.get("jitter_ms", 0) / 1000.0

    # Pre-schedule all events by step
    schedules = {name: [[] for _ in range(total_steps)] for name in ("melody", "bass", "lead")}
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
        # Check for shutdown at the start of each step
        if _shutdown_event.is_set():
            return False
            
        step_start = time.time()
        step_offset = swing * step_dur if (s % 2 == 1) else 0.0

        for name in ("melody", "bass", "lead"):
            if _shutdown_event.is_set():
                return False
                
            channel = ch_map[name]
            for ev in schedules[name][s]:
                note = int(ev["note"])
                vel = int(ev["vel"])
                length_steps = int(ev["len"])
                gate_frac = float(ev.get("gate", 0.95))

                dur = step_dur * length_steps
                micro = random.uniform(-jitter, jitter)
                delay0 = max(0.0, step_offset + micro)

                now = time.time()
                wait = delay0 - (now - step_start)
                if wait > 0:
                    if not interruptible_sleep(wait):
                        return False

                try:
                    port.send(Message('note_on', channel=channel, note=note, velocity=vel))
                    _note_tracker.note_on(channel, note)
                    
                    gate_time = max(MIN_GATE_TIME, dur * gate_frac)
                    if not interruptible_sleep(gate_time):
                        # Send note off before returning on shutdown
                        port.send(Message('note_off', channel=channel, note=note, velocity=0))
                        _note_tracker.note_off(channel, note)
                        return False
                    
                    port.send(Message('note_off', channel=channel, note=note, velocity=0))
                    _note_tracker.note_off(channel, note)
                except Exception as e:
                    print(f"[Playback] MIDI error: {e}")

        elapsed = time.time() - step_start
        rem = step_dur - elapsed
        if rem > 0:
            if not interruptible_sleep(rem):
                return False
    
    return True

# ---------------- Runtime Logic ----------------

def all_notes_off(port: mido.ports.BaseOutput, cfg: Dict[str, Any]) -> None:
    """
    Sends Note-Off for all tracked active notes and performs MIDI panic.
    """
    # First, turn off any tracked active notes
    for channel, note in _note_tracker.get_all_active():
        try:
            port.send(Message('note_off', channel=channel, note=note, velocity=0))
        except Exception:
            pass
    
    # Then do a full panic on all used channels
    channels = [
        cfg["melody_channel"] - 1,
        cfg["bass_channel"] - 1,
        cfg["lead_channel"] - 1,
    ]
    
    for ch in set(channels):  # Use set to avoid duplicates
        try:
            # All Notes Off CC
            port.send(Message('control_change', channel=ch, control=123, value=0))
            # All Sound Off CC
            port.send(Message('control_change', channel=ch, control=120, value=0))
        except Exception as e:
            print(f"[Cleanup] Error sending CC on channel {ch}: {e}")
    
    try:
        port.send(Message('reset'))  # System reset
    except Exception:
        pass  # Not all devices support this


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize configuration values."""
    validated = cfg.copy()
    
    # Ensure BPM is reasonable
    if "bpm" in validated:
        validated["bpm"] = max(20, min(300, int(validated["bpm"])))
    else:
        validated["bpm"] = DEFAULT_BPM
    
    # Ensure bars is positive
    if "bars" in validated:
        validated["bars"] = max(1, int(validated["bars"]))
    else:
        validated["bars"] = 8
    
    # Ensure steps_per_bar is reasonable
    if "steps_per_bar" in validated:
        validated["steps_per_bar"] = max(4, min(64, int(validated["steps_per_bar"])))
    else:
        validated["steps_per_bar"] = 16
    
    # Validate and fix root note - ensure it's a proper MIDI base note
    root = validated.get("root", "A3")
    midi_root = parse_root_note(root)
    # If root is unreasonably low (like 2 for "D"), assume it's a note name and set proper octave
    if isinstance(root, int) and root < 12:
        # AI probably meant a note number within an octave, map to reasonable range
        validated["root"] = 48 + root  # C3 + offset
        print(f"[Config] Root note {root} too low, adjusted to MIDI {validated['root']}")
    elif midi_root < 24:  # Below C1
        validated["root"] = 57  # Default A3
        print(f"[Config] Root note too low, defaulting to A3 (MIDI 57)")
    
    # Validate scale name (case-insensitive, handle compound names like "C major")
    scale = validated.get("scale", validated.get("mode"))
    if scale:
        # Extract just the scale/mode name (remove root note if AI included it)
        # e.g. "C major" -> "major", "D minor" -> "minor", "dorian" -> "dorian"
        scale_parts = scale.lower().split()
        scale_name = scale_parts[-1] if scale_parts else "dorian"  # Take last word
        
        if scale_name not in SCALES:
            print(f"[Config] Unknown scale '{scale}', defaulting to 'dorian'")
            validated["scale"] = "dorian"
        else:
            validated["scale"] = scale_name  # Normalize to just the scale name
    
    # Validate register offsets - limit to reasonable octave range
    for key in ["bass_register_offset_oct", "lead_register_offset_oct"]:
        if key in validated:
            validated[key] = max(-2, min(2, int(validated[key])))
    
    # Validate max_register_shift values
    for key in ["mel_max_register_shift", "bass_max_register_shift", "lead_max_register_shift"]:
        if key in validated:
            validated[key] = max(1, min(3, int(validated[key])))  # Limit to 3 octaves
    
    return validated


def print_sequence_parameters(cfg: Dict[str, Any]) -> None:
    """Prints the main musical parameters from the configuration."""
    scale = get_scale_name(cfg)
    bars = cfg.get("bars", 8)
    tempo = cfg.get("bpm", DEFAULT_BPM)
    root = cfg.get("root", "A")
    
    print("=" * 50)
    print("[Sequencer] Musical Parameters:")
    print(f"  • Scale/Mode: {scale}")
    print(f"  • Root Note:  {root}")
    print(f"  • Tempo:      {tempo} BPM")
    print(f"  • Bars:       {bars}")
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="AI Sequencer - Generative MIDI Sequencer")
    parser.add_argument("--device", type=str, help="MIDI device name hint (e.g. 'OXI', 'IAC', 'Arturia')")
    parser.add_argument("--prompt", type=str, default="prompt.txt", help="Path to prompt file")
    parser.add_argument("--no-ai", action="store_true", help="Skip AI generation, use prompt.txt as config")
    args = parser.parse_args()

    print("[SequencerVCV] AI Sequencer V1 starting...")
    
    # Load configuration
    if args.no_ai:
        print("[SequencerVCV] Loading configuration from prompt.txt (no AI)...")
        user_cfg = load_config_from_prompt(args.prompt)
    else:
        print("[SequencerVCV] Generating composition parameters from prompt.txt using AI...")
        user_cfg = generate_config_from_prompt(args.prompt)
    
    CONFIG = DEFAULT_CONFIG.copy()
    CONFIG.update(user_cfg)
    CONFIG = validate_config(CONFIG)

    # Ensure channel mapping: 1 = Bass, 2 = Melody, 3 = Lead
    CONFIG["bass_channel"] = 1
    CONFIG["melody_channel"] = 2
    CONFIG["lead_channel"] = 3

    # Use device from parameter if set
    if args.device:
        CONFIG["device"] = args.device

    # Display available devices
    print("\n[SequencerVCV] Available MIDI Out devices:")
    outputs = list_outputs()
    if not outputs:
        print("  No MIDI devices found!")
        print("  Please enable IAC Driver or connect a MIDI device.")
        sys.exit(1)
    
    for idx, name in enumerate(outputs):
        print(f"  [{idx}] {name}")
    
    # Optional: Device selection via input, only if no parameter is set
    if not args.device:
        user_hint = input("\nDevice name (Enter for default/IAC): ").strip()
        if user_hint:
            CONFIG["device"] = user_hint
        else:
            CONFIG["device"] = "IAC"

    print(f"\n[SequencerVCV] Active configuration:")
    for key, value in sorted(CONFIG.items()):
        print(f"  {key}: {value}")

    print_sequence_parameters(CONFIG)

    print("\n[SequencerVCV] Opening MIDI Out...")
    try:
        out = open_port(CONFIG["device"])
    except RuntimeError as e:
        print(f"[SequencerVCV] Error: {e}")
        sys.exit(1)
    
    print(f"[SequencerVCV] Sending to: {out.name}")

    clock = MidiClock(out, CONFIG)
    clock.start()

    st = GlobalState(CONFIG)
    mel_evo = int(CONFIG.get("mel_evolve_every_bars", 8))
    bass_evo = int(CONFIG.get("bass_evolve_every_bars", 16))
    lead_evo = int(CONFIG.get("lead_evolve_every_bars", 12))
    bars_passed = 0

    # Setup signal handler for graceful shutdown using global event
    def signal_handler(signum, frame):
        print("\n[SequencerVCV] Shutdown signal received, stopping...")
        _shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("\n[SequencerVCV] Playing... (Press Ctrl+C to stop)")
    
    try:
        while not _shutdown_event.is_set():
            patt = build_pattern(CONFIG, st)
            if not play_pattern(out, CONFIG, patt):
                break  # Interrupted by shutdown
            
            if _shutdown_event.is_set():
                break
                
            bars_passed += CONFIG.get("bars", 8)
            
            # Evolve lanes based on configuration
            if mel_evo > 0 and (bars_passed % mel_evo == 0):
                st.mel.evolve()
                print(f"[Evolution] Melody evolved at bar {bars_passed}")
            if bass_evo > 0 and (bars_passed % bass_evo == 0):
                st.bass.evolve()
                print(f"[Evolution] Bass evolved at bar {bars_passed}")
            if lead_evo > 0 and (bars_passed % lead_evo == 0):
                st.lead.evolve()
                print(f"[Evolution] Lead evolved at bar {bars_passed}")
    except Exception as e:
        print(f"[SequencerVCV] Error during playback: {e}")
    finally:
        print("[SequencerVCV] Cleaning up...")
        clock.stop()
        all_notes_off(out, CONFIG)
        out.close()
        print("[SequencerVCV] Stopped.")


if __name__ == "__main__":
    main()

