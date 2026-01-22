#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai-sequencer-v2.py — Prompt-to-Music AI Sequencer

Das Konzept: Beschreibe deine Musik wie eine Geschichte,
und die KI komponiert sie für dich.

ARCHITEKTUR:
┌─────────────────────────────────────────────────────────────────┐
│  STUFE 1: GPT-4 als Musik-Regisseur                            │
│  ───────────────────────────────────                            │
│  Prompt → GPT-4 → Strukturierter Kompositionsplan               │
│  (Akkorde, Melodie-Kontur, Dynamik, Stil)                       │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  STUFE 2: Magenta für musikalische MIDI-Generierung            │
│  ──────────────────────────────────────────────────             │
│  Kompositionsplan → MusicVAE/Algorithmic → MIDI-Noten           │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  STUFE 3: Evolution Engine (Live-Entwicklung)                  │
│  ─────────────────────────────────────────────                  │
│  MIDI-Patterns → Echtzeit-Evolution → OXI ONE / VCV Rack        │
└─────────────────────────────────────────────────────────────────┘

USAGE:
    # Mit Prompt-Datei
    python3 ai-sequencer-v2.py --device "OXI ONE" --prompt prompt.txt
    
    # Mit direktem Prompt
    python3 ai-sequencer-v2.py --device "IAC" --text "Düsterer Ambient, aufsteigend"
    
    # Nur MIDI generieren (ohne Live-Wiedergabe)
    python3 ai-sequencer-v2.py --generate-only --output my_composition.mid
    
    # Mit vorhandenem Kompositionsplan
    python3 ai-sequencer-v2.py --plan composition_plan.json

AUTHOR: AI Sequencer Project
DATE: 2026
"""

import os
import sys
import time
import json
import random
import signal
import argparse
import threading
from typing import Dict, List, Any, Optional
from dataclasses import asdict

import numpy as np
import mido
from mido import Message

# Lokale Module
from gpt_composer import GPTComposer, CompositionPlan
from magenta_generator import MagentaGenerator, GeneratedSequence

# ==================== Konstanten ====================

PPQN = 24  # Pulses per quarter note für MIDI Clock
DEFAULT_BPM = 120
MIN_GATE_TIME = 0.03

# ==================== Globaler State ====================

running = True
midi_out = None


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\n\n[Sequencer] 🛑 Shutdown signal received...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ==================== MIDI Utilities ====================

def list_midi_outputs() -> List[str]:
    """Lists available MIDI outputs"""
    try:
        return mido.get_output_names()
    except Exception as e:
        print(f"[MIDI] Error: {e}")
        return []


def open_midi_output(device_hint: str) -> Optional[mido.ports.BaseOutput]:
    """Opens a MIDI output"""
    names = list_midi_outputs()
    
    if not names:
        print("[MIDI] ❌ No MIDI outputs found!")
        print("[MIDI] Make sure a MIDI device is connected")
        print("[MIDI] or enable the IAC driver (macOS)")
        return None
    
    print(f"\n[MIDI] Available outputs:")
    for i, name in enumerate(names):
        print(f"  [{i}] {name}")
    
    # Find best match
    best_match = None
    for name in names:
        if device_hint.lower() in name.lower():
            best_match = name
            break
    
    if not best_match:
        best_match = names[0]
        print(f"[MIDI] No match for '{device_hint}', using: {best_match}")
    else:
        print(f"[MIDI] ✓ Found: {best_match}")
    
    try:
        port = mido.open_output(best_match)
        return port
    except Exception as e:
        print(f"[MIDI] ❌ Error opening: {e}")
        return None


def all_notes_off(port: mido.ports.BaseOutput):
    """Turns off all notes on all channels"""
    if port:
        for channel in range(4):
            for note in range(128):
                port.send(Message('note_off', channel=channel, note=note, velocity=0))
            port.send(Message('control_change', channel=channel, control=123, value=0))


# ==================== Evolution Engine ====================

class EvolutionEngine:
    """
    Evolves the generated patterns in real-time.
    Small, organic changes over time.
    """
    
    def __init__(self, sequences: Dict[str, GeneratedSequence], bpm: int):
        self.sequences = sequences
        self.bpm = bpm
        self.ticks_per_beat = 480
        self.evolution_rate = 0.05  # Probability of change per note
        
        # Statistics
        self.mutations_count = 0
        
    def evolve(self, intensity: float = 0.5):
        """
        Applies small mutations to the sequences.
        Intensity: 0.0 = no changes, 1.0 = many changes
        """
        for channel_name, seq in self.sequences.items():
            for note in seq.notes:
                if random.random() < self.evolution_rate * intensity:
                    mutation = random.choice(['pitch', 'velocity', 'timing', 'duration'])
                    
                    if mutation == 'pitch':
                        # Small pitch change (max 2 semitones)
                        shift = random.choice([-2, -1, 1, 2])
                        note['pitch'] = max(24, min(108, note['pitch'] + shift))
                        self.mutations_count += 1
                    
                    elif mutation == 'velocity':
                        # Slight velocity change
                        shift = random.randint(-10, 10)
                        note['velocity'] = max(30, min(127, note['velocity'] + shift))
                        self.mutations_count += 1
                    
                    elif mutation == 'timing':
                        # Minimal timing shift (humanization)
                        shift = random.randint(-20, 20)
                        note['start'] = max(0, note['start'] + shift)
                        self.mutations_count += 1
                    
                    elif mutation == 'duration':
                        # Note length variation
                        factor = random.uniform(0.8, 1.2)
                        note['duration'] = max(60, int(note['duration'] * factor))
                        self.mutations_count += 1
    
    def add_note(self, channel_name: str, base_pitch: int, intensity: float):
        """Occasionally adds a new note"""
        if random.random() < 0.02 * intensity:  # 2% chance at full intensity
            seq = self.sequences.get(channel_name)
            if seq and seq.notes:
                # Base new note on existing ones
                reference = random.choice(seq.notes)
                new_note = {
                    'pitch': reference['pitch'] + random.choice([-7, -5, -3, 3, 5, 7]),
                    'start': reference['start'] + random.randint(0, 1920),  # Irgendwo in der Nähe
                    'duration': reference['duration'],
                    'velocity': reference['velocity']
                }
                new_note['pitch'] = max(24, min(108, new_note['pitch']))
                seq.notes.append(new_note)
    
    def remove_note(self, channel_name: str, intensity: float):
        """Entfernt gelegentlich eine Note"""
        if random.random() < 0.01 * (1 - intensity):  # Mehr Entfernung bei niedriger Intensität
            seq = self.sequences.get(channel_name)
            if seq and len(seq.notes) > 4:  # Mindestens 4 Noten behalten
                idx = random.randint(0, len(seq.notes) - 1)
                seq.notes.pop(idx)


# ==================== Live Sequencer ====================

class LiveSequencer:
    """
    Plays the generated sequences in real-time
    and applies the Evolution Engine.
    """
    
    def __init__(self, midi_port: mido.ports.BaseOutput, 
                 sequences: Dict[str, GeneratedSequence],
                 plan: CompositionPlan,
                 loops: int = 0):
        self.port = midi_port
        self.sequences = sequences
        self.plan = plan
        self.bpm = plan.bpm
        self.max_loops = loops  # 0 = infinite, otherwise number of loops
        self.evolution = EvolutionEngine(sequences, self.bpm)
        
        # Timing
        self.ticks_per_beat = 480
        self.seconds_per_tick = 60.0 / (self.bpm * self.ticks_per_beat)
        
        # State
        self.current_tick = 0
        self.loop_length = plan.duration_bars * self.ticks_per_beat * plan.time_signature[0]
        self.active_notes: Dict[int, List[Dict]] = {0: [], 1: [], 2: [], 3: []}
        
        # Intensity curve
        self.intensity_curve = plan.intensity_curve
        self.intensity_index = 0
        
        # MIDI Clock
        self.send_clock = True
        self.clock_interval = 60.0 / (self.bpm * PPQN)
        
    def get_current_intensity(self) -> float:
        """Returns the current intensity"""
        if not self.intensity_curve:
            return 0.5
        
        progress = self.current_tick / self.loop_length
        idx = int(progress * len(self.intensity_curve))
        idx = min(idx, len(self.intensity_curve) - 1)
        return self.intensity_curve[idx]
    
    def send_midi_clock_pulse(self):
        """Sends a MIDI clock pulse"""
        if self.port and self.send_clock:
            self.port.send(Message('clock'))
    
    def note_on(self, channel: int, pitch: int, velocity: int):
        """Sends Note-On"""
        if self.port:
            self.port.send(Message('note_on', channel=channel, note=pitch, velocity=velocity))
            self.active_notes[channel].append({'pitch': pitch, 'start_time': time.time()})
    
    def note_off(self, channel: int, pitch: int):
        """Sends Note-Off"""
        if self.port:
            self.port.send(Message('note_off', channel=channel, note=pitch, velocity=0))
            self.active_notes[channel] = [n for n in self.active_notes[channel] if n['pitch'] != pitch]
    
    def run(self):
        """Main loop for live playback"""
        global running
        
        print(f"\n[Sequencer] 🎵 Starting playback...")
        print(f"[Sequencer] BPM: {self.bpm}")
        print(f"[Sequencer] Loop length: {self.plan.duration_bars} bars")
        print(f"[Sequencer] Key: {self.plan.key} {self.plan.scale}")
        print(f"[Sequencer] Loops: {'Infinite' if self.max_loops == 0 else self.max_loops}")
        print(f"[Sequencer] Press Ctrl+C to stop\n")
        
        # Send MIDI Start
        if self.port:
            self.port.send(Message('start'))
        
        last_clock_time = time.time()
        last_tick_time = time.time()
        loop_count = 0
        
        try:
            while running:
                current_time = time.time()
                
                # MIDI Clock
                if current_time - last_clock_time >= self.clock_interval:
                    self.send_midi_clock_pulse()
                    last_clock_time = current_time
                
                # Tick-Update
                if current_time - last_tick_time >= self.seconds_per_tick:
                    self.current_tick += 1
                    last_tick_time = current_time
                    
                    # Reset loop
                    if self.current_tick >= self.loop_length:
                        loop_count += 1
                        
                        # With limited loops: stop when reached
                        if self.max_loops > 0 and loop_count >= self.max_loops:
                            print(f"\n[Sequencer] ✓ Playback finished after {loop_count} loop(s)")
                            running = False
                            break
                        
                        self.current_tick = 0
                        intensity = self.get_current_intensity()
                        
                        # Apply evolution
                        self.evolution.evolve(intensity)
                        
                        if self.max_loops > 0:
                            print(f"\r[Loop {loop_count}/{self.max_loops}] Intensity: {intensity:.1%} | "
                                  f"Mutations: {self.evolution.mutations_count}", end="", flush=True)
                        else:
                            print(f"\r[Loop {loop_count}] Intensity: {intensity:.1%} | "
                                  f"Mutations: {self.evolution.mutations_count}", end="", flush=True)
                    
                    # Noten triggern
                    self._process_notes()
                
                # Save CPU
                time.sleep(0.0001)
                
        except Exception as e:
            print(f"\n[Sequencer] ❌ Error: {e}")
        
        finally:
            # Cleanup
            print("\n[Sequencer] Stopping...")
            if self.port:
                self.port.send(Message('stop'))
                all_notes_off(self.port)
    
    def _process_notes(self):
        """Verarbeitet Noten für den aktuellen Tick"""
        for channel_name, seq in self.sequences.items():
            for note in seq.notes:
                # Note-On
                if note['start'] == self.current_tick % self.loop_length:
                    self.note_on(seq.channel, note['pitch'], note['velocity'])
                
                # Note-Off
                end_tick = (note['start'] + note['duration']) % self.loop_length
                if end_tick == self.current_tick % self.loop_length:
                    self.note_off(seq.channel, note['pitch'])


# ==================== Main Pipeline ====================

class AISequencerV2:
    """
    Hauptklasse: Orchestriert den gesamten Workflow
    Prompt → GPT-4 → Magenta → Live-Sequencer
    """
    
    def __init__(self, device_hint: str = "IAC"):
        self.device_hint = device_hint
        self.composer = GPTComposer()
        self.generator = MagentaGenerator()
        self.plan: Optional[CompositionPlan] = None
        self.sequences: Optional[Dict[str, GeneratedSequence]] = None
        
    def compose_from_prompt(self, prompt: str, duration_bars: int = 32) -> CompositionPlan:
        """
        Stage 1: Prompt → Composition Plan
        """
        print("\n" + "="*60)
        print("📝 STAGE 1: GPT-4 Composition")
        print("="*60)
        
        self.plan = self.composer.compose(prompt, duration_bars)
        
        if self.plan:
            print(f"\n✓ Composition plan created:")
            print(f"  Title: {self.plan.title}")
            print(f"  Description: {self.plan.description[:80]}...")
        
        return self.plan
    
    def generate_midi(self) -> Dict[str, GeneratedSequence]:
        """
        Stage 2: Composition Plan → MIDI Sequences
        """
        if not self.plan:
            raise ValueError("No composition plan available. Call compose_from_prompt() first.")
        
        print("\n" + "="*60)
        print("🎹 STAGE 2: MIDI Generation (Magenta)")
        print("="*60)
        
        self.sequences = self.generator.generate_from_plan(self.plan)
        
        # Statistics
        total_notes = sum(len(seq.notes) for seq in self.sequences.values())
        print(f"\n✓ {total_notes} notes generated")
        
        return self.sequences
    
    def save_midi(self, output_path: str = "composition.mid") -> bool:
        """Saves the generated sequences as a MIDI file"""
        if not self.sequences:
            print("[Error] No sequences available")
            return False
        
        return self.generator.to_midi_file(self.sequences, self.plan.bpm, output_path)
    
    def save_plan(self, output_path: str = "composition_plan.json"):
        """Saves the composition plan as JSON"""
        if self.plan:
            self.composer.save_plan(self.plan, output_path)
    
    def load_plan(self, plan_path: str) -> CompositionPlan:
        """Loads a saved composition plan"""
        with open(plan_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.plan = self.composer._dict_to_plan(data)
        print(f"[Sequencer] Plan loaded: {self.plan.title}")
        return self.plan
    
    def play_live(self, loops: int = 0):
        """
        Stage 3: Live playback with Evolution
        
        Args:
            loops: Number of loops (0 = infinite)
        """
        if not self.sequences or not self.plan:
            raise ValueError("No sequences available. Call generate_midi() first.")
        
        print("\n" + "="*60)
        print("🎵 STAGE 3: Live Playback")
        print("="*60)
        
        # Open MIDI
        port = open_midi_output(self.device_hint)
        if not port:
            print("[Error] Could not open MIDI")
            return
        
        try:
            sequencer = LiveSequencer(port, self.sequences, self.plan, loops=loops)
            sequencer.run()
        finally:
            if port:
                all_notes_off(port)
                port.close()
    
    def run_full_pipeline(self, prompt: str, duration_bars: int = 32, 
                          live: bool = True, save_midi: bool = True,
                          loops: int = 0):
        """
        Runs the complete pipeline:
        Prompt → GPT-4 → Magenta → Live Playback
        
        Args:
            prompt: Music description
            duration_bars: Number of bars
            live: Enable live playback
            save_midi: Save MIDI file
            loops: Number of loops (0 = infinite)
        """
        print("\n" + "="*60)
        print("🎼 AI SEQUENCER V2 — Prompt-to-Music Pipeline")
        print("="*60)
        print(f"\n📖 Prompt: \"{prompt[:100]}...\"" if len(prompt) > 100 else f"\n📖 Prompt: \"{prompt}\"")
        
        # Stage 1: Composition
        self.compose_from_prompt(prompt, duration_bars)
        
        if not self.plan:
            print("[Error] Composition failed")
            return
        
        # Stage 2: Generate MIDI
        self.generate_midi()
        
        # Save MIDI
        if save_midi:
            # Ensure output/ exists
            os.makedirs('output', exist_ok=True)
            base_name = self.plan.title.replace(' ', '_').lower()
            filename = f"output/{base_name}.mid"
            self.save_midi(filename)
            self.save_plan(f"output/{base_name}_plan.json")
        
        # Stufe 3: Live-Wiedergabe
        if live:
            self.play_live(loops=loops)


# ==================== CLI ====================

def load_prompt_from_file(filepath: str) -> str:
    """Loads prompt from a file"""
    if not os.path.exists(filepath):
        return ""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove configuration lines (with :)
    lines = []
    for line in content.split('\n'):
        line = line.strip()
        if line and not line.startswith('#') and ':' not in line[:20]:
            lines.append(line)
    
    return ' '.join(lines).strip()


def main():
    parser = argparse.ArgumentParser(
        description="AI Sequencer V2 — Prompt-to-Music",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 ai-sequencer-v2.py --text "Dark ambient with ascending arpeggios"
  python3 ai-sequencer-v2.py --prompt prompt.txt --device "OXI ONE"
  python3 ai-sequencer-v2.py --generate-only --text "Minimal Techno" --output track.mid
        """
    )
    
    parser.add_argument('--device', '-d', type=str, default='IAC',
                        help='MIDI device (e.g. "IAC", "OXI ONE")')
    
    parser.add_argument('--text', '-t', type=str, default=None,
                        help='Direct prompt text')
    
    parser.add_argument('--prompt', '-p', type=str, default='prompt.txt',
                        help='Path to prompt file')
    
    parser.add_argument('--plan', type=str, default=None,
                        help='Path to saved composition plan (JSON)')
    
    parser.add_argument('--bars', '-b', type=int, default=32,
                        help='Number of bars (default: 32)')
    
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='MIDI output file')
    
    parser.add_argument('--generate-only', action='store_true',
                        help='Only generate MIDI, no live playback')
    
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save MIDI')
    
    parser.add_argument('--loops', '-l', type=int, default=0,
                        help='Number of loops (0 = infinite, default: 0)')
    
    parser.add_argument('--list-devices', action='store_true',
                        help='Show available MIDI devices')
    
    args = parser.parse_args()
    
    # List MIDI devices
    if args.list_devices:
        print("\nAvailable MIDI outputs:")
        for name in list_midi_outputs():
            print(f"  • {name}")
        return
    
    # Initialize sequencer
    sequencer = AISequencerV2(device_hint=args.device)
    
    # Load plan or generate new
    if args.plan:
        sequencer.load_plan(args.plan)
        sequencer.generate_midi()
        
        # Save MIDI if requested
        if args.output:
            sequencer.save_midi(args.output)
        
        # Live playback if not --generate-only
        if not args.generate_only:
            sequencer.play_live(loops=args.loops)
        
        return
    else:
        # Determine prompt
        prompt = args.text or load_prompt_from_file(args.prompt)
        
        if not prompt:
            print("❌ No prompt specified!")
            print("   Use --text or create prompt.txt")
            return
        
        # Run pipeline
        sequencer.run_full_pipeline(
            prompt=prompt,
            duration_bars=args.bars,
            live=not args.generate_only,
            save_midi=not args.no_save,
            loops=args.loops
        )
    
    # Save output if requested
    if args.output and sequencer.sequences:
        sequencer.save_midi(args.output)


if __name__ == "__main__":
    main()
