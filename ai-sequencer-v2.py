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
    print("\n\n[Sequencer] 🛑 Shutdown signal empfangen...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ==================== MIDI Utilities ====================

def list_midi_outputs() -> List[str]:
    """Listet verfügbare MIDI-Ausgänge"""
    try:
        return mido.get_output_names()
    except Exception as e:
        print(f"[MIDI] Fehler: {e}")
        return []


def open_midi_output(device_hint: str) -> Optional[mido.ports.BaseOutput]:
    """Öffnet einen MIDI-Ausgang"""
    names = list_midi_outputs()
    
    if not names:
        print("[MIDI] ❌ Keine MIDI-Ausgänge gefunden!")
        print("[MIDI] Stellen Sie sicher, dass ein MIDI-Gerät angeschlossen ist")
        print("[MIDI] oder aktivieren Sie den IAC-Treiber (macOS)")
        return None
    
    print(f"\n[MIDI] Verfügbare Ausgänge:")
    for i, name in enumerate(names):
        print(f"  [{i}] {name}")
    
    # Finde besten Match
    best_match = None
    for name in names:
        if device_hint.lower() in name.lower():
            best_match = name
            break
    
    if not best_match:
        best_match = names[0]
        print(f"[MIDI] Kein Match für '{device_hint}', verwende: {best_match}")
    else:
        print(f"[MIDI] ✓ Gefunden: {best_match}")
    
    try:
        port = mido.open_output(best_match)
        return port
    except Exception as e:
        print(f"[MIDI] ❌ Fehler beim Öffnen: {e}")
        return None


def all_notes_off(port: mido.ports.BaseOutput):
    """Schaltet alle Noten auf allen Kanälen aus"""
    if port:
        for channel in range(4):
            for note in range(128):
                port.send(Message('note_off', channel=channel, note=note, velocity=0))
            port.send(Message('control_change', channel=channel, control=123, value=0))


# ==================== Evolution Engine ====================

class EvolutionEngine:
    """
    Entwickelt die generierten Patterns in Echtzeit weiter.
    Kleine, organische Veränderungen über die Zeit.
    """
    
    def __init__(self, sequences: Dict[str, GeneratedSequence], bpm: int):
        self.sequences = sequences
        self.bpm = bpm
        self.ticks_per_beat = 480
        self.evolution_rate = 0.05  # Wahrscheinlichkeit für Änderung pro Note
        
        # Statistiken
        self.mutations_count = 0
        
    def evolve(self, intensity: float = 0.5):
        """
        Wendet kleine Mutationen auf die Sequenzen an.
        Intensity: 0.0 = keine Änderungen, 1.0 = viele Änderungen
        """
        for channel_name, seq in self.sequences.items():
            for note in seq.notes:
                if random.random() < self.evolution_rate * intensity:
                    mutation = random.choice(['pitch', 'velocity', 'timing', 'duration'])
                    
                    if mutation == 'pitch':
                        # Kleine Tonhöhen-Änderung (max 2 Halbtöne)
                        shift = random.choice([-2, -1, 1, 2])
                        note['pitch'] = max(24, min(108, note['pitch'] + shift))
                        self.mutations_count += 1
                    
                    elif mutation == 'velocity':
                        # Leichte Velocity-Änderung
                        shift = random.randint(-10, 10)
                        note['velocity'] = max(30, min(127, note['velocity'] + shift))
                        self.mutations_count += 1
                    
                    elif mutation == 'timing':
                        # Minimale Timing-Verschiebung (Humanisierung)
                        shift = random.randint(-20, 20)
                        note['start'] = max(0, note['start'] + shift)
                        self.mutations_count += 1
                    
                    elif mutation == 'duration':
                        # Notenlängen-Variation
                        factor = random.uniform(0.8, 1.2)
                        note['duration'] = max(60, int(note['duration'] * factor))
                        self.mutations_count += 1
    
    def add_note(self, channel_name: str, base_pitch: int, intensity: float):
        """Fügt gelegentlich eine neue Note hinzu"""
        if random.random() < 0.02 * intensity:  # 2% Chance bei voller Intensität
            seq = self.sequences.get(channel_name)
            if seq and seq.notes:
                # Basiere neue Note auf existierenden
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
    Spielt die generierten Sequenzen in Echtzeit ab
    und wendet die Evolution Engine an.
    """
    
    def __init__(self, midi_port: mido.ports.BaseOutput, 
                 sequences: Dict[str, GeneratedSequence],
                 plan: CompositionPlan):
        self.port = midi_port
        self.sequences = sequences
        self.plan = plan
        self.bpm = plan.bpm
        self.evolution = EvolutionEngine(sequences, self.bpm)
        
        # Timing
        self.ticks_per_beat = 480
        self.seconds_per_tick = 60.0 / (self.bpm * self.ticks_per_beat)
        
        # State
        self.current_tick = 0
        self.loop_length = plan.duration_bars * self.ticks_per_beat * plan.time_signature[0]
        self.active_notes: Dict[int, List[Dict]] = {0: [], 1: [], 2: [], 3: []}
        
        # Intensitätskurve
        self.intensity_curve = plan.intensity_curve
        self.intensity_index = 0
        
        # MIDI Clock
        self.send_clock = True
        self.clock_interval = 60.0 / (self.bpm * PPQN)
        
    def get_current_intensity(self) -> float:
        """Gibt die aktuelle Intensität zurück"""
        if not self.intensity_curve:
            return 0.5
        
        progress = self.current_tick / self.loop_length
        idx = int(progress * len(self.intensity_curve))
        idx = min(idx, len(self.intensity_curve) - 1)
        return self.intensity_curve[idx]
    
    def send_midi_clock_pulse(self):
        """Sendet einen MIDI Clock Pulse"""
        if self.port and self.send_clock:
            self.port.send(Message('clock'))
    
    def note_on(self, channel: int, pitch: int, velocity: int):
        """Sendet Note-On"""
        if self.port:
            self.port.send(Message('note_on', channel=channel, note=pitch, velocity=velocity))
            self.active_notes[channel].append({'pitch': pitch, 'start_time': time.time()})
    
    def note_off(self, channel: int, pitch: int):
        """Sendet Note-Off"""
        if self.port:
            self.port.send(Message('note_off', channel=channel, note=pitch, velocity=0))
            self.active_notes[channel] = [n for n in self.active_notes[channel] if n['pitch'] != pitch]
    
    def run(self):
        """Hauptschleife für Live-Wiedergabe"""
        global running
        
        print(f"\n[Sequencer] 🎵 Starte Wiedergabe...")
        print(f"[Sequencer] BPM: {self.bpm}")
        print(f"[Sequencer] Loop-Länge: {self.plan.duration_bars} Takte")
        print(f"[Sequencer] Key: {self.plan.key} {self.plan.scale}")
        print(f"[Sequencer] Drücke Ctrl+C zum Beenden\n")
        
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
                    
                    # Loop zurücksetzen
                    if self.current_tick >= self.loop_length:
                        self.current_tick = 0
                        loop_count += 1
                        intensity = self.get_current_intensity()
                        
                        # Evolution anwenden
                        self.evolution.evolve(intensity)
                        
                        print(f"\r[Loop {loop_count}] Intensity: {intensity:.1%} | "
                              f"Mutations: {self.evolution.mutations_count}", end="", flush=True)
                    
                    # Noten triggern
                    self._process_notes()
                
                # CPU schonen
                time.sleep(0.0001)
                
        except Exception as e:
            print(f"\n[Sequencer] ❌ Fehler: {e}")
        
        finally:
            # Cleanup
            print("\n[Sequencer] Beende...")
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
        Stufe 1: Prompt → Kompositionsplan
        """
        print("\n" + "="*60)
        print("📝 STUFE 1: GPT-4 Komposition")
        print("="*60)
        
        self.plan = self.composer.compose(prompt, duration_bars)
        
        if self.plan:
            print(f"\n✓ Kompositionsplan erstellt:")
            print(f"  Titel: {self.plan.title}")
            print(f"  Beschreibung: {self.plan.description[:80]}...")
        
        return self.plan
    
    def generate_midi(self) -> Dict[str, GeneratedSequence]:
        """
        Stufe 2: Kompositionsplan → MIDI-Sequenzen
        """
        if not self.plan:
            raise ValueError("Kein Kompositionsplan vorhanden. Zuerst compose_from_prompt() aufrufen.")
        
        print("\n" + "="*60)
        print("🎹 STUFE 2: MIDI-Generierung (Magenta)")
        print("="*60)
        
        self.sequences = self.generator.generate_from_plan(self.plan)
        
        # Statistiken
        total_notes = sum(len(seq.notes) for seq in self.sequences.values())
        print(f"\n✓ {total_notes} Noten generiert")
        
        return self.sequences
    
    def save_midi(self, output_path: str = "composition.mid") -> bool:
        """Speichert die generierten Sequenzen als MIDI-Datei"""
        if not self.sequences:
            print("[Error] Keine Sequenzen vorhanden")
            return False
        
        return self.generator.to_midi_file(self.sequences, self.plan.bpm, output_path)
    
    def save_plan(self, output_path: str = "composition_plan.json"):
        """Speichert den Kompositionsplan als JSON"""
        if self.plan:
            self.composer.save_plan(self.plan, output_path)
    
    def load_plan(self, plan_path: str) -> CompositionPlan:
        """Lädt einen gespeicherten Kompositionsplan"""
        with open(plan_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.plan = self.composer._dict_to_plan(data)
        print(f"[Sequencer] Plan geladen: {self.plan.title}")
        return self.plan
    
    def play_live(self):
        """
        Stufe 3: Live-Wiedergabe mit Evolution
        """
        if not self.sequences or not self.plan:
            raise ValueError("Keine Sequenzen vorhanden. Zuerst generate_midi() aufrufen.")
        
        print("\n" + "="*60)
        print("🎵 STUFE 3: Live-Wiedergabe")
        print("="*60)
        
        # MIDI öffnen
        port = open_midi_output(self.device_hint)
        if not port:
            print("[Error] Konnte MIDI nicht öffnen")
            return
        
        try:
            sequencer = LiveSequencer(port, self.sequences, self.plan)
            sequencer.run()
        finally:
            if port:
                all_notes_off(port)
                port.close()
    
    def run_full_pipeline(self, prompt: str, duration_bars: int = 32, 
                          live: bool = True, save_midi: bool = True):
        """
        Führt die komplette Pipeline aus:
        Prompt → GPT-4 → Magenta → Live-Wiedergabe
        """
        print("\n" + "="*60)
        print("🎼 AI SEQUENCER V2 — Prompt-to-Music Pipeline")
        print("="*60)
        print(f"\n📖 Prompt: \"{prompt[:100]}...\"" if len(prompt) > 100 else f"\n📖 Prompt: \"{prompt}\"")
        
        # Stufe 1: Komposition
        self.compose_from_prompt(prompt, duration_bars)
        
        if not self.plan:
            print("[Error] Komposition fehlgeschlagen")
            return
        
        # Stufe 2: MIDI generieren
        self.generate_midi()
        
        # MIDI speichern
        if save_midi:
            filename = f"{self.plan.title.replace(' ', '_').lower()}.mid"
            self.save_midi(filename)
            self.save_plan(f"{self.plan.title.replace(' ', '_').lower()}_plan.json")
        
        # Stufe 3: Live-Wiedergabe
        if live:
            self.play_live()


# ==================== CLI ====================

def load_prompt_from_file(filepath: str) -> str:
    """Lädt Prompt aus einer Datei"""
    if not os.path.exists(filepath):
        return ""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Entferne Konfigurationszeilen (mit :)
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
Beispiele:
  python3 ai-sequencer-v2.py --text "Düsterer Ambient mit aufsteigenden Arpeggios"
  python3 ai-sequencer-v2.py --prompt prompt.txt --device "OXI ONE"
  python3 ai-sequencer-v2.py --generate-only --text "Minimal Techno" --output track.mid
        """
    )
    
    parser.add_argument('--device', '-d', type=str, default='IAC',
                        help='MIDI-Gerät (z.B. "IAC", "OXI ONE")')
    
    parser.add_argument('--text', '-t', type=str, default=None,
                        help='Direkter Prompt-Text')
    
    parser.add_argument('--prompt', '-p', type=str, default='prompt.txt',
                        help='Pfad zur Prompt-Datei')
    
    parser.add_argument('--plan', type=str, default=None,
                        help='Pfad zu gespeichertem Kompositionsplan (JSON)')
    
    parser.add_argument('--bars', '-b', type=int, default=32,
                        help='Anzahl Takte (Standard: 32)')
    
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='MIDI-Ausgabedatei')
    
    parser.add_argument('--generate-only', action='store_true',
                        help='Nur MIDI generieren, keine Live-Wiedergabe')
    
    parser.add_argument('--no-save', action='store_true',
                        help='MIDI nicht speichern')
    
    parser.add_argument('--list-devices', action='store_true',
                        help='Zeigt verfügbare MIDI-Geräte')
    
    args = parser.parse_args()
    
    # MIDI-Geräte auflisten
    if args.list_devices:
        print("\nVerfügbare MIDI-Ausgänge:")
        for name in list_midi_outputs():
            print(f"  • {name}")
        return
    
    # Sequencer initialisieren
    sequencer = AISequencerV2(device_hint=args.device)
    
    # Plan laden oder neu generieren
    if args.plan:
        sequencer.load_plan(args.plan)
        sequencer.generate_midi()
        
        # MIDI speichern wenn gewünscht
        if args.output:
            sequencer.save_midi(args.output)
        
        # Live-Wiedergabe wenn nicht --generate-only
        if not args.generate_only:
            sequencer.play_live()
        
        return
    else:
        # Prompt bestimmen
        prompt = args.text or load_prompt_from_file(args.prompt)
        
        if not prompt:
            print("❌ Kein Prompt angegeben!")
            print("   Verwende --text oder erstelle prompt.txt")
            return
        
        # Pipeline ausführen
        sequencer.run_full_pipeline(
            prompt=prompt,
            duration_bars=args.bars,
            live=not args.generate_only,
            save_midi=not args.no_save
        )
    
    # Ausgabe speichern wenn gewünscht
    if args.output and sequencer.sequences:
        sequencer.save_midi(args.output)


if __name__ == "__main__":
    main()
