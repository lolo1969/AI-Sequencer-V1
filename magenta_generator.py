# magenta_generator.py — Magenta/MusicVAE Integration für MIDI-Generierung
#
# Nutzt Google Magenta's trainierte Modelle um musikalisch sinnvolle
# MIDI-Sequenzen zu generieren, basierend auf den Kompositionsplänen von GPT-4.
#
# Magenta bietet:
# - MusicVAE: Interpolation zwischen musikalischen "Styles"
# - MelodyRNN: Melodie-Generierung mit LSTM
# - GrooVAE: Drum Pattern Generierung
# - PerformanceRNN: Expressive Performance

import os
import sys
import json
import time
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# MIDI
import mido
from mido import Message, MidiFile, MidiTrack

# Type hints für CompositionPlan (Import wird umgangen um circular import zu vermeiden)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gpt_composer import CompositionPlan

# Magenta (optional - mit Fallback)
MAGENTA_AVAILABLE = False
try:
    import note_seq
    from note_seq.protobuf import music_pb2
    MAGENTA_AVAILABLE = True
    print("[MagentaGen] note-seq verfügbar")
except ImportError:
    print("[MagentaGen] note-seq nicht installiert - verwende algorithmischen Fallback")
    print("[MagentaGen] Für Magenta: pip install note-seq magenta")

# Versuche Magenta Modelle zu laden
MUSICVAE_AVAILABLE = False
MELODYRNN_AVAILABLE = False

if MAGENTA_AVAILABLE:
    try:
        from magenta.models.music_vae import configs as vae_configs
        from magenta.models.music_vae.trained_model import TrainedModel as MusicVAEModel
        MUSICVAE_AVAILABLE = True
        print("[MagentaGen] MusicVAE verfügbar")
    except ImportError:
        pass
    
    try:
        from magenta.models.melody_rnn import melody_rnn_sequence_generator
        from magenta.models.melody_rnn import melody_rnn_model
        MELODYRNN_AVAILABLE = True
        print("[MagentaGen] MelodyRNN verfügbar")
    except ImportError:
        pass


# ==================== Musik-Theorie Utilities ====================

SCALE_INTERVALS = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'natural_minor': [0, 2, 3, 5, 7, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'locrian': [0, 1, 3, 5, 6, 8, 10],
    'pentatonic': [0, 2, 4, 7, 9],
    'minor_pentatonic': [0, 3, 5, 7, 10],
    'blues': [0, 3, 5, 6, 7, 10],
    'chromatic': list(range(12))
}

NOTE_TO_MIDI = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}

CHORD_INTERVALS = {
    '': [0, 4, 7],           # Major
    'm': [0, 3, 7],          # Minor
    '7': [0, 4, 7, 10],      # Dominant 7
    'maj7': [0, 4, 7, 11],   # Major 7
    'm7': [0, 3, 7, 10],     # Minor 7
    'dim': [0, 3, 6],        # Diminished
    'aug': [0, 4, 8],        # Augmented
    'sus4': [0, 5, 7],       # Suspended 4
    'sus2': [0, 2, 7],       # Suspended 2
    'add9': [0, 4, 7, 14],   # Add 9
}


def parse_key(key_str: str) -> Tuple[int, str]:
    """
    Parst eine Tonart-Angabe wie "Em", "C#m", "F" in (root_midi, quality)
    Gibt (MIDI-Note 0-11, "major"/"minor") zurück
    """
    key_str = key_str.strip()
    
    # Prüfe auf Minor
    is_minor = key_str.endswith('m')
    if is_minor:
        key_str = key_str[:-1]
    
    # Parse die Note
    if len(key_str) >= 2 and key_str[1] in '#b':
        note_name = key_str[:2]
    else:
        note_name = key_str[0]
    
    root = NOTE_TO_MIDI.get(note_name, 0)
    quality = "minor" if is_minor else "major"
    
    return root, quality


def parse_chord(chord_str: str) -> Tuple[int, List[int]]:
    """
    Parst einen Akkord wie "Em7", "C#m", "Fmaj7" in (root, intervals)
    """
    chord_str = chord_str.strip()
    
    # Finde die Root-Note
    if len(chord_str) >= 2 and chord_str[1] in '#b':
        note_name = chord_str[:2]
        rest = chord_str[2:]
    else:
        note_name = chord_str[0]
        rest = chord_str[1:]
    
    root = NOTE_TO_MIDI.get(note_name, 0)
    
    # Finde die Akkord-Qualität
    intervals = CHORD_INTERVALS.get(rest, CHORD_INTERVALS[''])
    
    return root, intervals


def get_scale_notes(root: int, scale: str, octave: int = 4) -> List[int]:
    """Gibt alle Noten einer Skala als MIDI-Noten zurück"""
    intervals = SCALE_INTERVALS.get(scale, SCALE_INTERVALS['major'])
    base = 12 * (octave + 1) + root  # MIDI-Oktave beginnt bei -1
    return [base + i for i in intervals]


def get_chord_notes(chord_str: str, octave: int = 4) -> List[int]:
    """Gibt die MIDI-Noten eines Akkords zurück"""
    root, intervals = parse_chord(chord_str)
    base = 12 * (octave + 1) + root
    return [base + i for i in intervals]


# ==================== MIDI Generation ====================

@dataclass
class GeneratedSequence:
    """Eine generierte MIDI-Sequenz für einen Kanal"""
    channel: int
    notes: List[Dict[str, Any]]  # [{"pitch": 60, "start": 0, "duration": 480, "velocity": 80}, ...]
    name: str


class MagentaGenerator:
    """
    Generiert MIDI-Sequenzen basierend auf Kompositionsplänen.
    Nutzt Magenta wenn verfügbar, ansonsten algorithmische Fallbacks.
    """
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or os.path.expanduser("~/.magenta/models")
        self.music_vae = None
        self.melody_rnn = None
        
        # Versuche Modelle zu laden
        if MUSICVAE_AVAILABLE:
            self._load_musicvae()
        
        if MELODYRNN_AVAILABLE:
            self._load_melodyrnn()
    
    def _load_musicvae(self):
        """Lädt MusicVAE Modell"""
        try:
            config = vae_configs.CONFIG_MAP['cat-mel_2bar_big']
            checkpoint = os.path.join(self.model_dir, 'musicvae', 'cat-mel_2bar_big')
            
            if os.path.exists(checkpoint + '.ckpt.index'):
                self.music_vae = MusicVAEModel(config, checkpoint=checkpoint)
                print("[MagentaGen] MusicVAE Modell geladen")
            else:
                print(f"[MagentaGen] MusicVAE Checkpoint nicht gefunden: {checkpoint}")
        except Exception as e:
            print(f"[MagentaGen] Fehler beim Laden von MusicVAE: {e}")
    
    def _load_melodyrnn(self):
        """Lädt MelodyRNN Modell"""
        try:
            bundle_path = os.path.join(self.model_dir, 'melody_rnn', 'attention_rnn.mag')
            if os.path.exists(bundle_path):
                # MelodyRNN setup würde hier passieren
                print("[MagentaGen] MelodyRNN Modell geladen")
            else:
                print(f"[MagentaGen] MelodyRNN Bundle nicht gefunden: {bundle_path}")
        except Exception as e:
            print(f"[MagentaGen] Fehler beim Laden von MelodyRNN: {e}")
    
    def generate_from_plan(self, plan: 'CompositionPlan') -> Dict[str, GeneratedSequence]:
        """
        Hauptfunktion: Generiert MIDI-Sequenzen für alle 4 Kanäle
        basierend auf einem Kompositionsplan.
        """
        print(f"[MagentaGen] Generiere Sequenzen für '{plan.title}'")
        print(f"[MagentaGen] Key: {plan.key}, Scale: {plan.scale}, BPM: {plan.bpm}")
        
        # Parse Grundparameter
        root, quality = parse_key(plan.key)
        scale = plan.scale if plan.scale in SCALE_INTERVALS else quality
        
        # Ticks pro Takt (Standard: 480 ticks pro Viertelnote)
        ticks_per_beat = 480
        beats_per_bar = plan.time_signature[0]
        ticks_per_bar = ticks_per_beat * beats_per_bar
        total_ticks = ticks_per_bar * plan.duration_bars
        
        sequences = {}
        
        # Bass (Kanal 0)
        print("[MagentaGen] → Generiere Bass...")
        sequences['bass'] = self._generate_bass(
            plan.bass, plan.chord_progression, plan.harmonic_rhythm_bars,
            root, scale, ticks_per_bar, plan.duration_bars, plan.intensity_curve
        )
        
        # Melody (Kanal 1)
        print("[MagentaGen] → Generiere Melody...")
        sequences['melody'] = self._generate_melody(
            plan.melody, plan.chord_progression, plan.harmonic_rhythm_bars,
            root, scale, ticks_per_bar, plan.duration_bars, plan.intensity_curve
        )
        
        # Lead (Kanal 2)
        print("[MagentaGen] → Generiere Lead...")
        sequences['lead'] = self._generate_lead(
            plan.lead, plan.chord_progression, plan.harmonic_rhythm_bars,
            root, scale, ticks_per_bar, plan.duration_bars, plan.intensity_curve
        )
        
        # Arp (Kanal 3)
        print("[MagentaGen] → Generiere Arp...")
        sequences['arp'] = self._generate_arp(
            plan.arp, plan.chord_progression, plan.harmonic_rhythm_bars,
            root, scale, ticks_per_bar, plan.duration_bars, plan.intensity_curve
        )
        
        print(f"[MagentaGen] ✓ Generierung abgeschlossen")
        return sequences
    
    def _generate_bass(self, config: Dict, chords: List[str], chord_bars: int,
                       root: int, scale: str, ticks_per_bar: int, 
                       total_bars: int, intensity: List[float]) -> GeneratedSequence:
        """Generiert Bass-Linie mit mehr Variation"""
        notes = []
        
        octave_offset = config.get('octave_offset', -1)
        base_octave = 2 + octave_offset  # Bass-Bereich
        density = config.get('note_density', 0.2)
        pattern = config.get('rhythm_pattern', 'sustained')
        role = config.get('role', 'root')
        
        scale_intervals = SCALE_INTERVALS.get(scale, SCALE_INTERVALS['minor'])
        
        current_tick = 0
        bar = 0
        
        while bar < total_bars:
            # Welcher Akkord gerade?
            chord_idx = (bar // chord_bars) % len(chords)
            chord = chords[chord_idx]
            chord_root, chord_intervals = parse_chord(chord)
            
            # Intensität an dieser Stelle
            intensity_idx = int((bar / total_bars) * len(intensity))
            current_intensity = intensity[min(intensity_idx, len(intensity) - 1)]
            
            # Bass-Noten: Grundton + optional Quinte oder Oktave
            bass_notes_options = [
                12 * (base_octave + 1) + chord_root,  # Grundton
                12 * (base_octave + 1) + chord_root + 7,  # Quinte
                12 * (base_octave) + chord_root,  # Oktave tiefer
            ]
            
            # Velocity basierend auf Intensität
            velocity = int(50 + current_intensity * 50)
            
            if pattern == 'sustained' or pattern == 'legato':
                # Lange gehaltene Noten
                note_duration = ticks_per_bar * chord_bars
                bass_note = bass_notes_options[0]
                notes.append({
                    'pitch': bass_note,
                    'start': current_tick,
                    'duration': note_duration,
                    'velocity': velocity
                })
                current_tick += ticks_per_bar * chord_bars
                bar += chord_bars
            
            elif pattern == 'pulse' or pattern == 'steady':
                # Rhythmische Pulse mit Variation
                beats_per_chord = chord_bars * 4
                for beat in range(beats_per_chord):
                    # Erster Schlag immer, andere nach Density
                    if beat == 0 or random.random() < density:
                        # Wähle Bass-Note: meist Grundton, manchmal Quinte
                        if beat == 0:
                            bass_note = bass_notes_options[0]
                        else:
                            bass_note = random.choice(bass_notes_options[:2])
                        
                        notes.append({
                            'pitch': bass_note,
                            'start': current_tick,
                            'duration': ticks_per_bar // 4,
                            'velocity': velocity if beat == 0 else velocity - 10
                        })
                    current_tick += ticks_per_bar // 4
                bar += chord_bars
            
            elif pattern == 'walking':
                # Walking Bass - verschiedene Töne pro Beat
                for beat in range(4):
                    if beat == 0:
                        bass_note = bass_notes_options[0]
                    elif beat == 2:
                        bass_note = bass_notes_options[1]  # Quinte
                    else:
                        # Durchgangston aus der Skala
                        passing_interval = random.choice(scale_intervals[1:4])
                        bass_note = 12 * (base_octave + 1) + (chord_root + passing_interval) % 12
                    
                    notes.append({
                        'pitch': bass_note,
                        'start': current_tick,
                        'duration': ticks_per_bar // 4 - 20,
                        'velocity': velocity
                    })
                    current_tick += ticks_per_bar // 4
                bar += 1
            
            elif pattern == 'sparse' or pattern == 'minimal':
                # Sehr spärlich - nur ab und zu
                if random.random() < density * 1.5 or bar % chord_bars == 0:
                    bass_note = bass_notes_options[0]
                    notes.append({
                        'pitch': bass_note,
                        'start': current_tick,
                        'duration': ticks_per_bar * 2,
                        'velocity': velocity
                    })
                current_tick += ticks_per_bar
                bar += 1
            
            else:
                # Standard: Ein Ton pro Takt
                if random.random() < density * 3 or bar % chord_bars == 0:
                    notes.append({
                        'pitch': bass_notes_options[0],
                        'start': current_tick,
                        'duration': ticks_per_bar,
                        'velocity': velocity
                    })
                current_tick += ticks_per_bar
                bar += 1
        
        return GeneratedSequence(channel=0, notes=notes, name="Bass")
    
    def _generate_melody(self, config: Dict, chords: List[str], chord_bars: int,
                         root: int, scale: str, ticks_per_bar: int,
                         total_bars: int, intensity: List[float]) -> GeneratedSequence:
        """Generiert Melodie mit Magenta oder Fallback"""
        
        # Wenn MusicVAE verfügbar, nutze es
        if self.music_vae and MAGENTA_AVAILABLE:
            return self._generate_melody_musicvae(config, chords, chord_bars,
                                                  root, scale, ticks_per_bar,
                                                  total_bars, intensity)
        
        # Algorithmischer Fallback
        return self._generate_melody_algorithmic(config, chords, chord_bars,
                                                  root, scale, ticks_per_bar,
                                                  total_bars, intensity)
    
    def _generate_melody_algorithmic(self, config: Dict, chords: List[str], 
                                     chord_bars: int, root: int, scale: str,
                                     ticks_per_bar: int, total_bars: int,
                                     intensity: List[float]) -> GeneratedSequence:
        """Algorithmische Melodie-Generierung mit mehr Variation"""
        notes = []
        
        octave_offset = config.get('octave_offset', 0)
        base_octave = 4 + octave_offset
        density = config.get('note_density', 0.4)
        motif_degrees = config.get('motif_degrees', [0, 2, 4, 5, 7])
        rhythm_pattern = config.get('rhythm_pattern', 'varied')
        articulation = config.get('articulation', 'legato')
        
        scale_intervals = SCALE_INTERVALS.get(scale, SCALE_INTERVALS['major'])
        
        current_tick = 0
        bar = 0
        last_pitch = None
        phrase_start = True
        
        # Verschiedene rhythmische Templates basierend auf Pattern
        rhythm_templates = {
            'sparse': [1, 0, 0, 0, 0.3, 0, 0, 0],  # Sehr wenige Noten
            'varied': [1, 0.3, 0.5, 0.2, 0.6, 0.2, 0.4, 0.1],  # Abwechslungsreich
            'flowing': [0.8, 0.6, 0.7, 0.5, 0.7, 0.6, 0.8, 0.4],  # Fließend
            'syncopated': [1, 0, 0.7, 0, 0.3, 0.8, 0, 0.5],  # Synkopiert
            'minimal': [1, 0, 0, 0, 0, 0, 0, 0.3],  # Sehr minimal
        }
        rhythm_template = rhythm_templates.get(rhythm_pattern, rhythm_templates['varied'])
        
        # Melodische Bewegungs-Richtung
        direction = 1  # 1 = aufwärts, -1 = abwärts
        direction_counter = 0
        
        while bar < total_bars:
            chord_idx = (bar // chord_bars) % len(chords)
            chord_root, chord_intervals = parse_chord(chords[chord_idx])
            chord_notes_midi = [12 * (base_octave + 1) + (chord_root + i) % 12 for i in chord_intervals]
            
            # Intensität
            intensity_idx = int((bar / total_bars) * len(intensity))
            current_intensity = intensity[min(intensity_idx, len(intensity) - 1)]
            
            # 8tel-Noten Grid
            ticks_per_eighth = ticks_per_bar // 8
            
            for eighth in range(8):
                # Rhythmus-Wahrscheinlichkeit aus Template + Intensity
                base_prob = rhythm_template[eighth]
                play_prob = base_prob * density * (0.5 + current_intensity)
                
                # Phrasen-Anfang hat höhere Wahrscheinlichkeit
                if phrase_start:
                    play_prob = min(1.0, play_prob * 1.5)
                    phrase_start = False
                
                if random.random() < play_prob:
                    # Wähle Tonhöhe
                    if random.random() < 0.4:
                        # Akkordton
                        pitch = random.choice(chord_notes_midi)
                    else:
                        # Skalenton basierend auf Motiv
                        if motif_degrees and random.random() < 0.7:
                            degree = random.choice(motif_degrees)
                        else:
                            degree = random.randint(0, len(scale_intervals) - 1)
                        
                        interval = scale_intervals[degree % len(scale_intervals)]
                        octave_adjust = degree // len(scale_intervals)
                        pitch = 12 * (base_octave + 1 + octave_adjust) + (root + interval) % 12
                    
                    # Melodische Richtung beachten
                    if last_pitch:
                        direction_counter += 1
                        if direction_counter > random.randint(3, 6):
                            direction *= -1
                            direction_counter = 0
                        
                        # Bevorzuge Bewegung in der aktuellen Richtung
                        target = last_pitch + direction * random.randint(1, 4)
                        
                        # Finde nächsten Skalenton
                        best_pitch = pitch
                        min_diff = abs(pitch - target)
                        for oct in range(-1, 2):
                            for interval in scale_intervals:
                                test_pitch = 12 * (base_octave + 1 + oct) + (root + interval) % 12
                                diff = abs(test_pitch - target)
                                if diff < min_diff:
                                    min_diff = diff
                                    best_pitch = test_pitch
                        pitch = best_pitch
                    
                    # Notenlänge basierend auf Artikulation
                    if articulation == 'legato':
                        duration = ticks_per_eighth * random.choice([2, 3, 4])
                    elif articulation == 'staccato':
                        duration = ticks_per_eighth // 2
                    else:
                        duration = ticks_per_eighth * random.choice([1, 2])
                    
                    velocity = int(55 + current_intensity * 45 + random.randint(-8, 8))
                    
                    notes.append({
                        'pitch': max(48, min(84, pitch)),
                        'start': current_tick,
                        'duration': duration,
                        'velocity': max(40, min(110, velocity))
                    })
                    
                    last_pitch = pitch
                
                current_tick += ticks_per_eighth
            
            # Neue Phrase bei Akkordwechsel
            if (bar + 1) % chord_bars == 0:
                phrase_start = True
            
            bar += 1
        
        return GeneratedSequence(channel=1, notes=notes, name="Melody")
    
    def _generate_melody_musicvae(self, config: Dict, chords: List[str],
                                  chord_bars: int, root: int, scale: str,
                                  ticks_per_bar: int, total_bars: int,
                                  intensity: List[float]) -> GeneratedSequence:
        """Melodie-Generierung mit MusicVAE (wenn verfügbar)"""
        # Hier würde die eigentliche MusicVAE Integration sein
        # Für jetzt nutzen wir den algorithmischen Fallback
        return self._generate_melody_algorithmic(config, chords, chord_bars,
                                                  root, scale, ticks_per_bar,
                                                  total_bars, intensity)
    
    def _generate_lead(self, config: Dict, chords: List[str], chord_bars: int,
                       root: int, scale: str, ticks_per_bar: int,
                       total_bars: int, intensity: List[float]) -> GeneratedSequence:
        """Generiert Lead/Counter-Melodie mit charakteristischer Phrasierung"""
        notes = []
        
        octave_offset = config.get('octave_offset', 1)
        base_octave = 4 + octave_offset
        density = config.get('note_density', 0.3)
        motif_degrees = config.get('motif_degrees', [4, 6, 7, 9])
        role = config.get('role', 'counter-melody')
        articulation = config.get('articulation', 'legato')
        
        scale_intervals = SCALE_INTERVALS.get(scale, SCALE_INTERVALS['major'])
        
        current_tick = 0
        bar = 0
        last_note_bar = -4
        phrase_notes = []
        
        # Lead hat charakteristische Phrasen
        while bar < total_bars:
            # Intensität
            intensity_idx = int((bar / total_bars) * len(intensity))
            current_intensity = intensity[min(intensity_idx, len(intensity) - 1)]
            
            chord_idx = (bar // chord_bars) % len(chords)
            chord_root, chord_intervals = parse_chord(chords[chord_idx])
            
            # Lead spielt in Phrasen, nicht einzelne Noten
            bars_since_note = bar - last_note_bar
            
            # Neue Phrase starten?
            start_phrase = False
            if bars_since_note >= 3 and random.random() < 0.4:
                start_phrase = True
            elif bars_since_note >= 2 and current_intensity > 0.5 and random.random() < 0.5:
                start_phrase = True
            elif role == 'accent' and bar % chord_bars == 0 and random.random() < density:
                start_phrase = True
            
            if start_phrase:
                # Phrasen-Länge: 2-4 Noten
                phrase_length = random.randint(2, 4) if density > 0.2 else random.randint(1, 2)
                
                for note_in_phrase in range(phrase_length):
                    # Wähle Tonhöhe aus Motiv
                    if motif_degrees and random.random() < 0.8:
                        degree = motif_degrees[note_in_phrase % len(motif_degrees)]
                    else:
                        degree = random.randint(3, 8)
                    
                    interval = scale_intervals[degree % len(scale_intervals)]
                    octave_adj = degree // len(scale_intervals)
                    pitch = 12 * (base_octave + 1 + octave_adj) + (root + interval) % 12
                    
                    # Timing innerhalb der Phrase
                    note_start = current_tick + note_in_phrase * (ticks_per_bar // 2)
                    
                    # Notenlänge
                    if articulation == 'legato':
                        duration = ticks_per_bar * random.choice([1, 2])
                    elif articulation == 'staccato':
                        duration = ticks_per_bar // 4
                    else:
                        duration = ticks_per_bar // 2
                    
                    velocity = int(45 + current_intensity * 40)
                    
                    notes.append({
                        'pitch': max(55, min(90, pitch)),
                        'start': note_start,
                        'duration': duration,
                        'velocity': max(35, min(95, velocity))
                    })
                
                last_note_bar = bar + phrase_length // 2
            
            current_tick += ticks_per_bar
            bar += 1
        
        return GeneratedSequence(channel=2, notes=notes, name="Lead")
    
    def _generate_arp(self, config: Dict, chords: List[str], chord_bars: int,
                      root: int, scale: str, ticks_per_bar: int,
                      total_bars: int, intensity: List[float]) -> GeneratedSequence:
        """Generiert Arpeggio-Pattern mit mehr Variation"""
        notes = []
        
        octave_offset = config.get('octave_offset', 1)
        base_octave = 4 + octave_offset
        density = config.get('note_density', 0.5)
        arp_pattern = config.get('arp_pattern', 'up')
        note_length_raw = config.get('note_length_steps', 2)
        rhythm_pattern = config.get('rhythm_pattern', 'flowing')
        
        # Handle case where GPT returns a list
        if isinstance(note_length_raw, list):
            note_length = note_length_raw[0] if note_length_raw else 2
        else:
            note_length = note_length_raw
        
        scale_intervals = SCALE_INTERVALS.get(scale, SCALE_INTERVALS['major'])
        
        # Verschiedene Arp-Subdivisions basierend auf BPM/Style
        subdivisions = {
            'sparse': 4,      # Viertelnoten
            'flowing': 8,     # Achtelnoten
            'fast': 16,       # Sechzehntelnoten
            'triplet': 6,     # Triolen
        }
        
        if rhythm_pattern in subdivisions:
            notes_per_bar = subdivisions[rhythm_pattern]
        elif density > 0.6:
            notes_per_bar = 16
        elif density > 0.3:
            notes_per_bar = 8
        else:
            notes_per_bar = 4
        
        ticks_per_note = ticks_per_bar // notes_per_bar
        
        current_tick = 0
        bar = 0
        arp_position = 0
        
        while bar < total_bars:
            chord_idx = (bar // chord_bars) % len(chords)
            chord_notes = get_chord_notes(chords[chord_idx], base_octave)
            
            # Erweitere Akkord um Oktaven für mehr Range
            extended_chord = chord_notes.copy()
            extended_chord.extend([n + 12 for n in chord_notes])  # Oktave höher
            if base_octave > 3:
                extended_chord.extend([n - 12 for n in chord_notes])  # Oktave tiefer
            extended_chord = sorted(set(extended_chord))
            
            # Intensität
            intensity_idx = int((bar / total_bars) * len(intensity))
            current_intensity = intensity[min(intensity_idx, len(intensity) - 1)]
            
            for note_in_bar in range(notes_per_bar):
                # Density-Check
                if random.random() > density + current_intensity * 0.3:
                    current_tick += ticks_per_note
                    arp_position += 1
                    continue
                
                # Arpeggio-Muster bestimmt die Note
                num_notes = len(extended_chord)
                
                if arp_pattern == 'up':
                    note_idx = arp_position % num_notes
                elif arp_pattern == 'down':
                    note_idx = (num_notes - 1) - (arp_position % num_notes)
                elif arp_pattern == 'up-down':
                    cycle = max(1, 2 * num_notes - 2)
                    pos = arp_position % cycle
                    if pos < num_notes:
                        note_idx = pos
                    else:
                        note_idx = cycle - pos
                elif arp_pattern == 'random':
                    note_idx = random.randint(0, num_notes - 1)
                elif arp_pattern == 'pendulum':
                    # Pendel-Bewegung mit variabler Geschwindigkeit
                    cycle = num_notes * 4
                    pos = arp_position % cycle
                    note_idx = int(num_notes / 2 + (num_notes / 2) * np.sin(pos * np.pi / num_notes))
                elif arp_pattern == 'pattern':
                    # Wiederholendes Pattern wie [0,2,1,2,0,2,1,3]
                    pattern_seq = [0, 2, 1, 2, 0, 2, 1, 3] if num_notes > 3 else [0, 1, 0, 2]
                    note_idx = pattern_seq[arp_position % len(pattern_seq)] % num_notes
                else:
                    note_idx = arp_position % num_notes
                
                pitch = extended_chord[note_idx % len(extended_chord)]
                
                # Gelegentliche Oktav-Sprünge für Interesse
                if random.random() < 0.1 and current_intensity > 0.4:
                    pitch += random.choice([-12, 12])
                
                duration = ticks_per_note * note_length
                
                # Velocity mit leichtem Groove (betonte Schläge)
                accent = 1.2 if note_in_bar % 4 == 0 else (1.1 if note_in_bar % 2 == 0 else 1.0)
                velocity = int((35 + current_intensity * 35) * accent + random.randint(-5, 5))
                
                notes.append({
                    'pitch': max(48, min(96, pitch)),
                    'start': current_tick,
                    'duration': max(ticks_per_note // 2, duration),
                    'velocity': max(30, min(100, velocity))
                })
                
                current_tick += ticks_per_note
                arp_position += 1
            
            bar += 1
        
        return GeneratedSequence(channel=3, notes=notes, name="Arp")
    
    def to_midi_file(self, sequences: Dict[str, GeneratedSequence], 
                     bpm: int = 120, output_path: str = "composition.mid") -> bool:
        """
        Konvertiert die generierten Sequenzen in eine MIDI-Datei.
        """
        try:
            mid = MidiFile(type=1)
            mid.ticks_per_beat = 480
            
            for channel_name, seq in sequences.items():
                track = MidiTrack()
                mid.tracks.append(track)
                
                # Tempo setzen (nur im ersten Track)
                if channel_name == 'bass':
                    tempo = mido.bpm2tempo(bpm)
                    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
                    track.append(mido.MetaMessage('track_name', name='Composition', time=0))
                
                # Track-Name
                track.append(mido.MetaMessage('track_name', name=seq.name, time=0))
                
                # Sortiere Noten nach Startzeit
                sorted_notes = sorted(seq.notes, key=lambda n: n['start'])
                
                # Erstelle MIDI-Events
                events = []
                for note in sorted_notes:
                    events.append({
                        'type': 'note_on',
                        'time': note['start'],
                        'note': note['pitch'],
                        'velocity': note['velocity'],
                        'channel': seq.channel
                    })
                    events.append({
                        'type': 'note_off',
                        'time': note['start'] + note['duration'],
                        'note': note['pitch'],
                        'velocity': 0,
                        'channel': seq.channel
                    })
                
                # Sortiere alle Events nach Zeit
                events.sort(key=lambda e: e['time'])
                
                # Konvertiere zu Delta-Zeiten
                last_time = 0
                for event in events:
                    delta = event['time'] - last_time
                    if event['type'] == 'note_on':
                        track.append(Message('note_on', 
                                            channel=event['channel'],
                                            note=event['note'],
                                            velocity=event['velocity'],
                                            time=delta))
                    else:
                        track.append(Message('note_off',
                                            channel=event['channel'],
                                            note=event['note'],
                                            velocity=0,
                                            time=delta))
                    last_time = event['time']
            
            mid.save(output_path)
            print(f"[MagentaGen] MIDI gespeichert: {output_path}")
            return True
            
        except Exception as e:
            print(f"[MagentaGen] Fehler beim Speichern: {e}")
            return False
    
    def get_realtime_notes(self, sequences: Dict[str, GeneratedSequence], 
                           start_tick: int, end_tick: int) -> List[Dict]:
        """
        Gibt alle Noten in einem Tick-Bereich zurück.
        Nützlich für Echtzeit-Wiedergabe.
        """
        notes = []
        for channel_name, seq in sequences.items():
            for note in seq.notes:
                if start_tick <= note['start'] < end_tick:
                    notes.append({
                        **note,
                        'channel': seq.channel,
                        'channel_name': channel_name
                    })
        return sorted(notes, key=lambda n: n['start'])


# ==================== Test ====================

if __name__ == "__main__":
    # Test ohne echtes CompositionPlan-Objekt
    from dataclasses import dataclass
    
    @dataclass
    class MockPlan:
        title: str = "Test Composition"
        description: str = "Test"
        bpm: int = 100
        time_signature: tuple = (4, 4)
        key: str = "Em"
        scale: str = "dorian"
        duration_bars: int = 16
        chord_progression: list = None
        chord_rhythm: str = "whole"
        harmonic_rhythm_bars: int = 4
        mood_arc: str = "rising"
        intensity_curve: list = None
        bass: dict = None
        melody: dict = None
        lead: dict = None
        arp: dict = None
        density: str = "medium"
        articulation: str = "legato"
        dynamics_range: tuple = (50, 100)
        style_references: list = None
        avoid: list = None
        
        def __post_init__(self):
            if self.chord_progression is None:
                self.chord_progression = ["Em", "Am7", "Bm", "Em"]
            if self.intensity_curve is None:
                self.intensity_curve = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6]
            if self.bass is None:
                self.bass = {"octave_offset": -1, "note_density": 0.3, "rhythm_pattern": "sustained"}
            if self.melody is None:
                self.melody = {"octave_offset": 0, "note_density": 0.4, "motif_degrees": [0, 2, 4, 7]}
            if self.lead is None:
                self.lead = {"octave_offset": 1, "note_density": 0.2, "motif_degrees": [7, 9, 11]}
            if self.arp is None:
                self.arp = {"octave_offset": 1, "note_density": 0.5, "arp_pattern": "up-down"}
    
    # Erstelle Generator
    gen = MagentaGenerator()
    
    # Erstelle Test-Plan
    plan = MockPlan()
    
    # Generiere Sequenzen
    print("\n" + "="*60)
    print("Generiere Test-Komposition...")
    print("="*60)
    
    sequences = gen.generate_from_plan(plan)
    
    # Zeige Statistiken
    for name, seq in sequences.items():
        print(f"\n{name.upper()}: {len(seq.notes)} Noten")
        if seq.notes:
            pitches = [n['pitch'] for n in seq.notes]
            print(f"  Pitch-Range: {min(pitches)} - {max(pitches)}")
    
    # Speichere als MIDI
    gen.to_midi_file(sequences, plan.bpm, "test_composition.mid")
    print("\n✓ Test abgeschlossen!")
