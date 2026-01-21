# gpt_composer.py — GPT-4 als Musik-Regisseur
#
# GPT-4 interpretiert natürlichsprachliche Prompts und generiert
# strukturierte Kompositionsanweisungen für die MIDI-Generierung.
#
# Das Konzept: Wie beim Schreiben einer Geschichte beschreibst du
# Stimmung, Stil und Handlung - GPT-4 erstellt einen "Kompositionsplan"

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# OpenAI
try:
    from openai import OpenAI
except ImportError:
    print("[GPTComposer] OpenAI nicht installiert. Bitte: pip install openai")
    OpenAI = None


class MoodArc(Enum):
    """Emotionale Entwicklung über die Zeit"""
    STATIC = "static"           # Gleichbleibend
    RISING = "rising"           # Aufbauend
    FALLING = "falling"         # Abklingend
    WAVE = "wave"               # Auf und Ab
    CLIMAX = "climax"           # Aufbau → Höhepunkt → Ende
    TENSION = "tension"         # Spannung aufbauen
    RELEASE = "release"         # Auflösung


@dataclass
class CompositionPlan:
    """
    Strukturierter Kompositionsplan, generiert von GPT-4.
    Dies ist die "Partitur" die an Magenta/MusicVAE weitergegeben wird.
    """
    # Grundlegende Parameter
    title: str
    description: str
    bpm: int
    time_signature: tuple  # (4, 4), (3, 4), etc.
    key: str               # "C", "Em", "F#m", etc.
    scale: str             # "major", "minor", "dorian", etc.
    duration_bars: int     # Länge in Takten
    
    # Harmonische Struktur
    chord_progression: List[str]         # ["Em", "Am", "Bm", "Em"]
    chord_rhythm: str                    # "whole", "half", "quarter"
    harmonic_rhythm_bars: int            # Akkordwechsel alle X Takte
    
    # Emotionale Entwicklung
    mood_arc: str                        # "rising", "falling", "wave", etc.
    intensity_curve: List[float]         # [0.2, 0.3, 0.5, 0.7, 0.9, 0.6]
    
    # Kanal-spezifische Anweisungen
    bass: Dict[str, Any]
    melody: Dict[str, Any]
    lead: Dict[str, Any]
    arp: Dict[str, Any]
    
    # Textur und Arrangement
    density: str                         # "sparse", "medium", "dense"
    articulation: str                    # "legato", "staccato", "mixed"
    dynamics_range: tuple                # (40, 100) = pp to ff
    
    # Stilistische Referenzen (für Magenta)
    style_references: List[str]          # ["ambient", "minimal", "electronic"]
    avoid: List[str]                     # ["drums", "vocals", "distortion"]


class GPTComposer:
    """
    Verwendet GPT-4 um natürlichsprachliche Prompts in strukturierte
    Kompositionspläne zu übersetzen.
    """
    
    SYSTEM_PROMPT = """Du bist ein erfahrener Komponist und Musiktheoretiker.
Deine Aufgabe ist es, kreative Beschreibungen in präzise musikalische Anweisungen zu übersetzen.

Du erstellst einen strukturierten JSON-Kompositionsplan mit folgenden Elementen:

1. **Grundparameter**: BPM, Tonart, Taktart, Skala
2. **Harmonische Struktur**: Akkordfolge, harmonischer Rhythmus
3. **Emotionaler Bogen**: Wie entwickelt sich das Stück über die Zeit?
4. **4 Kanäle**: Bass, Melody, Lead, Arp - jeweils mit:
   - register: Oktavlage ("low", "mid", "high")
   - note_density: Notendichte (0.0-1.0)
   - rhythm_pattern: Rhythmisches Muster
   - articulation: Spielweise
   - role: Funktion im Arrangement

5. **Stilistische Anweisungen**: Was soll vermieden werden, Referenzen

Antworte NUR mit validem JSON. Keine Erklärungen, kein Markdown."""

    EXAMPLE_OUTPUT = """{
  "title": "Nebelschwaden",
  "description": "Langsam aufsteigender Ambient mit ätherischen Texturen",
  "bpm": 72,
  "time_signature": [4, 4],
  "key": "Em",
  "scale": "dorian",
  "duration_bars": 32,
  "chord_progression": ["Em", "Am7", "Bm7", "Em"],
  "chord_rhythm": "whole",
  "harmonic_rhythm_bars": 4,
  "mood_arc": "rising",
  "intensity_curve": [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
  "bass": {
    "register": "low",
    "note_density": 0.1,
    "rhythm_pattern": "sustained",
    "articulation": "legato",
    "role": "foundation",
    "octave_offset": -1,
    "motif_degrees": [0, 4],
    "note_length_bars": 2,
    "dynamics": "pp-mp"
  },
  "melody": {
    "register": "mid",
    "note_density": 0.2,
    "rhythm_pattern": "sparse",
    "articulation": "legato",
    "role": "theme",
    "octave_offset": 0,
    "motif_degrees": [0, 2, 4, 7, 9],
    "note_length_bars": 0.5,
    "dynamics": "p-mf"
  },
  "lead": {
    "register": "high",
    "note_density": 0.15,
    "rhythm_pattern": "punctuated",
    "articulation": "legato",
    "role": "counter-melody",
    "octave_offset": 1,
    "motif_degrees": [7, 9, 11],
    "note_length_bars": 1,
    "dynamics": "pp-p"
  },
  "arp": {
    "register": "mid-high",
    "note_density": 0.4,
    "rhythm_pattern": "flowing",
    "articulation": "staccato",
    "role": "texture",
    "octave_offset": 1,
    "motif_degrees": [0, 4, 7, 11],
    "arp_pattern": "up-down",
    "note_length_steps": 2,
    "dynamics": "ppp-p"
  },
  "density": "sparse",
  "articulation": "legato",
  "dynamics_range": [30, 80],
  "style_references": ["ambient", "minimal", "stars of the lid", "eluvium"],
  "avoid": ["drums", "percussion", "sudden changes", "dissonance"]
}"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        
        if self.api_key and OpenAI:
            self.client = OpenAI(api_key=self.api_key)
            print(f"[GPTComposer] Initialisiert mit {model}")
        else:
            print("[GPTComposer] Kein API-Key oder OpenAI nicht verfügbar")
    
    def compose(self, prompt: str, duration_bars: int = 32) -> Optional[CompositionPlan]:
        """
        Hauptfunktion: Nimmt einen kreativen Prompt und erstellt einen Kompositionsplan.
        
        Beispiel-Prompts:
        - "Düstere Ambient-Landschaft, langsam anschwellend wie aufziehender Nebel"
        - "Energetischer Minimal Techno, hypnotisch und treibend"
        - "Melancholische Klavierballade im Stil von Nils Frahm"
        """
        if not self.client:
            print("[GPTComposer] Kein OpenAI Client - verwende Fallback")
            return self._fallback_composition(prompt, duration_bars)
        
        try:
            print(f"[GPTComposer] Interpretiere Prompt: '{prompt[:50]}...'")
            
            user_message = f"""Erstelle einen Kompositionsplan für folgende Beschreibung:

"{prompt}"

Länge: {duration_bars} Takte

Beachte:
- Wir haben 4 MIDI-Kanäle: Bass, Melody, Lead, Arp
- Die Musik soll sich organisch entwickeln
- Berücksichtige die emotionale Beschreibung
- Wähle passende Tonart und Skala basierend auf der Stimmung
- Gib für jeden Kanal spezifische Anweisungen

Beispiel-Ausgabe (nur als Referenz für das Format):
{self.EXAMPLE_OUTPUT}

Jetzt erstelle deinen eigenen Kompositionsplan basierend auf dem Prompt."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.8,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON
            # Entferne mögliche Markdown-Blöcke
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            
            plan_dict = json.loads(content)
            
            print(f"[GPTComposer] ✓ Kompositionsplan erstellt: '{plan_dict.get('title', 'Untitled')}'")
            print(f"[GPTComposer]   Key: {plan_dict.get('key')} {plan_dict.get('scale')}")
            print(f"[GPTComposer]   BPM: {plan_dict.get('bpm')}")
            print(f"[GPTComposer]   Mood: {plan_dict.get('mood_arc')}")
            print(f"[GPTComposer]   Chords: {' → '.join(plan_dict.get('chord_progression', []))}")
            
            return self._dict_to_plan(plan_dict)
            
        except json.JSONDecodeError as e:
            print(f"[GPTComposer] JSON Parse Error: {e}")
            print(f"[GPTComposer] Raw response: {content[:500]}")
            return self._fallback_composition(prompt, duration_bars)
            
        except Exception as e:
            print(f"[GPTComposer] Error: {e}")
            return self._fallback_composition(prompt, duration_bars)
    
    def _dict_to_plan(self, d: Dict) -> CompositionPlan:
        """Konvertiert Dictionary zu CompositionPlan"""
        return CompositionPlan(
            title=d.get("title", "Untitled"),
            description=d.get("description", ""),
            bpm=d.get("bpm", 120),
            time_signature=tuple(d.get("time_signature", [4, 4])),
            key=d.get("key", "C"),
            scale=d.get("scale", "major"),
            duration_bars=d.get("duration_bars", 32),
            chord_progression=d.get("chord_progression", ["C", "Am", "F", "G"]),
            chord_rhythm=d.get("chord_rhythm", "whole"),
            harmonic_rhythm_bars=d.get("harmonic_rhythm_bars", 4),
            mood_arc=d.get("mood_arc", "static"),
            intensity_curve=d.get("intensity_curve", [0.5] * 8),
            bass=d.get("bass", {}),
            melody=d.get("melody", {}),
            lead=d.get("lead", {}),
            arp=d.get("arp", {}),
            density=d.get("density", "medium"),
            articulation=d.get("articulation", "mixed"),
            dynamics_range=tuple(d.get("dynamics_range", [60, 100])),
            style_references=d.get("style_references", []),
            avoid=d.get("avoid", [])
        )
    
    def _fallback_composition(self, prompt: str, duration_bars: int) -> CompositionPlan:
        """
        Fallback wenn GPT-4 nicht verfügbar ist.
        Verwendet einfache Keyword-Analyse für grundlegende Parameter.
        """
        print("[GPTComposer] Verwende Fallback-Komposition")
        
        prompt_lower = prompt.lower()
        
        # Einfache Keyword-basierte Analyse
        bpm = 120
        scale = "major"
        key = "C"
        mood_arc = "static"
        density = "medium"
        
        # Tempo-Keywords
        if any(w in prompt_lower for w in ["slow", "langsam", "meditative", "ambient"]):
            bpm = random.randint(60, 80)
        elif any(w in prompt_lower for w in ["fast", "schnell", "energetic", "driving"]):
            bpm = random.randint(130, 160)
        elif any(w in prompt_lower for w in ["moderate", "mittel"]):
            bpm = random.randint(100, 120)
        
        # Stimmungs-Keywords
        if any(w in prompt_lower for w in ["dark", "düster", "minor", "sad", "melancholic"]):
            scale = "minor"
            key = random.choice(["Am", "Em", "Dm", "Bm"])
        elif any(w in prompt_lower for w in ["bright", "happy", "major", "uplifting"]):
            scale = "major"
            key = random.choice(["C", "G", "D", "F"])
        elif any(w in prompt_lower for w in ["modal", "dorian", "exotic"]):
            scale = "dorian"
            key = random.choice(["D", "E", "A"])
        
        # Entwicklungs-Keywords
        if any(w in prompt_lower for w in ["rising", "building", "aufsteigend", "crescendo"]):
            mood_arc = "rising"
        elif any(w in prompt_lower for w in ["falling", "fading", "decrescendo"]):
            mood_arc = "falling"
        elif any(w in prompt_lower for w in ["wave", "ebb", "flow"]):
            mood_arc = "wave"
        
        # Dichte-Keywords
        if any(w in prompt_lower for w in ["sparse", "minimal", "space"]):
            density = "sparse"
        elif any(w in prompt_lower for w in ["dense", "full", "rich"]):
            density = "dense"
        
        return CompositionPlan(
            title="Generated Composition",
            description=prompt[:100],
            bpm=bpm,
            time_signature=(4, 4),
            key=key,
            scale=scale,
            duration_bars=duration_bars,
            chord_progression=self._generate_chord_progression(key, scale),
            chord_rhythm="whole",
            harmonic_rhythm_bars=4,
            mood_arc=mood_arc,
            intensity_curve=self._generate_intensity_curve(mood_arc, 8),
            bass={
                "register": "low",
                "note_density": 0.2,
                "rhythm_pattern": "sustained",
                "articulation": "legato",
                "role": "foundation",
                "octave_offset": -1,
                "motif_degrees": [0, 4],
                "dynamics": "mp"
            },
            melody={
                "register": "mid",
                "note_density": 0.4,
                "rhythm_pattern": "melodic",
                "articulation": "legato",
                "role": "theme",
                "octave_offset": 0,
                "motif_degrees": [0, 2, 4, 5, 7],
                "dynamics": "mf"
            },
            lead={
                "register": "high",
                "note_density": 0.3,
                "rhythm_pattern": "punctuated",
                "articulation": "legato",
                "role": "counter-melody",
                "octave_offset": 1,
                "motif_degrees": [7, 9, 11],
                "dynamics": "p"
            },
            arp={
                "register": "mid-high",
                "note_density": 0.5,
                "rhythm_pattern": "arpeggio",
                "articulation": "staccato",
                "role": "texture",
                "octave_offset": 1,
                "motif_degrees": [0, 4, 7, 11],
                "dynamics": "p"
            },
            density=density,
            articulation="legato" if "ambient" in prompt_lower else "mixed",
            dynamics_range=(40, 90),
            style_references=[],
            avoid=[]
        )
    
    def _generate_chord_progression(self, key: str, scale: str) -> List[str]:
        """Generiert eine passende Akkordfolge"""
        # Vereinfachte Akkordfolgen
        if "m" in key or scale == "minor":
            progressions = [
                ["Am", "Dm", "Em", "Am"],
                ["Em", "Am", "Bm", "Em"],
                ["Dm", "Gm", "Am", "Dm"],
                ["Bm", "Em", "F#m", "Bm"]
            ]
        else:
            progressions = [
                ["C", "Am", "F", "G"],
                ["G", "Em", "C", "D"],
                ["D", "Bm", "G", "A"],
                ["F", "Dm", "Bb", "C"]
            ]
        
        import random
        return random.choice(progressions)
    
    def _generate_intensity_curve(self, mood_arc: str, points: int) -> List[float]:
        """Generiert eine Intensitätskurve basierend auf dem Mood Arc"""
        import numpy as np
        
        if mood_arc == "rising":
            return list(np.linspace(0.2, 0.9, points))
        elif mood_arc == "falling":
            return list(np.linspace(0.9, 0.2, points))
        elif mood_arc == "wave":
            x = np.linspace(0, 2 * np.pi, points)
            return list(0.5 + 0.4 * np.sin(x))
        elif mood_arc == "climax":
            # Aufbau bis 70%, dann Höhepunkt, dann Abfall
            mid = points // 2
            rise = list(np.linspace(0.3, 0.9, mid))
            fall = list(np.linspace(0.9, 0.4, points - mid))
            return rise + fall
        else:  # static
            return [0.5] * points
    
    def to_json(self, plan: CompositionPlan) -> str:
        """Exportiert den Kompositionsplan als JSON"""
        d = asdict(plan)
        return json.dumps(d, indent=2, ensure_ascii=False)
    
    def save_plan(self, plan: CompositionPlan, filepath: str):
        """Speichert den Kompositionsplan in eine Datei"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json(plan))
        print(f"[GPTComposer] Plan gespeichert: {filepath}")


# Benötigt für Fallback
import random


# Demo
if __name__ == "__main__":
    composer = GPTComposer()
    
    test_prompts = [
        "Düstere Ambient-Landschaft, langsam aufsteigend wie Morgennebel über einem See",
        "Energetischer Minimal Techno, hypnotisch und treibend mit pulsierenden Arpeggios",
        "Melancholische Nacht-Stimmung, spärliche Noten, viel Raum, inspiriert von Nils Frahm"
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print('='*60)
        
        plan = composer.compose(prompt, duration_bars=16)
        
        if plan:
            print(f"\nErgebnis:")
            print(f"  Titel: {plan.title}")
            print(f"  BPM: {plan.bpm}")
            print(f"  Key: {plan.key} {plan.scale}")
            print(f"  Akkorde: {' → '.join(plan.chord_progression)}")
            print(f"  Mood: {plan.mood_arc}")
