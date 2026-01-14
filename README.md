# AI Sequencer V1

Generativer Minimal-Sequencer für beliebige MIDI-Geräte mit 4 Kanälen: Bass, Melody, Lead, Arp

## Features
- 4 unabhängige Kanäle: Bass, Melody, Lead, Arp (Arpeggio)
- Startet mit wenigen Noten (Motiv)
- Langsame Evolution: kleine Änderungen pro Zyklus (Add/Rotate/Register-Shift)
- Lange Gates/Ties (steuerbar), subtile Phasenverschiebungen, Humanisierung
- Polyphoner MIDI-Output (alle Kanäle gleichzeitig)
- MIDI-Range- und Config-Validierung
- Stabiles Tempo (empfohlen: Hardware-Clock oder internes Clock-Modul)
- Shutdown funktioniert zuverlässig
- **Arp-Kanal:** Schnelle Arpeggios, kurze Noten, staccato, chord-basiert
- OpenAI GPT-3.5-turbo für musikalische Parameter aus Textprompt

## Installation

### Requirements
- Python 3.8 oder höher
- Virtuelle Umgebung (empfohlen)
- Python Libraries:
  - `mido`
  - `python-rtmidi`
  - `numpy`
  - `openai`

### Setup
1. Repository klonen oder Dateien herunterladen
2. Projektverzeichnis öffnen:
   ```bash
   cd /path/to/AI-Sequencer-V1
   ```
3. Virtuelle Umgebung erstellen und aktivieren:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Libraries installieren:
   ```bash
   pip install --upgrade pip
   pip install mido python-rtmidi numpy openai
   ```

## MIDI-Geräte Setup
1. MIDI-Interface oder virtuelles MIDI-Gerät aktivieren (z.B. IAC Driver auf macOS, LoopMIDI auf Windows)
2. Zielgerät/MIDI-Software:
   - 4x MIDI-Empfänger (Synthesizer, DAW, Modular, etc.) auf die gewünschten Kanäle stellen:
     - **Bass**: Channel 1
     - **Melody**: Channel 2
     - **Lead**: Channel 3
     - **Arp**: Channel 4
   - Device: z.B. "IAC Driver Bus 1" oder anderes verfügbares MIDI-Out
   - V/OCT und GATE verbinden (bei Modular)
3. (Optional) Clock-Modul für Synchronisation

## Sequencer starten
1. (Optional) OPENAI_API_KEY setzen:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```
2. Sequencer starten:
   ```bash
   python3 ai-sequencer.py --device <MIDI-Device-Name>
   ```
3. Stoppen: `Ctrl+C`

## prompt.txt Beispiel
```
Slow, meditative ambient in E major. Minimal notes, maximum space. Long sustained tones with gentle evolution. Inspired by Stars of the Lid and Eluvium.

bpm: 50
scale: major
root: E3
bars: 16
steps_per_bar: 8

mel_base_motif_degrees: [0, 4, 7]
mel_min_len_steps: 8
mel_max_len_steps: 32
mel_tie_bias: 0.95
mel_ghost_prob: 0.0
mel_evolve_every_bars: 32

bass_base_motif_degrees: [0]
bass_min_len_steps: 16
bass_max_len_steps: 64
bass_tie_bias: 0.98
bass_register_offset_oct: -1
bass_evolve_every_bars: 64

lead_base_motif_degrees: [7, 11]
lead_min_len_steps: 12
lead_max_len_steps: 48
lead_tie_bias: 0.9
lead_register_offset_oct: 1
lead_evolve_every_bars: 48

arp_base_motif_degrees: [0, 4, 7, 11, 12]
arp_min_len_steps: 2
arp_max_len_steps: 4
arp_tie_bias: 0.1
arp_register_offset_oct: 1
arp_evolve_every_bars: 8

harmony_enable: true
progression_degrees: [0, 4, 5, 3]
chord_change_every_bars: 4
chord_tone_bias: 0.9

swing: 0.0
jitter_ms: 0
clock_enable: true
```

## Lizenz
MIT License