# AI Sequencer V1

Generative minimal sequencer for any MIDI device with 4 channels: Bass, Melody, Lead, Arp

## Features
- 4 independent channels: Bass, Melody, Lead, Arp (Arpeggio)
- Starts with a small motif (few notes)
- Slowly evolves: small changes per cycle (add/rotate/register-shift)
- Long gates/ties (configurable), subtle phase shifts, humanization
- Multi-channel MIDI output (all channels play simultaneously)
- MIDI range and config validation
- Stable tempo (recommended: hardware clock or internal clock module)
- Reliable shutdown
- **Arp channel:** Fast arpeggios, short notes, staccato, chord-based
- OpenAI GPT-3.5-turbo for musical parameters from text prompt

## Installation

### Requirements
- Python 3.8 or higher
- Virtual environment (recommended)
- Python libraries:
  - `mido`
  - `python-rtmidi`
  - `numpy`
  - `openai`

### Setup
1. Clone the repository or download the project files
2. Open the project directory:
   ```bash
   cd /path/to/AI-Sequencer-V1
   ```
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install the libraries:
   ```bash
   pip install --upgrade pip
   pip install mido python-rtmidi numpy openai
   ```

## MIDI Device Setup
1. Activate your MIDI interface or virtual MIDI device (e.g. IAC Driver on macOS, LoopMIDI on Windows)
2. Target device/software:
   - Set up 4 MIDI receivers (synths, DAW, modular, etc.) to the desired channels:
     - **Bass**: Channel 1
     - **Melody**: Channel 2
     - **Lead**: Channel 3
     - **Arp**: Channel 4
   - Device: e.g. "IAC Driver Bus 1" or any available MIDI output
   - Connect V/OCT and GATE (for modular)
3. (Optional) Add a clock module for synchronization

## Running the Sequencer
1. (Optional) Set your OPENAI_API_KEY:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```
2. Start the sequencer:
   ```bash
   python3 ai-sequencer.py --device <MIDI-Device-Name>
   ```
3. Stop: `Ctrl+C`

## Example prompt.txt
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

## License
MIT License