# 🎵 AI Sequencer V2

> **Prompt → Music**: Describe your music like a story, and the AI composes it for you.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎼 The Concept

AI Sequencer V2 transforms natural language descriptions into living MIDI compositions. Imagine describing to a composer what music you want to hear — and they compose it for you instantly.

```
┌─────────────────────────────────────────────────────────────────┐
│  "Dark ambient landscape, slowly ascending                     │
│   like morning mist over a lake, inspired by                   │
│   Stars of the Lid..."                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: GPT-4 → Composition Plan (Chords, Structure, Dynamics)│
│  STAGE 2: Magenta → MIDI Generation (Bass, Melody, Lead, Arp)  │
│  STAGE 3: Evolution → Live Playback with Mutations             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                       🎹 MIDI Output
              (DAW, Hardware Synths, Modular, ...)
```

---

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API Key
export OPENAI_API_KEY="your-api-key"
```

### First Composition

**Easiest way with the start script:**

```bash
# Launch interactive menu
./start.sh
```

**Or directly via command line:**

```bash
# Send live to DAW (use --list-devices to find your device name)
./start.sh --device "Your MIDI Device" --text "Melancholic ambient in E minor"

# Generate MIDI file only
./start.sh --generate-only --text "Hypnotic minimal techno" --bars 32

# Replay saved plan
./start.sh --plan output/my_composition_plan.json
```

---

## 🎛️ Start Script (start.sh)

The interactive menu offers the following options:

| Option | Description |
|--------|-------------|
| **1) Live Mode** | Send MIDI live to device/DAW |
| **2) Generate Only** | Create MIDI file without playback |
| **3) Custom Prompt** | Describe music with your own text |
| **4) Load Plan** | Replay saved composition |
| **5) Show Devices** | List available MIDI outputs |

**Direct usage with arguments:**

```bash
./start.sh --device "Your MIDI Device" --bars 16
./start.sh --generate-only --output my_track.mid
./start.sh --plan output/my_composition_plan.json
```

---

## 📖 How to Write Good Prompts

### The Art of Music Description

| Element | Examples |
|---------|----------|
| **Mood** | dark, bright, melancholic, euphoric, ominous |
| **Tempo** | slow, meditative, driving, pulsating, 120 BPM |
| **Texture** | sparse, dense, ethereal, massive, delicate |
| **Key** | E minor, A minor, D Dorian, F# Phrygian |
| **Development** | ascending, fading, wave-like, building to climax |
| **References** | "in the style of...", "inspired by..." |

### Example Prompts

```
# Ambient
Deep ambient landscape with ethereal pad swells.
Very few notes, maximum space between sounds.
Slowly building like rising fog. E Dorian.
Inspired by Stars of the Lid and Eluvium.

# Minimal Techno  
Hypnotic minimal techno. Driving bassline in A minor.
Dotted arpeggios that slowly evolve.
Mechanical yet organic. 124 BPM.

# Neoclassical
Melancholic piano ballad in the style of Nils Frahm.
Sparse, deliberate melody. Occasional deep bass notes.
Lots of silence and room to breathe.
```

More examples in `example_prompts.txt`.

---

## 🔧 Architecture

### Stage 1: GPT-4 as Music Director
GPT-4 interprets your prompt and creates a structured **composition plan**:
- Key and scale
- Chord progression with timing
- Emotional arc (intensity curve)
- Instructions for each channel (Bass, Melody, Lead, Arp)

### Stage 2: MIDI Generation
The composition plan is translated into concrete MIDI notes:
- **MusicVAE** (optional): Neural melody generation
- **Algorithmic Generator**: Music theory-based generation
- Follows chord structure and intensity curve

### Stage 3: Evolution Engine
Generated patterns evolve during live playback:
- Small mutations (pitch, velocity, timing)
- Organic adding/removing of notes
- Follows the intensity curve

---

## 📁 Project Structure

```
AI-Sequencer-V1/
├── start.sh                # 🚀 Starter script (interactive menu)
├── ai-sequencer-v2.py      # Main script
├── gpt_composer.py         # GPT-4 composition module
├── magenta_generator.py    # MIDI generation
├── requirements.txt        # Python dependencies
│
├── prompt.txt              # Your creative prompt
├── example_prompts.txt     # Examples for different styles
│
└── output/                 # Generated compositions
    ├── <title>.mid              # MIDI file
    ├── <title>_plan.json        # Composition plan (reusable)
    └── ...
```

---

## ⚙️ Command Line Options

```
Options:
  --device, -d      MIDI device (run --list-devices to see available)
  --text, -t        Direct prompt text
  --prompt, -p      Path to prompt file (default: prompt.txt)
  --plan            Load saved composition plan (JSON)
  --bars, -b        Number of bars (default: 32)
  --output, -o      MIDI output file
  --generate-only   Generate MIDI only, no live playback
  --no-save         Don't save MIDI automatically
  --list-devices    Show available MIDI devices
```

**Examples:**

```bash
# Live playback with 16 bars
python3 ai-sequencer-v2.py --device "Your MIDI Device" --bars 16

# Prompt from file, send to hardware
python3 ai-sequencer-v2.py --prompt prompt.txt --device "USB MIDI"

# Generate MIDI only with custom name
python3 ai-sequencer-v2.py --generate-only --text "Dark Ambient" --output dark.mid

# Play saved plan
python3 ai-sequencer-v2.py --plan output/my_composition_plan.json
```

---

## 🔌 MIDI Setup

### Virtual MIDI (for DAW routing)
- **macOS**: Use IAC Driver (Audio MIDI Setup → MIDI Studio)
- **Windows**: Use loopMIDI or similar virtual MIDI cable
- **Linux**: Use ALSA virtual MIDI ports

### Hardware (USB MIDI, etc.)
```bash
# Show available devices
./start.sh --list-devices

# Send to hardware
./start.sh --device "Your Device Name"
```

### DAW Integration
Set your DAW to the corresponding MIDI input and create 4 MIDI tracks:

| Channel | Instrument |
|---------|------------|
| 1 | Bass |
| 2 | Melody |
| 3 | Lead |
| 4 | Arp |

---

## 🎹 Saved Compositions

Each generation automatically saves:
- **MIDI file** (`.mid`) — Directly importable to DAW
- **Composition plan** (`_plan.json`) — Reusable for new variations

**Replay a plan:**
```bash
./start.sh --plan output/example_ambient_plan.json
```

Your generated compositions will appear in the `output/` folder with names derived from the prompt.

---

## 🧪 Optional Magenta Installation

For neural melody generation you can install Magenta:

```bash
pip install note-seq magenta
```

**Note**: The sequencer works without Magenta. It uses the algorithmic fallback which also produces music theory-based results.

---

## 🔮 Future Features

- [ ] Real-time prompt changes during playback
- [ ] Multi-track recording to DAW
- [ ] Web interface
- [ ] MusicGen audio generation
- [ ] Integration with Suno, Udio, etc.

---

## 📝 License

MIT License

---

💡 **Tip**: Best results come from detailed, vivid descriptions. Think like a director describing a film scene to a composer!
