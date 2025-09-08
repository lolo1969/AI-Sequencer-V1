# AI Sequencer V1

Generative Minimal Sequencer (VCV Rack via IAC) with 3 Channels + MIDI Clock

## Features
- Starts with very few notes (small motif)
- Slowly evolves: only small changes per cycle (Add/Rotate/Register-Shift)
- Long gates/ties (controllable), subtle phase shifts, minimal humanization
- No API access, purely local
- **NEW**: Third channel (Lead/Upper Voice) + per-lane gate variations

## Installation

### Requirements
- Python 3.8 or higher
- Virtual environment (recommended)
- Required Python libraries:
  - `mido`
  - `python-rtmidi`
  - `numpy`
  - `openai`

### Setup
1. Clone the repository or download the project files.
2. Navigate to the project directory:
   ```bash
   cd /path/to/AI-Sequencer-V1
   ```
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Upgrade `pip` and install the required libraries:
   ```bash
   pip install --upgrade pip
   pip install mido python-rtmidi numpy openai
   ```

## Usage

### Setup (macOS + VCV Rack)
1. Enable the IAC Driver:
   - Open **Audio MIDI Setup** → **MIDI Studio** → **IAC Driver** → Check "Enable Device."
2. Configure VCV Rack:
   - Add one instance of "Core → MIDI-CV" for each channel:
     - **Bass**: Channel 1
     - **Melody**: Channel 2
     - **Lead**: Channel 3
   - Set the device to "IAC Driver Bus 1" (or equivalent).
3. (Optional) Add a "MIDI Clock" module to synchronize the clock.

### Running the Sequencer
1. Ensure the `OPENAI_API_KEY` environment variable is set if using OpenAI features:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```
2. Start the sequencer:
   ```bash
   python3 ai-sequencer.py
   ```
3. Follow the prompts to select the MIDI device (default: IAC).

### Writing a Prompt
- The `prompt.txt` file should contain a description of the kind of music you want to generate.
- Example prompt:
  ```
  Create a slow, evolving sequence in the style of Philip Glass. Use repeating arpeggios and minimal harmonic changes to build a meditative atmosphere. The rhythm should be steady but not mechanical, with subtle variations over time. Keep the dynamics soft and flowing, letting layers overlap gradually. Focus on consonant harmonies and smooth transitions, evoking a sense of timelessness and quiet intensity.
  ```

### Stopping the Sequencer
- Press `Ctrl+C` to stop the sequencer gracefully.

## License
This project is licensed under the MIT License.