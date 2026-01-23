#!/usr/bin/env python3
"""
midi_panic.py - Schaltet alle MIDI-Noten auf allen Kanälen aus

Verwendung:
    python3 midi_panic.py              # Verwendet ersten verfügbaren Port
    python3 midi_panic.py IAC          # Verwendet IAC Driver
    python3 midi_panic.py "OXI ONE"    # Verwendet OXI ONE
"""

import sys
import time
import mido
from mido import Message


def midi_panic(device_hint: str = ""):
    """Sendet MIDI Panic an alle Kanäle"""
    
    names = mido.get_output_names()
    
    if not names:
        print("❌ Keine MIDI-Ausgänge gefunden!")
        return
    
    print("Verfügbare MIDI-Ausgänge:")
    for i, name in enumerate(names):
        print(f"  [{i}] {name}")
    
    # Finde passenden Port
    target = None
    if device_hint:
        for name in names:
            if device_hint.lower() in name.lower():
                target = name
                break
    
    if not target:
        target = names[0]
    
    print(f"\n🔇 Sende MIDI Panic an: {target}")
    
    try:
        port = mido.open_output(target)
        
        for channel in range(16):
            # CC 123 = All Notes Off
            port.send(Message('control_change', channel=channel, control=123, value=0))
            # CC 120 = All Sound Off
            port.send(Message('control_change', channel=channel, control=120, value=0))
            # CC 121 = Reset All Controllers
            port.send(Message('control_change', channel=channel, control=121, value=0))
            
            # Explizite Note-Offs für alle Noten
            for note in range(128):
                port.send(Message('note_on', channel=channel, note=note, velocity=0))
            
            print(f"  ✓ Kanal {channel + 1} zurückgesetzt")
        
        time.sleep(0.1)
        port.close()
        
        print("\n✅ Alle Noten ausgeschaltet!")
        
    except Exception as e:
        print(f"❌ Fehler: {e}")


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else ""
    midi_panic(device)
