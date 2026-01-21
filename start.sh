#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  🎵 AI Sequencer V2 — Starter Script
# ═══════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
SEQUENCER="$SCRIPT_DIR/ai-sequencer-v2.py"

# Farben
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Header anzeigen
show_header() {
    echo -e "${MAGENTA}"
    echo "═══════════════════════════════════════════════════════════"
    echo "  🎵 AI Sequencer V2 — Prompt-to-Music"
    echo "═══════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

# Menü anzeigen
show_menu() {
    echo -e "${CYAN}Was möchtest du tun?${NC}\n"
    echo -e "  ${GREEN}1)${NC} 🎹 Live-Modus (an MIDI-Gerät senden)"
    echo -e "  ${GREEN}2)${NC} 💾 Nur MIDI-Datei generieren"
    echo -e "  ${GREEN}3)${NC} 📝 Mit eigenem Prompt starten"
    echo -e "  ${GREEN}4)${NC} 📂 Vorhandenen Plan laden"
    echo -e "  ${GREEN}5)${NC} 🔌 MIDI-Geräte anzeigen"
    echo -e "  ${GREEN}6)${NC} ❌ Beenden"
    echo ""
}

# Prompt eingeben
get_prompt() {
    echo -e "${YELLOW}Beschreibe deine Musik:${NC}"
    echo -e "${BLUE}(z.B. 'Düsterer Ambient, E-Moll, langsam aufsteigend')${NC}"
    read -r -p "> " PROMPT_TEXT
    echo "$PROMPT_TEXT"
}

# Takte abfragen
get_bars() {
    echo -e "${YELLOW}Anzahl Takte [Standard: 16]:${NC}"
    read -r -p "> " BARS
    if [ -z "$BARS" ]; then
        BARS=16
    fi
    echo "$BARS"
}

# MIDI-Gerät auswählen
select_device() {
    echo -e "${YELLOW}Verfügbare MIDI-Geräte:${NC}"
    "$VENV_PYTHON" "$SEQUENCER" --list-devices 2>/dev/null
    echo ""
    echo -e "${YELLOW}MIDI-Gerät eingeben [Standard: IAC]:${NC}"
    read -r -p "> " DEVICE
    if [ -z "$DEVICE" ]; then
        DEVICE="IAC"
    fi
    echo "$DEVICE"
}

# Dateiname abfragen
get_output_name() {
    echo -e "${YELLOW}Dateiname (ohne .mid) [Standard: auto]:${NC}"
    read -r -p "> " FILENAME
    echo "$FILENAME"
}

# Live-Modus
live_mode() {
    echo ""
    DEVICE=$(select_device)
    echo ""
    BARS=$(get_bars)
    echo ""
    echo -e "${GREEN}▶ Starte Live-Modus auf '$DEVICE' mit $BARS Takten...${NC}\n"
    "$VENV_PYTHON" "$SEQUENCER" --device "$DEVICE" --bars "$BARS"
}

# Nur generieren
generate_only() {
    echo ""
    BARS=$(get_bars)
    FILENAME=$(get_output_name)
    echo ""
    
    if [ -z "$FILENAME" ]; then
        echo -e "${GREEN}▶ Generiere MIDI mit $BARS Takten...${NC}\n"
        "$VENV_PYTHON" "$SEQUENCER" --generate-only --bars "$BARS"
    else
        echo -e "${GREEN}▶ Generiere '$FILENAME.mid' mit $BARS Takten...${NC}\n"
        "$VENV_PYTHON" "$SEQUENCER" --generate-only --bars "$BARS" --output "$FILENAME.mid"
    fi
}

# Mit eigenem Prompt
custom_prompt() {
    echo ""
    PROMPT_TEXT=$(get_prompt)
    echo ""
    BARS=$(get_bars)
    echo ""
    
    echo -e "${CYAN}Ausgabe-Modus:${NC}"
    echo -e "  ${GREEN}1)${NC} Live an MIDI-Gerät"
    echo -e "  ${GREEN}2)${NC} Nur MIDI-Datei speichern"
    read -r -p "> " OUTPUT_MODE
    echo ""
    
    if [ "$OUTPUT_MODE" = "1" ]; then
        DEVICE=$(select_device)
        echo ""
        echo -e "${GREEN}▶ Starte mit Prompt: '$PROMPT_TEXT'${NC}\n"
        "$VENV_PYTHON" "$SEQUENCER" --device "$DEVICE" --bars "$BARS" --text "$PROMPT_TEXT"
    else
        FILENAME=$(get_output_name)
        echo ""
        if [ -z "$FILENAME" ]; then
            echo -e "${GREEN}▶ Generiere mit Prompt: '$PROMPT_TEXT'${NC}\n"
            "$VENV_PYTHON" "$SEQUENCER" --generate-only --bars "$BARS" --text "$PROMPT_TEXT"
        else
            echo -e "${GREEN}▶ Generiere '$FILENAME.mid' mit Prompt: '$PROMPT_TEXT'${NC}\n"
            "$VENV_PYTHON" "$SEQUENCER" --generate-only --bars "$BARS" --text "$PROMPT_TEXT" --output "$FILENAME.mid"
        fi
    fi
}

# Plan laden
load_plan() {
    echo ""
    echo -e "${YELLOW}Verfügbare Pläne:${NC}"
    ls -1 "$SCRIPT_DIR/output/"*_plan.json 2>/dev/null | while read -r file; do
        basename "$file"
    done
    ls -1 "$SCRIPT_DIR/"*_plan.json 2>/dev/null | while read -r file; do
        basename "$file"
    done
    echo ""
    echo -e "${YELLOW}Plan-Datei eingeben:${NC}"
    read -r -p "> " PLAN_FILE
    
    # Prüfen ob im output/ Ordner
    if [ -f "$SCRIPT_DIR/output/$PLAN_FILE" ]; then
        PLAN_FILE="$SCRIPT_DIR/output/$PLAN_FILE"
    elif [ -f "$SCRIPT_DIR/$PLAN_FILE" ]; then
        PLAN_FILE="$SCRIPT_DIR/$PLAN_FILE"
    fi
    
    echo ""
    echo -e "${CYAN}Ausgabe-Modus:${NC}"
    echo -e "  ${GREEN}1)${NC} Live an MIDI-Gerät"
    echo -e "  ${GREEN}2)${NC} Nur MIDI-Datei speichern"
    read -r -p "> " OUTPUT_MODE
    echo ""
    
    if [ "$OUTPUT_MODE" = "1" ]; then
        DEVICE=$(select_device)
        echo ""
        echo -e "${GREEN}▶ Lade Plan: $PLAN_FILE${NC}\n"
        "$VENV_PYTHON" "$SEQUENCER" --device "$DEVICE" --plan "$PLAN_FILE"
    else
        echo -e "${GREEN}▶ Lade Plan: $PLAN_FILE${NC}\n"
        "$VENV_PYTHON" "$SEQUENCER" --generate-only --plan "$PLAN_FILE"
    fi
}

# MIDI-Geräte anzeigen
list_devices() {
    echo ""
    "$VENV_PYTHON" "$SEQUENCER" --list-devices
    echo ""
}

# ═══════════════════════════════════════════════════════════════════
#  Hauptprogramm
# ═══════════════════════════════════════════════════════════════════

# Prüfen ob venv existiert
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${YELLOW}⚠️  Virtual Environment nicht gefunden!${NC}"
    echo "Bitte erst einrichten mit: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Falls Argumente übergeben wurden, direkt ausführen
if [ $# -gt 0 ]; then
    "$VENV_PYTHON" "$SEQUENCER" "$@"
    exit $?
fi

# Interaktiver Modus
clear
show_header

while true; do
    show_menu
    read -r -p "Auswahl [1-6]: " choice
    
    case $choice in
        1)
            live_mode
            echo ""
            echo -e "${CYAN}Drücke Enter um fortzufahren...${NC}"
            read -r
            clear
            show_header
            ;;
        2)
            generate_only
            echo ""
            echo -e "${CYAN}Drücke Enter um fortzufahren...${NC}"
            read -r
            clear
            show_header
            ;;
        3)
            custom_prompt
            echo ""
            echo -e "${CYAN}Drücke Enter um fortzufahren...${NC}"
            read -r
            clear
            show_header
            ;;
        4)
            load_plan
            echo ""
            echo -e "${CYAN}Drücke Enter um fortzufahren...${NC}"
            read -r
            clear
            show_header
            ;;
        5)
            list_devices
            echo -e "${CYAN}Drücke Enter um fortzufahren...${NC}"
            read -r
            clear
            show_header
            ;;
        6)
            echo -e "\n${MAGENTA}🎵 Auf Wiedersehen!${NC}\n"
            exit 0
            ;;
        *)
            echo -e "${YELLOW}Ungültige Auswahl. Bitte 1-6 wählen.${NC}"
            ;;
    esac
done
