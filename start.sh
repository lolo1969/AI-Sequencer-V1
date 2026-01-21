#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  🎵 AI Sequencer V2 — Starter Script
# ═══════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
SEQUENCER="$SCRIPT_DIR/ai-sequencer-v2.py"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Show header
show_header() {
    echo -e "${MAGENTA}"
    echo "═══════════════════════════════════════════════════════════"
    echo "  🎵 AI Sequencer V2 — Prompt-to-Music"
    echo "═══════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

# Show menu
show_menu() {
    echo -e "${CYAN}What would you like to do?${NC}\n"
    echo -e "  ${GREEN}1)${NC} 🎹 Live Mode (stream MIDI to device/DAW)"
    echo -e "  ${GREEN}2)${NC} 💾 Generate Only (create MIDI file)"
    echo -e "  ${GREEN}3)${NC} 📝 Custom Prompt (describe your music)"
    echo -e "  ${GREEN}4)${NC} 📂 Load Plan (replay saved composition)"
    echo -e "  ${GREEN}5)${NC} 🔌 Show MIDI Devices"
    echo -e "  ${GREEN}6)${NC} ❌ Exit"
    echo ""
}

# Get prompt
get_prompt() {
    echo -e "${YELLOW}Describe your music:${NC}"
    echo -e "${BLUE}(e.g. 'Dark ambient, E minor, slowly ascending')${NC}"
    read -r -p "> " PROMPT_TEXT
    echo "$PROMPT_TEXT"
}

# Get bars
get_bars() {
    echo -e "${YELLOW}Number of bars [default: 16]:${NC}"
    read -r -p "> " BARS
    if [ -z "$BARS" ]; then
        BARS=16
    fi
    echo "$BARS"
}

# Select MIDI device
select_device() {
    echo -e "${YELLOW}Available MIDI devices:${NC}"
    "$VENV_PYTHON" "$SEQUENCER" --list-devices 2>/dev/null
    echo ""
    echo -e "${YELLOW}Enter MIDI device name [default: IAC]:${NC}"
    read -r -p "> " DEVICE
    if [ -z "$DEVICE" ]; then
        DEVICE="IAC"
    fi
    echo "$DEVICE"
}

# Get output filename
get_output_name() {
    echo -e "${YELLOW}Filename (without .mid) [default: auto]:${NC}"
    read -r -p "> " FILENAME
    echo "$FILENAME"
}

# Live mode
live_mode() {
    echo ""
    DEVICE=$(select_device)
    echo ""
    BARS=$(get_bars)
    echo ""
    echo -e "${GREEN}▶ Starting Live Mode on '$DEVICE' with $BARS bars...${NC}\n"
    "$VENV_PYTHON" "$SEQUENCER" --device "$DEVICE" --bars "$BARS"
}

# Generate only
generate_only() {
    echo ""
    BARS=$(get_bars)
    FILENAME=$(get_output_name)
    echo ""
    
    if [ -z "$FILENAME" ]; then
        echo -e "${GREEN}▶ Generating MIDI with $BARS bars...${NC}\n"
        "$VENV_PYTHON" "$SEQUENCER" --generate-only --bars "$BARS"
    else
        echo -e "${GREEN}▶ Generating '$FILENAME.mid' with $BARS bars...${NC}\n"
        "$VENV_PYTHON" "$SEQUENCER" --generate-only --bars "$BARS" --output "$FILENAME.mid"
    fi
}

# Custom prompt
custom_prompt() {
    echo ""
    PROMPT_TEXT=$(get_prompt)
    echo ""
    BARS=$(get_bars)
    echo ""
    
    echo -e "${CYAN}Output mode:${NC}"
    echo -e "  ${GREEN}1)${NC} Live to MIDI device"
    echo -e "  ${GREEN}2)${NC} Save MIDI file only"
    read -r -p "> " OUTPUT_MODE
    echo ""
    
    if [ "$OUTPUT_MODE" = "1" ]; then
        DEVICE=$(select_device)
        echo ""
        echo -e "${GREEN}▶ Starting with prompt: '$PROMPT_TEXT'${NC}\n"
        "$VENV_PYTHON" "$SEQUENCER" --device "$DEVICE" --bars "$BARS" --text "$PROMPT_TEXT"
    else
        FILENAME=$(get_output_name)
        echo ""
        if [ -z "$FILENAME" ]; then
            echo -e "${GREEN}▶ Generating with prompt: '$PROMPT_TEXT'${NC}\n"
            "$VENV_PYTHON" "$SEQUENCER" --generate-only --bars "$BARS" --text "$PROMPT_TEXT"
        else
            echo -e "${GREEN}▶ Generating '$FILENAME.mid' with prompt: '$PROMPT_TEXT'${NC}\n"
            "$VENV_PYTHON" "$SEQUENCER" --generate-only --bars "$BARS" --text "$PROMPT_TEXT" --output "$FILENAME.mid"
        fi
    fi
}

# Load plan
load_plan() {
    echo ""
    echo -e "${YELLOW}Available plans:${NC}"
    ls -1 "$SCRIPT_DIR/output/"*_plan.json 2>/dev/null | while read -r file; do
        basename "$file"
    done
    ls -1 "$SCRIPT_DIR/"*_plan.json 2>/dev/null | while read -r file; do
        basename "$file"
    done
    echo ""
    echo -e "${YELLOW}Enter plan filename:${NC}"
    read -r -p "> " PLAN_FILE
    
    # Check if in output/ folder
    if [ -f "$SCRIPT_DIR/output/$PLAN_FILE" ]; then
        PLAN_FILE="$SCRIPT_DIR/output/$PLAN_FILE"
    elif [ -f "$SCRIPT_DIR/$PLAN_FILE" ]; then
        PLAN_FILE="$SCRIPT_DIR/$PLAN_FILE"
    fi
    
    echo ""
    echo -e "${CYAN}Output mode:${NC}"
    echo -e "  ${GREEN}1)${NC} Live to MIDI device"
    echo -e "  ${GREEN}2)${NC} Save MIDI file only"
    read -r -p "> " OUTPUT_MODE
    echo ""
    
    if [ "$OUTPUT_MODE" = "1" ]; then
        DEVICE=$(select_device)
        echo ""
        echo -e "${GREEN}▶ Loading plan: $PLAN_FILE${NC}\n"
        "$VENV_PYTHON" "$SEQUENCER" --device "$DEVICE" --plan "$PLAN_FILE"
    else
        echo -e "${GREEN}▶ Loading plan: $PLAN_FILE${NC}\n"
        "$VENV_PYTHON" "$SEQUENCER" --generate-only --plan "$PLAN_FILE"
    fi
}

# List MIDI devices
list_devices() {
    echo ""
    "$VENV_PYTHON" "$SEQUENCER" --list-devices
    echo ""
}

# ═══════════════════════════════════════════════════════════════════
#  Main Program
# ═══════════════════════════════════════════════════════════════════

# Check if venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found!${NC}"
    echo "Please set up first: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# If arguments passed, run directly
if [ $# -gt 0 ]; then
    "$VENV_PYTHON" "$SEQUENCER" "$@"
    exit $?
fi

# Interactive mode
clear
show_header

while true; do
    show_menu
    read -r -p "Choice [1-6]: " choice
    
    case $choice in
        1)
            live_mode
            echo ""
            echo -e "${CYAN}Press Enter to continue...${NC}"
            read -r
            clear
            show_header
            ;;
        2)
            generate_only
            echo ""
            echo -e "${CYAN}Press Enter to continue...${NC}"
            read -r
            clear
            show_header
            ;;
        3)
            custom_prompt
            echo ""
            echo -e "${CYAN}Press Enter to continue...${NC}"
            read -r
            clear
            show_header
            ;;
        4)
            load_plan
            echo ""
            echo -e "${CYAN}Press Enter to continue...${NC}"
            read -r
            clear
            show_header
            ;;
        5)
            list_devices
            echo -e "${CYAN}Press Enter to continue...${NC}"
            read -r
            clear
            show_header
            ;;
        6)
            echo -e "\n${MAGENTA}🎵 Goodbye!${NC}\n"
            exit 0
            ;;
        *)
            echo -e "${YELLOW}Invalid choice. Please select 1-6.${NC}"
            ;;
    esac
done
