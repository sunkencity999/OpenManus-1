#!/bin/bash
# Launch script for LocalManus

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
BLUE="\033[0;34m"
NC="\033[0m" # No Color

# Function to print colored messages
print_message() {
    echo -e "${BLUE}${BOLD}[LocalManus]${NC} $1"
}

print_success() {
    echo -e "${GREEN}${BOLD}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}${BOLD}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}${BOLD}[ERROR]${NC} $1"
}

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    print_error "Virtual environment not found. Please run the installation script first."
    print_message "Run: ./install_macos.sh"
    exit 1
fi

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    print_warning "Ollama is not running."
    print_message "Starting Ollama..."
    ollama serve &
    sleep 2
fi

# Print welcome message
echo -e "${BOLD}===============================================${NC}"
echo -e "${BOLD}             LocalManus                       ${NC}"
echo -e "${BOLD}===============================================${NC}"
echo ""

# Activate virtual environment
print_message "Activating virtual environment..."
source .venv/bin/activate
print_success "Virtual environment activated."

# Start LocalManus
print_message "Starting LocalManus..."
python3 main.py

# Deactivate virtual environment when done
print_message "Shutting down LocalManus..."
deactivate
print_success "LocalManus closed."
