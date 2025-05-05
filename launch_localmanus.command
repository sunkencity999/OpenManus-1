#!/bin/bash
# Launch script for LocalManus on macOS
# This script can be double-clicked in Finder to launch LocalManus

# Change to the directory where the script is located
cd "$(dirname "$0")"

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
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    print_error "Ollama is not installed. Please run the installation script first."
    print_message "Run: ./install_macos.sh"
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    print_warning "Ollama is not running."
    print_message "Starting Ollama..."
    open -a Ollama
    
    # Wait for Ollama to start
    print_message "Waiting for Ollama to initialize..."
    for i in {1..10}; do
        if curl -s http://localhost:11434/api/tags > /dev/null; then
            print_success "Ollama is now running."
            break
        fi
        sleep 1
        echo -n "."
    done
    echo ""
    
    # Final check if Ollama is running
    if ! curl -s http://localhost:11434/api/tags > /dev/null; then
        print_error "Failed to start Ollama. Please start it manually."
        read -p "Press Enter to continue once Ollama is running, or Ctrl+C to exit..."
    fi
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

# Check if config.toml exists
if [ ! -f "config/config.toml" ]; then
    print_warning "Configuration file not found."
    print_message "Creating default configuration file..."
    cp config/config.example-model-ollama.toml config/config.toml
    print_success "Created default configuration for Ollama."
    print_message "Edit config/config.toml to customize your settings."
fi

# Start LocalManus
print_message "Starting LocalManus..."
python3 main.py

# Deactivate virtual environment when done
print_message "Shutting down LocalManus..."
deactivate
print_success "LocalManus closed."

# Keep terminal window open if there was an error
if [ $? -ne 0 ]; then
    print_error "LocalManus exited with an error."
    read -p "Press Enter to exit..."
fi
