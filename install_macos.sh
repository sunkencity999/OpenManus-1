#!/bin/bash
# LocalManus Installation Script for macOS
# This script installs the complete environment for LocalManus, including Ollama and the default model

set -e  # Exit immediately if a command exits with a non-zero status

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

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to prompt user for confirmation
confirm() {
    read -p "$1 [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY]) 
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Print welcome message
echo -e "${BOLD}===============================================${NC}"
echo -e "${BOLD}      LocalManus Installation for macOS       ${NC}"
echo -e "${BOLD}===============================================${NC}"
echo ""
print_message "This script will install LocalManus and its dependencies."
echo ""

# Check for Python 3.8+
print_message "Checking for Python 3.8+..."
if command_exists python3; then
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ $(echo "$python_version >= 3.8" | bc) -eq 1 ]]; then
        print_success "Python $python_version detected."
    else
        print_error "Python $python_version detected. LocalManus requires Python 3.8 or higher."
        if confirm "Would you like to install Python 3.8+ using Homebrew?"; then
            if ! command_exists brew; then
                print_message "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            print_message "Installing Python 3.8+..."
            brew install python
        else
            print_error "Please install Python 3.8+ manually and run this script again."
            exit 1
        fi
    fi
else
    print_error "Python 3 not found."
    if confirm "Would you like to install Python 3.8+ using Homebrew?"; then
        if ! command_exists brew; then
            print_message "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        print_message "Installing Python 3.8+..."
        brew install python
    else
        print_error "Please install Python 3.8+ manually and run this script again."
        exit 1
    fi
fi

# Check for pip
print_message "Checking for pip..."
if ! command_exists pip3; then
    print_error "pip3 not found."
    print_message "Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
    rm get-pip.py
fi
print_success "pip3 is installed."

# Check for virtualenv
print_message "Checking for virtualenv..."
if ! command_exists virtualenv; then
    print_message "Installing virtualenv..."
    pip3 install virtualenv
fi
print_success "virtualenv is installed."

# Check for Ollama
print_message "Checking for Ollama..."
if ! command_exists ollama; then
    print_message "Ollama not found. Installing Ollama..."
    if confirm "Would you like to install Ollama?"; then
        print_message "Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
    else
        print_warning "Ollama installation skipped. You'll need to install it manually."
    fi
else
    print_success "Ollama is already installed."
fi

# Create virtual environment
print_message "Creating virtual environment..."
if [ -d ".venv" ]; then
    if confirm "Virtual environment already exists. Would you like to recreate it?"; then
        rm -rf .venv
        virtualenv -p python3 .venv
    fi
else
    virtualenv -p python3 .venv
fi
print_success "Virtual environment created."

# Activate virtual environment
print_message "Activating virtual environment..."
source .venv/bin/activate
print_success "Virtual environment activated."

# Install dependencies
print_message "Installing dependencies..."
pip3 install -r requirements.txt
print_success "Dependencies installed."

# Pull default model with Ollama
if command_exists ollama; then
    print_message "Pulling default model with Ollama..."
    MODEL="llama3"
    
    # Check if the model is already downloaded
    if ollama list | grep -q "$MODEL"; then
        print_success "Model $MODEL is already downloaded."
    else
        print_message "Downloading model $MODEL. This may take a while..."
        ollama pull $MODEL
        if [ $? -eq 0 ]; then
            print_success "Model $MODEL downloaded successfully."
        else
            print_error "Failed to download model $MODEL."
            print_warning "You will need to download a model manually using 'ollama pull MODEL_NAME'."
        fi
    fi
else
    print_warning "Ollama is not installed. Skipping model download."
    print_warning "After installing Ollama, you'll need to download a model using 'ollama pull MODEL_NAME'."
fi

# Create configuration file
print_message "Creating configuration file..."
if [ ! -f "config.yaml" ]; then
    cat > config.yaml << EOL
# LocalManus Configuration
model: llama3  # Default model to use with Ollama
workspace_root: $(pwd)/workspace
api_key: ""  # Your API key if using OpenAI
EOL
    print_success "Configuration file created."
else
    print_warning "Configuration file already exists. Skipping."
fi

# Create workspace directory
print_message "Creating workspace directory..."
mkdir -p workspace
print_success "Workspace directory created."

# Final message
echo ""
echo -e "${BOLD}===============================================${NC}"
print_success "LocalManus installation completed!"
echo -e "${BOLD}===============================================${NC}"
echo ""
print_message "To start LocalManus, run:"
echo -e "  ${BOLD}source .venv/bin/activate${NC}"
echo -e "  ${BOLD}python main.py${NC}"
echo ""
print_message "If you encounter any issues, please check the documentation or report them on GitHub."
echo ""

# Deactivate virtual environment
deactivate

# Make the script executable
chmod +x launch_localmanus.sh
