# LocalManus Installation Script for Windows
# This script installs the complete environment for LocalManus, including Ollama and the default model

# Function to print colored messages
function Write-Message {
    param (
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host "[LocalManus] $Message" -ForegroundColor $Color
}

function Write-Success {
    param (
        [string]$Message
    )
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param (
        [string]$Message
    )
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param (
        [string]$Message
    )
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to check if a command exists
function Test-Command {
    param (
        [string]$Command
    )
    try {
        $null = Get-Command $Command -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# Function to prompt user for confirmation
function Confirm-Action {
    param (
        [string]$Message
    )
    $response = Read-Host "$Message [y/N]"
    return $response -match '^[yY]'
}

# Print welcome message
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "      LocalManus Installation for Windows      " -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Message "This script will install LocalManus and its dependencies." -Color Cyan
Write-Host ""

# Check for administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Error "This script requires administrator privileges. Please run PowerShell as Administrator and try again."
    exit 1
}

# Check for Python 3.8+
Write-Message "Checking for Python 3.8+..." -Color Cyan
if (Test-Command "python") {
    try {
        $pythonVersion = (python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") | Out-String
        $pythonVersion = $pythonVersion.Trim()
        $versionParts = $pythonVersion.Split('.')
        $major = [int]$versionParts[0]
        $minor = [int]$versionParts[1]
        
        if ($major -ge 3 -and $minor -ge 8) {
            Write-Success "Python $pythonVersion detected."
        } else {
            Write-Error "Python $pythonVersion detected. LocalManus requires Python 3.8 or higher."
            if (Confirm-Action "Would you like to install Python 3.8+?") {
                Write-Message "Installing Python 3.8+..." -Color Cyan
                
                # Check if Chocolatey is installed
                if (-not (Test-Command "choco")) {
                    Write-Message "Installing Chocolatey..." -Color Cyan
                    Set-ExecutionPolicy Bypass -Scope Process -Force
                    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
                    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
                }
                
                # Install Python using Chocolatey
                choco install python -y
                
                # Refresh environment variables
                $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
                
                Write-Success "Python installed. Please restart this script."
                exit 0
            } else {
                Write-Error "Please install Python 3.8+ manually and run this script again."
                exit 1
            }
        }
    } catch {
        Write-Error "Failed to determine Python version. Please make sure Python is installed correctly."
        exit 1
    }
} else {
    Write-Error "Python not found."
    if (Confirm-Action "Would you like to install Python 3.8+?") {
        Write-Message "Installing Python 3.8+..." -Color Cyan
        
        # Check if Chocolatey is installed
        if (-not (Test-Command "choco")) {
            Write-Message "Installing Chocolatey..." -Color Cyan
            Set-ExecutionPolicy Bypass -Scope Process -Force
            [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
            Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
        }
        
        # Install Python using Chocolatey
        choco install python -y
        
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        
        Write-Success "Python installed. Please restart this script."
        exit 0
    } else {
        Write-Error "Please install Python 3.8+ manually and run this script again."
        exit 1
    }
}

# Check for pip
Write-Message "Checking for pip..." -Color Cyan
if (-not (Test-Command "pip")) {
    Write-Error "pip not found."
    Write-Message "Installing pip..." -Color Cyan
    Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile get-pip.py
    python get-pip.py
    Remove-Item get-pip.py
}
Write-Success "pip is installed."

# Check for Ollama
Write-Message "Checking for Ollama..." -Color Cyan
if (-not (Test-Command "ollama")) {
    Write-Message "Ollama not found." -Color Cyan
    if (Confirm-Action "Would you like to install Ollama?") {
        Write-Message "Installing Ollama..." -Color Cyan
        
        # Download Ollama installer
        $ollamaInstallerUrl = "https://ollama.com/download/ollama-windows-amd64.zip"
        $ollamaZipPath = "$env:TEMP\ollama-windows-amd64.zip"
        $ollamaExtractPath = "$env:TEMP\ollama"
        
        Write-Message "Downloading Ollama..." -Color Cyan
        Invoke-WebRequest -Uri $ollamaInstallerUrl -OutFile $ollamaZipPath
        
        # Extract Ollama
        Write-Message "Extracting Ollama..." -Color Cyan
        Expand-Archive -Path $ollamaZipPath -DestinationPath $ollamaExtractPath -Force
        
        # Create Ollama directory
        $ollamaDir = "$env:LOCALAPPDATA\Ollama"
        if (-not (Test-Path $ollamaDir)) {
            New-Item -ItemType Directory -Path $ollamaDir -Force | Out-Null
        }
        
        # Copy Ollama executable
        Copy-Item -Path "$ollamaExtractPath\ollama.exe" -Destination $ollamaDir -Force
        
        # Add Ollama to PATH
        $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
        if ($userPath -notlike "*$ollamaDir*") {
            [System.Environment]::SetEnvironmentVariable("Path", "$userPath;$ollamaDir", "User")
        }
        
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        
        # Clean up
        Remove-Item $ollamaZipPath -Force
        Remove-Item $ollamaExtractPath -Recurse -Force
        
        Write-Success "Ollama installed."
    } else {
        Write-Warning "Ollama installation skipped. You'll need to install it manually."
    }
} else {
    Write-Success "Ollama is already installed."
}

# Create virtual environment
Write-Message "Creating virtual environment..." -Color Cyan
if (Test-Path ".venv") {
    if (Confirm-Action "Virtual environment already exists. Would you like to recreate it?") {
        Remove-Item -Path ".venv" -Recurse -Force
        python -m venv .venv
    }
} else {
    python -m venv .venv
}
Write-Success "Virtual environment created."

# Activate virtual environment
Write-Message "Activating virtual environment..." -Color Cyan
& ".\.venv\Scripts\Activate.ps1"
Write-Success "Virtual environment activated."

# Install dependencies
Write-Message "Installing dependencies..." -Color Cyan
pip install -r requirements.txt
Write-Success "Dependencies installed."

# Pull default model with Ollama
if (Test-Command "ollama") {
    Write-Message "Pulling default model with Ollama..." -Color Cyan
    $MODEL = "llama3"
    
    # Check if Ollama service is running
    $ollamaService = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
    if ($null -eq $ollamaService) {
        Write-Message "Starting Ollama service..." -Color Cyan
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 5
    }
    
    # Check if the model is already downloaded
    $modelList = ollama list
    if ($modelList -match $MODEL) {
        Write-Success "Model $MODEL is already downloaded."
    } else {
        Write-Message "Downloading model $MODEL. This may take a while..." -Color Cyan
        ollama pull $MODEL
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Model $MODEL downloaded successfully."
        } else {
            Write-Error "Failed to download model $MODEL."
            Write-Warning "You will need to download a model manually using 'ollama pull MODEL_NAME'."
        }
    }
} else {
    Write-Warning "Ollama is not installed. Skipping model download."
    Write-Warning "After installing Ollama, you'll need to download a model using 'ollama pull MODEL_NAME'."
}

# Create configuration file
Write-Message "Creating configuration file..." -Color Cyan
if (-not (Test-Path "config.yaml")) {
    @"
# LocalManus Configuration
model: llama3  # Default model to use with Ollama
workspace_root: $((Get-Location).Path)\workspace
api_key: ""  # Your API key if using OpenAI
"@ | Out-File -FilePath "config.yaml" -Encoding utf8
    Write-Success "Configuration file created."
} else {
    Write-Warning "Configuration file already exists. Skipping."
}

# Create workspace directory
Write-Message "Creating workspace directory..." -Color Cyan
if (-not (Test-Path "workspace")) {
    New-Item -ItemType Directory -Path "workspace" -Force | Out-Null
}
Write-Success "Workspace directory created."

# Create launch script
Write-Message "Creating launch script..." -Color Cyan
@"
@echo off
:: Launch script for LocalManus

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Start LocalManus
python main.py

:: Deactivate virtual environment when done
deactivate
"@ | Out-File -FilePath "launch_localmanus.bat" -Encoding ascii
Write-Success "Launch script created."

# Final message
Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Success "LocalManus installation completed!"
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Message "To start LocalManus, run:" -Color Cyan
Write-Host "  launch_localmanus.bat" -ForegroundColor White
Write-Host ""
Write-Message "If you encounter any issues, please check the documentation or report them on GitHub." -Color Cyan
Write-Host ""

# Deactivate virtual environment
deactivate
