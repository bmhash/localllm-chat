#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_NAME="${VENV_NAME:-llama_env}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PATH="$(dirname "$SCRIPT_DIR")/$VENV_NAME"
PYTHON_MIN_VERSION="3.12"
NODE_MIN_VERSION="14"
CUDA_REQUIRED=true

# Error handling
set -e
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo -e "${RED}Error: command \"${last_command}\" failed${NC}"' ERR

echo -e "${BLUE}Starting installation of LLM Chat Interface...${NC}"

# Function to check command status
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1 successful${NC}"
    else
        echo -e "${RED}✗ $1 failed${NC}"
        exit 1
    fi
}

# Function to check if command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed${NC}"
        echo "Please install $1 and try again"
        exit 1
    fi
}

# Function to compare versions
version_greater_equal() {
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# Check CUDA availability
check_cuda() {
    if [ "$CUDA_REQUIRED" = true ]; then
        if ! command -v nvidia-smi &> /dev/null; then
            echo -e "${RED}Error: NVIDIA GPU driver not found${NC}"
            echo "Please install NVIDIA drivers and CUDA toolkit"
            exit 1
        fi
        
        if ! nvidia-smi &> /dev/null; then
            echo -e "${RED}Error: Unable to communicate with NVIDIA GPU${NC}"
            echo "Please check your GPU installation"
            exit 1
        fi
        
        echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    fi
}

# System checks
echo -e "\n${BLUE}Checking system requirements...${NC}"
check_command python3
check_command pip3
check_command node
check_command npm
check_command git
check_cuda

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if ! version_greater_equal "$PYTHON_VERSION" "$PYTHON_MIN_VERSION"; then
    echo -e "${RED}Error: Python $PYTHON_MIN_VERSION or higher is required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version $PYTHON_VERSION${NC}"

# Check Node.js version
NODE_VERSION=$(node -v | cut -d 'v' -f 2)
if ! version_greater_equal "$NODE_VERSION" "$NODE_MIN_VERSION"; then
    echo -e "${RED}Error: Node.js $NODE_MIN_VERSION or higher is required (found $NODE_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js version $NODE_VERSION${NC}"

# Create virtual environment if it doesn't exist
echo -e "\n${BLUE}Setting up Python virtual environment...${NC}"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating new virtual environment at $VENV_PATH"
    python3 -m venv "$VENV_PATH"
    check_status "Virtual environment creation"
else
    echo -e "${YELLOW}Virtual environment already exists at $VENV_PATH${NC}"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"
check_status "Virtual environment activation"

# Upgrade pip
echo -e "\n${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip
check_status "Pip upgrade"

# Install Python dependencies
echo -e "\n${BLUE}Installing Python dependencies...${NC}"
pip install -r requirements.txt
check_status "Python dependencies installation"

# Set up frontend
echo -e "\n${BLUE}Setting up frontend...${NC}"
cd chat-interface || {
    echo -e "${RED}Error: chat-interface directory not found${NC}"
    exit 1
}

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
if [ -d "node_modules" ]; then
    echo -e "${YELLOW}Node modules already exist, running npm install to update...${NC}"
fi
npm install
check_status "Node.js dependencies installation"

# Create .env file and handle HuggingFace token
cd ..
if [ ! -f ".env" ]; then
    echo -e "\n${BLUE}Setting up environment configuration...${NC}"
    cp .env.example .env
    
    echo -e "\n${BLUE}Do you have a HuggingFace account and token? (y/n)${NC}"
    read -r has_token
    
    if [[ $has_token =~ ^[Nn]$ ]]; then
        echo -e "\n${YELLOW}To use this application, you'll need a HuggingFace account and token:${NC}"
        echo -e "1. Go to ${GREEN}https://huggingface.co/join${NC} to create an account"
        echo -e "2. After signing up, visit ${GREEN}https://huggingface.co/settings/tokens${NC} to create a token"
        echo -e "3. Once you have your token, edit the ${YELLOW}.env${NC} file and set HUGGING_FACE_TOKEN=your_token"
    else
        echo -e "\n${BLUE}Please enter your HuggingFace token:${NC}"
        read -r token
        if [ -n "$token" ]; then
            sed -i "s/HUGGING_FACE_TOKEN=/HUGGING_FACE_TOKEN=$token/" .env
            echo -e "${GREEN}✓ Token added to .env file${NC}"
        else
            echo -e "${YELLOW}No token provided. Please edit the .env file later and add your token${NC}"
        fi
    fi
fi

echo -e "\n${GREEN}Installation completed successfully!${NC}"
echo -e "${BLUE}Next steps:${NC}"
if [[ $has_token =~ ^[Nn]$ ]] || [ -z "$token" ]; then
    echo -e "1. Get your HuggingFace token and add it to the ${YELLOW}.env${NC} file"
fi
echo -e "2. Run ${YELLOW}./start.sh${NC} to start the application"
echo -e "3. Open ${YELLOW}http://localhost:3000${NC} in your browser\n"
