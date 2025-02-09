#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_NAME=".venv"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PATH="$SCRIPT_DIR/$VENV_NAME"
PYTHON_MIN_VERSION="3.12"
NODE_MIN_VERSION="14"
CUDA_REQUIRED=true
HF_TOKEN_FILE=".env"

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

# Function to handle Hugging Face token
setup_huggingface_token() {
    if [ ! -f "$HF_TOKEN_FILE" ]; then
        echo -e "\n${YELLOW}Hugging Face token not found${NC}"
        read -p "Enter your Hugging Face token (or press enter to skip): " token
        if [ ! -z "$token" ]; then
            echo "HUGGING_FACE_HUB_TOKEN=$token" > "$HF_TOKEN_FILE"
            echo -e "${GREEN}Token saved to $HF_TOKEN_FILE${NC}"
        else
            echo -e "${YELLOW}No token provided. You may need to set HUGGING_FACE_HUB_TOKEN manually later.${NC}"
        fi
    else
        echo -e "${GREEN}Found existing Hugging Face token${NC}"
    fi
}

# Function to check CUDA
check_cuda() {
    if [ "$CUDA_REQUIRED" = true ]; then
        if ! command -v nvidia-smi &> /dev/null; then
            echo -e "${RED}Error: CUDA not found${NC}"
            echo "Please install CUDA and ensure nvidia-smi is available"
            exit 1
        fi
        echo -e "${GREEN}✓ CUDA is available${NC}"
    fi
}

# System checks
echo -e "\n${BLUE}Checking system requirements...${NC}"
check_command python3
check_command pip3
check_command npm

# Check if safetensors is installed
if ! python3 -c "import safetensors" &> /dev/null; then
    echo -e "${YELLOW}Warning: safetensors not found, will be installed${NC}"
fi

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

# Check CUDA
check_cuda

# Create and activate virtual environment
echo -e "\n${BLUE}Setting up Python virtual environment...${NC}"
if [ -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}Found existing virtual environment, recreating...${NC}"
    rm -rf "$VENV_PATH"
fi
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
check_status "Virtual environment creation"

# Upgrade pip
echo -e "\n${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip
check_status "pip upgrade"

# Install Python dependencies
echo -e "\n${BLUE}Installing Python dependencies...${NC}"
pip install -r requirements.txt
check_status "Python dependencies installation"

# Setup Hugging Face token
setup_huggingface_token

# Check if token exists before downloading models
if [ -f ".env" ] && grep -q "HUGGING_FACE_HUB_TOKEN=.*[^[:space:]]" .env; then
    echo -e "\n${BLUE}Downloading language models...${NC}"
    source .env
    python3 install_models.py
    check_status "Model download"
else
    echo -e "${RED}Error: Cannot download models without a valid HuggingFace token in .env${NC}"
    exit 1
fi

# Install frontend dependencies
echo -e "\n${BLUE}Installing frontend dependencies...${NC}"
cd "$SCRIPT_DIR/chat-interface"
npm install
check_status "Frontend dependencies installation"

echo -e "\n${GREEN}Installation completed successfully!${NC}"
echo -e "You can now run ${BLUE}./start.sh${NC} to start the application"
