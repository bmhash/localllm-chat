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
    local HF_TOKEN_FILE=".env"
    local token
    
    if [ -f "$HF_TOKEN_FILE" ] && grep -q "HUGGING_FACE_HUB_TOKEN=" "$HF_TOKEN_FILE"; then
        echo -e "${GREEN}Hugging Face token already configured${NC}"
        return 0
    fi
    
    echo -e "${YELLOW}A Hugging Face account and API token are required to download models.${NC}"
    echo -e "1. Create an account at ${BLUE}https://huggingface.co/join${NC}"
    echo -e "2. Go to ${BLUE}https://huggingface.co/settings/tokens${NC} to create an API token"
    echo -e "\nPlease choose:"
    echo -e "1) Enter your Hugging Face API token"
    echo -e "2) Skip for now (you won't be able to download models)"
    
    while true; do
        read -p "Enter your choice (1/2): " choice
        case $choice in
            1)
                read -sp "Please paste your Hugging Face API token: " token
                echo
                
                # Validate token is not empty
                if [ -z "$token" ]; then
                    echo -e "${RED}Error: Token cannot be empty${NC}"
                    continue
                fi
                
                # Create .env file if it doesn't exist
                if [ ! -f "$HF_TOKEN_FILE" ]; then
                    cp .env.example "$HF_TOKEN_FILE"
                fi
                
                # Remove any existing token line
                sed -i '/HUGGING_FACE_HUB_TOKEN=/d' "$HF_TOKEN_FILE"
                
                # Add new token
                echo "HUGGING_FACE_HUB_TOKEN=$token" >> "$HF_TOKEN_FILE"
                echo -e "${GREEN}✓ Hugging Face token configured successfully${NC}"
                
                # Install huggingface_hub if not already installed
                echo -e "\n${BLUE}Installing Hugging Face Hub...${NC}"
                if ! python3 -m pip install --quiet huggingface_hub; then
                    echo -e "${RED}Error: Failed to install huggingface_hub${NC}"
                    return 1
                fi
                
                # Login to Hugging Face Hub
                echo -e "\n${BLUE}Logging in to Hugging Face Hub...${NC}"
                source "$HF_TOKEN_FILE"
                if ! python3 -c "from huggingface_hub import login; login(token='$token', write_permission=False)"; then
                    echo -e "${RED}Error: Failed to login to Hugging Face Hub${NC}"
                    return 1
                fi
                echo -e "${GREEN}✓ Successfully logged in to Hugging Face Hub${NC}"
                break
                ;;
            2)
                echo -e "${YELLOW}Skipping token configuration${NC}"
                break
                ;;
            *)
                echo -e "${RED}Invalid choice. Please enter 1 or 2${NC}"
                ;;
        esac
    done
}

# Check CUDA availability
check_cuda() {
    if [ "$CUDA_REQUIRED" = true ]; then
        if ! command -v nvcc &> /dev/null; then
            echo -e "${YELLOW}Warning: CUDA not found. GPU acceleration will not be available.${NC}"
            echo -e "If you want to use GPU acceleration, please install CUDA toolkit."
            read -p "Continue without GPU support? [y/N] " response
            if [[ ! "$response" =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}Installation aborted by user${NC}"
                exit 0
            fi
        else
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
            echo -e "${GREEN}✓ Found CUDA version $CUDA_VERSION${NC}"
        fi
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

# Setup Hugging Face token
setup_huggingface_token

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
    cp .env.example .env
    echo -e "\n${BLUE}Do you have a HuggingFace account and token? (y/n)${NC}"
    read -r has_token
    
    if [ "$has_token" != "y" ]; then
        echo -e "\n${YELLOW}To use this application, you'll need a HuggingFace account and token:${NC}"
        echo -e "1. Sign up at ${GREEN}https://huggingface.co/join${NC}"
        echo -e "2. After signing up, visit ${GREEN}https://huggingface.co/settings/tokens${NC} to create a token"
        echo -e "3. Once you have your token, edit the ${YELLOW}.env${NC} file and set HUGGING_FACE_HUB_TOKEN=your_token"
    else
        echo -e "\n${BLUE}Please enter your HuggingFace token:${NC}"
        read -r token
        if [ -n "$token" ]; then
            sed -i "s/HUGGING_FACE_HUB_TOKEN=.*/HUGGING_FACE_HUB_TOKEN=$token/" .env
        fi
    fi
fi

# Check if token exists before downloading models
if [ -f ".env" ] && grep -q "HUGGING_FACE_HUB_TOKEN=.*[^[:space:]]" .env; then
    echo -e "\n${BLUE}Downloading language models...${NC}"
    python3 install_models.py
    check_status "Model download"
else
    echo -e "${RED}Error: Cannot download models without a valid HuggingFace token in .env${NC}"
    exit 1
fi

echo -e "\n${GREEN}Installation completed successfully!${NC}"
echo -e "${BLUE}Next steps:${NC}"
echo -e "1. Run ${YELLOW}./start.sh${NC} to start the application"
echo -e "2. Open ${YELLOW}http://localhost:3000${NC} in your browser\n"
