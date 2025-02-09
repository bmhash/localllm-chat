#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VENV_NAME="${VENV_NAME:-.venv}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PATH="$SCRIPT_DIR/$VENV_NAME"
BACKEND_PORT=8000
FRONTEND_PORT=3000
MAX_STARTUP_WAIT=300  # Maximum seconds to wait for services

# Function to check server health
check_server_ready() {
    local response
    response=$(curl -s http://localhost:${BACKEND_PORT}/health)
    if [ $? -eq 0 ] && [[ "$response" == *"status"* ]]; then
        return 0
    fi
    return 1
}

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${RED}Error: Port $1 is already in use${NC}"
        exit 1
    fi
}

# Function to check if model files exist
check_model_files() {
    local model_id="$1"
    local cache_dir="${HF_HOME:-$HOME/.cache/huggingface}/hub"
    
    if [ ! -d "$cache_dir/models--${model_id/\//__}" ]; then
        echo -e "${YELLOW}Warning: Model $model_id not found in cache${NC}"
        echo -e "${BLUE}Downloading model files...${NC}"
        
        # Activate virtual environment
        source "$VENV_PATH/bin/activate"
        
        # Download model files
        python3 - <<EOF
from huggingface_hub import hf_hub_download, HfApi
import os

def download_model(repo_id):
    try:
        # Get list of files
        api = HfApi()
        files = api.list_repo_files(repo_id)
        
        # Download each file
        for file in files:
            if file.endswith('.safetensors') or file.endswith('.json') or file.endswith('.model'):
                print(f"Downloading {file}...")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file,
                    token=os.getenv('HUGGING_FACE_HUB_TOKEN')
                )
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

success = download_model('$model_id')
exit(0 if success else 1)
EOF
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to download model files${NC}"
            exit 1
        fi
    fi
}

# Cleanup function
cleanup() {
    echo -e "\n${BLUE}Cleaning up...${NC}"
    
    # Kill all backend processes
    pkill -f "uvicorn server:app" || true
    
    # Kill all frontend processes
    pkill -f "next" || true
    
    # Remove PID files
    rm -f .pid.backend .pid.frontend
    
    echo -e "${GREEN}Cleanup complete${NC}"
}

# Register cleanup for script termination
trap 'cleanup; exit 0' SIGINT SIGTERM

# Initial cleanup
cleanup

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
    echo "Please run install.sh first"
    exit 1
fi

# Check ports
check_port $BACKEND_PORT
check_port $FRONTEND_PORT

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Check for model files
echo -e "\n${BLUE}Checking model files...${NC}"
check_model_files "meta-llama/Llama-3.2-3B-Instruct"

# Start backend server
echo -e "\n${BLUE}Starting backend server...${NC}"
cd "$SCRIPT_DIR"
python -m uvicorn server:app --host 0.0.0.0 --port $BACKEND_PORT &
echo $! > .pid.backend

# Wait for backend to be ready
echo -e "${YELLOW}Waiting for backend server to start...${NC}"
COUNTER=0
while ! check_server_ready; do
    if [ $COUNTER -ge $MAX_STARTUP_WAIT ]; then
        echo -e "${RED}Error: Backend server failed to start within ${MAX_STARTUP_WAIT} seconds${NC}"
        cleanup
        exit 1
    fi
    sleep 1
    ((COUNTER++))
done
echo -e "${GREEN}Backend server is ready${NC}"

# Start frontend server
echo -e "\n${BLUE}Starting frontend server...${NC}"
cd "$SCRIPT_DIR/chat-interface"
npm run dev &
echo $! > .pid.frontend

echo -e "\n${GREEN}All services started successfully!${NC}"
echo -e "Frontend server is running at ${BLUE}http://localhost:${FRONTEND_PORT}${NC}"
echo -e "Backend server is running at ${BLUE}http://localhost:${BACKEND_PORT}${NC}"
echo -e "\nPress Ctrl+C to stop all services\n"

# Wait for any process to exit
wait -n

# Exit with an error if any process has failed
exit 1
