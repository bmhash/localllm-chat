#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VENV_NAME="${VENV_NAME:-locallmchat_venv}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PATH="$(dirname "$SCRIPT_DIR")/$VENV_NAME"
BACKEND_PORT=8000
FRONTEND_PORT=3000
MAX_STARTUP_WAIT=300  # Maximum seconds to wait for services

# Function to check server health
check_server_ready() {
    local response
    response=$(curl -s http://localhost:${BACKEND_PORT}/health)
    if [[ $? -eq 0 && $response == *"loaded_models"* ]]; then
        local total_models
        total_models=$(echo $response | grep -o '"total_models":[0-9]*' | cut -d':' -f2)
        if [[ $total_models -gt 0 ]]; then
            echo -e "${GREEN}Server ready with $total_models models loaded${NC}"
            return 0
        fi
    fi
    return 1
}

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        return 0
    fi
    return 1
}

# Cleanup function
cleanup() {
    echo -e "\n${BLUE}Cleaning up processes...${NC}"
    
    # Kill Next.js process
    if pgrep -f "next dev" > /dev/null; then
        echo -e "${YELLOW}Stopping frontend server...${NC}"
        pkill -f "next dev"
    fi
    
    # Kill FastAPI server
    if pgrep -f "uvicorn server:app" > /dev/null; then
        echo -e "${YELLOW}Stopping backend server...${NC}"
        pkill -f "uvicorn server:app"
    fi
    
    # Remove temporary files
    echo -e "${YELLOW}Cleaning up temporary files...${NC}"
    rm -rf offload
    
    echo -e "${GREEN}Cleanup complete${NC}"
}

# Register cleanup for script termination
trap 'cleanup; exit 0' SIGINT SIGTERM

# Initial cleanup
echo -e "${BLUE}Performing initial cleanup...${NC}"
cleanup

# Check if ports are available
if check_port $BACKEND_PORT; then
    echo -e "${RED}Error: Port $BACKEND_PORT is already in use${NC}"
    exit 1
fi

if check_port $FRONTEND_PORT; then
    echo -e "${RED}Error: Port $FRONTEND_PORT is already in use${NC}"
    exit 1
fi

echo -e "${BLUE}Starting LLM Chat Application...${NC}"

# Activate virtual environment
echo -e "${GREEN}Activating Python virtual environment...${NC}"
source "$VENV_PATH/bin/activate" || {
    echo -e "${RED}Failed to activate virtual environment at $VENV_PATH${NC}"
    echo "Please run ./install.sh first"
    exit 1
}

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Please copy .env.example to .env and add your Hugging Face token"
    exit 1
fi

# Check for Hugging Face token
if ! grep -q "HUGGING_FACE_HUB_TOKEN" .env || grep -q "HUGGING_FACE_HUB_TOKEN=your_token_here" .env; then
    echo -e "${RED}Error: Hugging Face token not set in .env${NC}"
    echo -e "Please set your token in .env file"
    exit 1
fi

# Login to Hugging Face Hub
echo -e "${BLUE}Logging in to Hugging Face Hub...${NC}"
source .env
if ! python3 -c "from huggingface_hub import login; login(token='$HUGGING_FACE_HUB_TOKEN', write_permission=False)"; then
    echo -e "${RED}Error: Failed to login to Hugging Face Hub${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Successfully logged in to Hugging Face Hub${NC}"

# Verify models are downloaded
echo -e "${BLUE}Verifying model downloads...${NC}"
python3 -c "
from install_models import check_model_files, MODELS_CONFIG
missing = [model for model in MODELS_CONFIG.keys() if not check_model_files(model)]
if missing:
    print('Missing models:', ', '.join(missing))
    print('Please run ./install.sh to download missing models')
    exit(1)
print('All models verified')
"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Some models are missing${NC}"
    exit 1
fi
echo -e "${GREEN}✓ All models verified${NC}"

# Start backend server
echo -e "${GREEN}Starting backend server...${NC}"
uvicorn server:app --host 0.0.0.0 --port $BACKEND_PORT &
BACKEND_PID=$!

# Wait for backend to be ready
echo -e "${YELLOW}Waiting for backend server to be ready...${NC}"
COUNTER=0
while ! check_server_ready; do
    if ! ps -p $BACKEND_PID > /dev/null; then
        echo -e "${RED}Backend server failed to start${NC}"
        cleanup
        exit 1
    fi
    
    if [ $COUNTER -gt $MAX_STARTUP_WAIT ]; then
        echo -e "${RED}Timeout waiting for backend server${NC}"
        cleanup
        exit 1
    fi
    
    echo -n "."
    sleep 1
    ((COUNTER++))
done
echo -e "\n${GREEN}Backend server is ready!${NC}"

# Start frontend
echo -e "${GREEN}Starting frontend server...${NC}"
cd chat-interface || {
    echo -e "${RED}Error: chat-interface directory not found${NC}"
    cleanup
    exit 1
}

# Install/update frontend dependencies if needed
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install || {
        echo -e "${RED}Failed to install frontend dependencies${NC}"
        cleanup
        exit 1
    }
fi

echo -e "${GREEN}Starting frontend development server...${NC}"
npm run dev &
FRONTEND_PID=$!

# Wait for frontend to be ready
echo -e "${YELLOW}Waiting for frontend server to be ready...${NC}"
COUNTER=0
while ! curl -s http://localhost:$FRONTEND_PORT > /dev/null; do
    if ! ps -p $FRONTEND_PID > /dev/null; then
        echo -e "${RED}Frontend server failed to start${NC}"
        cleanup
        exit 1
    fi
    
    if [ $COUNTER -gt $MAX_STARTUP_WAIT ]; then
        echo -e "${RED}Timeout waiting for frontend server${NC}"
        cleanup
        exit 1
    fi
    
    echo -n "."
    sleep 1
    ((COUNTER++))
done

echo -e "\n${BLUE}All services started successfully!${NC}"
echo -e "Backend server: ${GREEN}http://localhost:$BACKEND_PORT${NC}"
echo -e "Frontend app:   ${GREEN}http://localhost:$FRONTEND_PORT${NC}"
echo -e "\nPress ${YELLOW}Ctrl+C${NC} to stop all services\n"

# Wait for user interrupt
wait
