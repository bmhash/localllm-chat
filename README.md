# LocalLLM-Chat

A modern, efficient, and privacy-focused chat interface for running Large Language Models locally. Built with Python FastAPI backend and Next.js frontend, it leverages HuggingFace models while keeping all interactions on your machine.

## Features

- 🔒 Complete privacy - all processing happens locally
- 🚀 Fast and responsive chat interface
- 💾 Efficient model loading (lazy loading)
- 🎯 Support for multiple LLM models
- 🎨 Modern UI with Next.js
- 🔄 Real-time streaming responses
- 📱 Responsive design for all devices
- 🤗 Easy model management with HuggingFace integration
- 🔐 Gated model access support for controlled deployment

## Prerequisites

- Python 3.12 or higher
- Node.js 14 or higher
- CUDA-capable GPU (NVIDIA RTX series recommended)
- Git

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/bmhash/localllm-chat.git
cd localllm-chat
```

2. Run the installation script:
```bash
./install.sh
```

3. Create a `.env` file from the example:
```bash
cp .env.example .env
```

4. Add your Hugging Face token to the `.env` file:
```
HF_TOKEN=your_token_here
```

5. Start the application:
```bash
./start.sh
```

6. Open your browser and navigate to http://localhost:3000

## Models

The following models are supported:
- Llama 3.2 3B (default)
- Deepseek Coder 7B
- CodeLlama 7B/13B
- Mistral 7B
- Deepseek MoE 16B

Models are loaded on demand to optimize memory usage. The first request for each model may take a few moments as the model is loaded.

## System Requirements

- **RAM**: Minimum 16GB, Recommended 32GB
- **GPU**: NVIDIA GPU with at least 8GB VRAM
- **Storage**: At least 50GB free space
- **OS**: Linux (tested on Ubuntu 20.04+)

## Project Structure

```
.
├── server.py           # FastAPI backend server
├── install.sh          # Installation script
├── start.sh           # Startup script
├── install_models.py  # Model installation and management
├── config/           # Configuration files
│   └── models.py     # Model definitions and settings
├── chat_formatting.py # Chat message formatting utilities
├── requirements.txt   # Python dependencies
├── chat-interface/    # Next.js frontend
│   ├── src/
│   │   └── components/
│   │       └── ChatInterface.tsx  # Main chat component
└── README.md         # This file
```

## Development

To run the frontend in development mode:
```bash
cd chat-interface
npm run dev
```

To run the backend in development mode:
```bash
source .venv/bin/activate  # or your virtual environment path
python server.py
```

## Dependencies

Key dependencies include:
- FastAPI 0.109.0
- PyTorch 2.2.1+cu118
- Transformers 4.36.2
- Next.js (latest)
- Python 3.12+

For a complete list of Python dependencies, see `requirements.txt`.

## Privacy & Security

This application is designed with privacy in mind:
- All processing happens locally on your machine
- No data is sent to external servers (except for model downloads)
- Models are downloaded once and stored locally
- Chat history stays on your computer

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - feel free to use this project for your own purposes.

## Acknowledgments

- Built with FastAPI and Next.js
- Uses Hugging Face Transformers
- Inspired by various chat interfaces in the open-source community
