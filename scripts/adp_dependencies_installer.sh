#!/bin/bash

# ADP Video Pipeline Dependencies Installer for A10 GPU
# Comprehensive system setup script with failsafe mechanisms

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root. Please run as a regular user with sudo privileges."
    fi
}

# Check GPU availability
check_gpu() {
    log "Checking for NVIDIA GPU..."
    if ! command -v nvidia-smi &> /dev/null; then
        warn "nvidia-smi not found. GPU drivers may not be installed."
        return 1
    fi
    
    local gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
    info "Detected GPU: $gpu_info"
    
    if [[ $gpu_info == *"A10"* ]]; then
        log "NVIDIA A10 GPU detected - perfect for this pipeline!"
    else
        warn "GPU detected but not A10. Pipeline should still work but performance may vary."
    fi
}

# Update system packages
update_system() {
    log "Updating system packages..."
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y software-properties-common apt-transport-https ca-certificates curl gnupg lsb-release
}

# Install Python 3.10+ and pip
install_python() {
    log "Installing Python 3.10+ and development tools..."
    
    # Add deadsnakes PPA for latest Python versions
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    
    # Install Python 3.10 and essential tools
    sudo apt install -y \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3.10-distutils \
        python3-pip \
        build-essential \
        cmake \
        pkg-config
    
    # Set Python 3.10 as default python3
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
    
    # Upgrade pip
    python3 -m pip install --upgrade pip setuptools wheel
}

# Install NVIDIA drivers and CUDA toolkit
install_nvidia_cuda() {
    log "Installing NVIDIA drivers and CUDA toolkit..."
    
    # Remove any existing NVIDIA packages to avoid conflicts
    sudo apt remove --purge nvidia* -y || true
    sudo apt autoremove -y
    
    # Install NVIDIA driver (recommended for A10)
    sudo apt install -y nvidia-driver-535 nvidia-utils-535
    
    # Add NVIDIA package repositories
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt update
    
    # Install CUDA toolkit
    sudo apt install -y cuda-toolkit-12-2
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    # Install cuDNN
    sudo apt install -y libcudnn8 libcudnn8-dev
}

# Install multimedia libraries
install_multimedia() {
    log "Installing multimedia processing libraries..."
    
    sudo apt install -y \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        libavresample-dev \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libasound2-dev \
        libportaudio2 \
        libportaudiocpp0 \
        portaudio19-dev \
        libsndfile1-dev \
        libopencv-dev \
        python3-opencv
}

# Install PostgreSQL
install_postgresql() {
    log "Installing PostgreSQL..."
    
    # Install PostgreSQL
    sudo apt install -y postgresql postgresql-contrib postgresql-client
    
    # Start and enable PostgreSQL
    sudo systemctl start postgresql
    sudo systemctl enable postgresql
    
    # Create database user and database
    sudo -u postgres createuser -s postgres || true
    sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';" || true
    sudo -u postgres createdb stadprin || true
    
    info "PostgreSQL installed. Default user: postgres, password: postgres"
}

# Install system libraries for document processing
install_document_libs() {
    log "Installing document processing libraries..."
    
    sudo apt install -y \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        libtesseract-dev \
        libleptonica-dev \
        libxml2-dev \
        libxslt1-dev \
        libffi-dev \
        libssl-dev \
        zlib1g-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libopenjp2-7-dev
}

# Install audio processing libraries
install_audio_libs() {
    log "Installing audio processing libraries..."
    
    sudo apt install -y \
        libsox-dev \
        sox \
        libmad0-dev \
        libgsm1-dev \
        libsox-fmt-all \
        librosa \
        libasound2-dev \
        pulseaudio \
        alsa-utils
}

# Create Python virtual environment
setup_venv() {
    log "Setting up Python virtual environment..."
    
    # Create virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip in venv
    pip install --upgrade pip setuptools wheel
    
    # Install PyTorch with CUDA support
    log "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install core ML/AI dependencies
    log "Installing core ML/AI dependencies..."
    pip install \
        transformers \
        accelerate \
        bitsandbytes \
        qwen-vl-utils \
        huggingface-hub \
        xformers \
        flash-attn --no-build-isolation
    
    # Install web framework dependencies
    log "Installing web framework dependencies..."
    pip install \
        fastapi \
        uvicorn \
        python-multipart \
        pydantic \
        sqlalchemy \
        psycopg2-binary \
        requests
    
    # Install multimedia processing dependencies
    log "Installing multimedia processing dependencies..."
    pip install \
        moviepy \
        opencv-python \
        Pillow \
        librosa \
        soundfile \
        pydub \
        scipy \
        numpy
    
    # Install document processing dependencies
    log "Installing document processing dependencies..."
    pip install \
        docling \
        pypdf2 \
        python-docx \
        openpyxl \
        pandas
    
    # Install monitoring and utility dependencies
    log "Installing utility dependencies..."
    pip install \
        psutil \
        tqdm \
        python-dotenv \
        argparse
    
    deactivate
}

# Set up environment variables
setup_environment() {
    log "Setting up environment variables..."
    
    # Create .env file if it doesn't exist
    if [[ ! -f .env ]]; then
        cat > .env << EOF
# CUDA Configuration
TORCH_USE_CUDA_DSA=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTORCH_NO_CUDA_MEMORY_CACHING=1
CUDA_LAUNCH_BLOCKING=1
MAX_JOBS=4

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_DB=stadprin
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_PORT=5432
EOF
        info "Created .env file with default configuration"
    fi
}

# Create necessary directories
setup_directories() {
    log "Creating necessary directories..."
    
    mkdir -p models/prompt_generator
    mkdir -p input
    mkdir -p output
    mkdir -p segments
    mkdir -p audio
    mkdir -p source_data
    
    # Set permissions
    chmod -R 755 models input output segments audio source_data
}

# Initialize database
init_database() {
    log "Initializing database..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run database setup if script exists
    if [[ -f scripts/database_setup.py ]]; then
        python scripts/database_setup.py
        info "Database initialized successfully"
    else
        warn "Database setup script not found. You may need to run it manually."
    fi
    
    deactivate
}

# Verify installation
verify_installation() {
    log "Verifying installation..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Test GPU availability
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
    
    # Test core imports
    python -c "
import transformers
import torch
import psycopg2
import cv2
import librosa
print('All core dependencies imported successfully!')
"
    
    # Test database connection if check_db.py exists
    if [[ -f check_db.py ]]; then
        python check_db.py
    fi
    
    deactivate
    
    log "Installation verification completed!"
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    rm -f cuda-keyring_1.0-1_all.deb
    sudo apt autoremove -y
    sudo apt autoclean
}

# Main installation function
main() {
    log "Starting ADP Video Pipeline installation for NVIDIA A10..."
    
    check_root
    
    # System setup
    update_system
    install_python
    
    # GPU and CUDA setup
    if check_gpu; then
        install_nvidia_cuda
    else
        warn "Proceeding without GPU optimization. Install NVIDIA drivers manually if needed."
    fi
    
    # Install system dependencies
    install_multimedia
    install_postgresql
    install_document_libs
    install_audio_libs
    
    # Python environment setup
    setup_venv
    setup_environment
    setup_directories
    
    # Initialize services
    init_database
    
    # Final verification
    verify_installation
    cleanup
    
    log "Installation completed successfully!"
    info "To activate the environment, run: source venv/bin/activate"
    info "To start the server, run: uvicorn app.main:app --host 0.0.0.0 --port 8000"
    warn "System reboot recommended to ensure all drivers are properly loaded."
}

# Run main function
main "$@"