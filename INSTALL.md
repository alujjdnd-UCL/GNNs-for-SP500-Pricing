# Installation Guide

This guide provides detailed instructions for setting up the replication package environment.

## Prerequisites

Before starting, ensure you have:
- Python 3.8 or higher
- pip package manager
- (Optional but recommended) NVIDIA GPU with CUDA 11.8+
- 16GB+ RAM
- ~1GB free disk space

## Installation Steps

### 1. Clone or Download Repository

If using Git:
```bash
git clone <repository-url>
cd <repository-name>
```

Or download and extract the ZIP archive.

### 2. Create Virtual Environment

We strongly recommend using a virtual environment to avoid dependency conflicts.

#### On Linux/macOS (bash/zsh):
```bash
python -m venv venv
source venv/bin/activate
```

#### On macOS/Linux (csh/tcsh):
```bash
python -m venv venv
source venv/bin/activate.csh
```

#### On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### On Windows (Command Prompt):
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

### 3. Upgrade pip and setuptools

```bash
pip install --upgrade pip setuptools wheel
```

### 4. Install PyTorch with CUDA Support

The package requires PyTorch with CUDA support for GPU acceleration.

#### For CUDA 11.8 (recommended):
```bash
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

#### For CPU-only (not recommended, very slow):
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu
```

#### For other CUDA versions:
Visit https://pytorch.org/get-started/locally/ and follow instructions for your system.

### 5. Install PyTorch Geometric Dependencies

PyTorch Geometric requires specific versions of extension libraries:

```bash
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 \
    -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
```

If you encounter issues, see: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### 6. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch Geometric (2.6.1)
- PyTorch Geometric Temporal (0.54.0)
- pandas, numpy, scipy
- matplotlib, seaborn (for visualization)
- scikit-learn (for metrics)
- statsmodels (for ARIMA baseline)
- Technical analysis library (ta)
- RDFlib, SPARQLWrapper (for Wikidata)
- Jupyter (for notebooks)

## Verification

### Verify Installation

Run the following commands to verify your installation:

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"
python -c "import torch_geometric_temporal; print('PyG Temporal:', torch_geometric_temporal.__version__)"
```

Expected output (versions may vary slightly):
```
PyTorch: 2.5.1+cu118
CUDA Available: True
PyG: 2.6.1
PyG Temporal: 0.54.0
```

### Test CUDA GPU

To verify GPU is accessible:
```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available')"
```

### Run Quick Test

Test that the package structure is correct:
```bash
cd experiments
export PYTHONPATH=..  # On Windows: set PYTHONPATH=..
python -c "from src.models import MLP, LSTM, GRU, TGCN, DCRNN, A3TGCN; print('All models imported successfully')"
```

## Troubleshooting

### Issue: CUDA out of memory

**Solution**: Reduce batch size or use a smaller model. Edit `experiments/config/experiment_config.py` to adjust parameters.

### Issue: torch-scatter installation fails

**Solution**: Install pre-built binaries manually:
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
```

Or install from source (requires C++ compiler):
```bash
pip install torch-scatter --no-binary torch-scatter
```

### Issue: "No module named 'src'" error

**Solution**: Ensure you set PYTHONPATH before running scripts:
```bash
cd experiments
export PYTHONPATH=..  # On Windows: set PYTHONPATH=..
```

Or add to your shell profile (~/.bashrc, ~/.zshrc):
```bash
export PYTHONPATH="/path/to/replication-package:$PYTHONPATH"
```

### Issue: Wikidata queries timeout

**Solution**: The Wikidata fetching script includes retry logic. If persistent, check:
1. Internet connectivity
2. Wikidata SPARQL endpoint status: https://query.wikidata.org/
3. Increase timeout in `data/get_wikidata.py`

### Issue: Missing data files

**Solution**: Ensure you have prepared data according to `data/documentation/DATA_FORMAT.md`. The package does NOT include raw stock price data due to licensing restrictions.

### Issue: Incompatible dependency versions

**Solution**: Create a fresh virtual environment and reinstall:
```bash
deactivate  # If in a venv
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
# Follow installation steps 4-6 above
```

### Issue: Different Python version

**Solution**: The package requires Python 3.8+. Check your version:
```bash
python --version
```

If needed, install a compatible Python version:
- **Linux**: Use your package manager (apt, yum, etc.)
- **macOS**: Use Homebrew: `brew install python@3.10`
- **Windows**: Download from https://www.python.org/downloads/

## Platform-Specific Notes

### Linux (Ubuntu/Debian)

Install system dependencies:
```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip build-essential
```

### macOS

Install Xcode command line tools:
```bash
xcode-select --install
```

Note: CUDA is not available on macOS. Use CPU-only PyTorch or run on Linux/Windows with NVIDIA GPU.

### Windows

Install Visual C++ Build Tools from:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

Required for compiling some PyTorch extensions.

## GPU Lab Setup (UCL Students)

For UCL students using GPU lab machines in MPEB 1.05/1.21:

```bash
# SSH into GPU lab
ssh <username>@<gpu-lab-machine>

# Load module (if available)
module load python/3.10
module load cuda/11.8

# Create virtual environment
python -m venv ~/venvs/replication
source ~/venvs/replication/bin/activate.csh  # csh shell

# Follow installation steps 3-6 above
```

## Next Steps

After successful installation:
1. Read `data/documentation/DATA_FORMAT.md` to understand data requirements
2. Prepare your data files
3. Review `experiments/config/experiment_config.py` for configuration options
4. Run experiments: `cd experiments && python run_all.py`

## Getting Help

If you encounter issues not covered here:
1. Check error messages carefully
2. Verify all prerequisites are met
3. Ensure data is formatted correctly
4. Open an issue on the repository with:
   - Your operating system
   - Python version
   - Full error message
   - Steps to reproduce
