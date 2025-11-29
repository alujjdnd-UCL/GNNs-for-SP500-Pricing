"""
Environment validation script for replication package.

This script checks that all dependencies are installed correctly
and the environment is properly configured.

Usage:
    cd tests
    export PYTHONPATH=..
    python validate_setup.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def print_status(check_name, passed, message=""):
    """Print colored status message."""
    status = "✓ PASS" if passed else "✗ FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} | {check_name}")
    if message:
        print(f"       {message}")
    return passed


def check_python_version():
    """Check Python version is 3.8+."""
    version = sys.version_info
    passed = version.major == 3 and version.minor >= 8
    msg = f"Python {version.major}.{version.minor}.{version.micro}"
    return print_status("Python Version (3.8+)", passed, msg)


def check_pytorch():
    """Check PyTorch is installed."""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda_available else "CPU only"
        msg = f"PyTorch {version} | {device}"
        return print_status("PyTorch Installation", True, msg)
    except ImportError as e:
        return print_status("PyTorch Installation", False, str(e))


def check_pytorch_geometric():
    """Check PyTorch Geometric is installed."""
    try:
        import torch_geometric
        version = torch_geometric.__version__
        msg = f"PyG {version}"
        return print_status("PyTorch Geometric", True, msg)
    except ImportError as e:
        return print_status("PyTorch Geometric", False, str(e))


def check_pytorch_geometric_temporal():
    """Check PyTorch Geometric Temporal is installed."""
    try:
        import torch_geometric_temporal
        version = torch_geometric_temporal.__version__
        msg = f"PyG Temporal {version}"
        return print_status("PyTorch Geometric Temporal", True, msg)
    except ImportError as e:
        return print_status("PyTorch Geometric Temporal", False, str(e))


def check_core_dependencies():
    """Check core scientific libraries."""
    libs = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "sklearn": "scikit-learn"
    }
    
    all_passed = True
    for import_name, display_name in libs.items():
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            msg = f"{display_name} {version}"
            passed = print_status(f"{display_name}", True, msg)
            all_passed = all_passed and passed
        except ImportError as e:
            passed = print_status(f"{display_name}", False, str(e))
            all_passed = False
    
    return all_passed


def check_models():
    """Check that all model classes can be imported."""
    try:
        from src.models import MLP, LSTM, GRU, TGCN, DCRNN, A3TGCN
        msg = "All 6 models imported successfully"
        return print_status("Model Imports", True, msg)
    except ImportError as e:
        return print_status("Model Imports", False, str(e))


def check_datasets():
    """Check that dataset classes can be imported."""
    try:
        from src.datasets import SP500Stocks
        msg = "Dataset class imported successfully"
        return print_status("Dataset Imports", True, msg)
    except ImportError as e:
        return print_status("Dataset Imports", False, str(e))


def check_utilities():
    """Check that utility modules can be imported."""
    try:
        from src.utils import train, evaluate
        msg = "Training and evaluation utilities imported"
        return print_status("Utility Imports", True, msg)
    except ImportError as e:
        return print_status("Utility Imports", False, str(e))


def check_data_directory():
    """Check that data directory structure exists."""
    base_path = Path(__file__).parent.parent / "data" / "SP500"
    
    required_dirs = [
        base_path / "raw",
        base_path / "raw" / "phases",
        base_path / "processed"
    ]
    
    all_exist = all(d.exists() for d in required_dirs)
    
    if all_exist:
        msg = "Data directory structure is correct"
    else:
        missing = [str(d.relative_to(base_path.parent.parent)) for d in required_dirs if not d.exists()]
        msg = f"Missing directories: {', '.join(missing)}"
    
    return print_status("Data Directory Structure", all_exist, msg)


def check_data_files():
    """Check if required data files are present (optional check)."""
    base_path = Path(__file__).parent.parent / "data" / "SP500"
    
    required_files = [
        base_path / "filtered_symbols.csv",
        base_path / "symbol_industry.csv"
    ]
    
    existing = [f for f in required_files if f.exists()]
    
    if len(existing) == len(required_files):
        msg = "All required metadata files present"
        return print_status("Data Files (Metadata)", True, msg)
    else:
        missing = [f.name for f in required_files if not f.exists()]
        msg = f"Missing files: {', '.join(missing)} (May need to be added)"
        return print_status("Data Files (Metadata)", False, msg)


def check_config():
    """Check that configuration file can be imported."""
    try:
        from experiments.config.experiment_config import (
            EPOCHS, MODELS_TO_EVALUATE, EDGE_FORMATION_STRATEGIES
        )
        msg = f"Config loaded: {len(MODELS_TO_EVALUATE)} models, {len(EDGE_FORMATION_STRATEGIES)} strategies"
        return print_status("Configuration", True, msg)
    except ImportError as e:
        return print_status("Configuration", False, str(e))


def main():
    """Run all validation checks."""
    print("\n" + "="*80)
    print("REPLICATION PACKAGE - ENVIRONMENT VALIDATION")
    print("="*80 + "\n")
    
    checks = [
        ("Core", [
            check_python_version,
            check_pytorch,
            check_pytorch_geometric,
            check_pytorch_geometric_temporal,
            check_core_dependencies
        ]),
        ("Package Structure", [
            check_models,
            check_datasets,
            check_utilities,
            check_config
        ]),
        ("Data Setup", [
            check_data_directory,
            check_data_files
        ])
    ]
    
    all_passed = True
    
    for category, check_funcs in checks:
        print(f"\n{category}:")
        print("-" * 80)
        category_passed = all(func() for func in check_funcs)
        all_passed = all_passed and category_passed
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Environment is ready!")
    else:
        print("✗ SOME CHECKS FAILED - Please review errors above")
        print("\nRefer to INSTALL.md for installation instructions")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
