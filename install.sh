#!/bin/bash
# filepath: /home/jangruhnert/Documents/GitHub/master-project/install.sh

# Install PyTorch Geometric extensions
echo "Installing PyTorch Geometric extensions..."
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.', '')[:3] if torch.version.cuda else 'cpu')")
TORCH_VERSION="2.1.0"
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html"
