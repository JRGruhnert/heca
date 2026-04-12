pip install -e .

# PyG extensions (not in setup.cfg)
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.', '')[:3] if torch.version.cuda else 'cpu')")
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f "https://data.pyg.org/whl/torch-2.1.0+cu${CUDA_VERSION}.html"