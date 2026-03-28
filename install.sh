#!/bin/bash
# filepath: /home/jangruhnert/Documents/GitHub/master-project/install.sh

export HRL_ROOT=$(pwd)
mkdir -p "$HRL_ROOT/external/"

export CALVIN_ENV_ROOT="$HRL_ROOT/external/calvin_env_modified"
if [ ! -d "$CALVIN_ENV_ROOT" ] ; then
    echo "Cloning calvin_env_modified..."
    git clone --recursive https://github.com/JRGruhnert/calvin_env_modified.git $CALVIN_ENV_ROOT
    cd "$CALVIN_ENV_ROOT"
else
    echo "Updating calvin_env_modified..."
    cd "$CALVIN_ENV_ROOT"
    git pull --recurse-submodules
fi
pip install .


# Install riepybdlib
export RIEPYBDLIB_ROOT="$HRL_ROOT/external/riepybdlib"
if [ ! -d "$RIEPYBDLIB_ROOT" ] ; then
    echo "Cloning riepybdlib..."
    git clone https://github.com/vonHartz/riepybdlib.git $RIEPYBDLIB_ROOT
    cd "$RIEPYBDLIB_ROOT"
else
    echo "Updating riepybdlib..."
    cd "$RIEPYBDLIB_ROOT"
    git pull --recurse-submodules
fi
pip install .

# Install TAPAS
export TAPAS_ROOT="$HRL_ROOT/external/tapas_gmm_modified"
if [ ! -d "$TAPAS_ROOT" ] ; then
    echo "Cloning tapas modified..."
    git clone https://github.com/JRGruhnert/TapasCalvin.git $TAPAS_ROOT
    cd "$TAPAS_ROOT"
    git checkout main  # Switch to main branch (separation branch)
else
    echo "Updating tapas modified..."
    cd "$TAPAS_ROOT"
    git pull --recurse-submodules
fi
pip install .

cd $HRL_ROOT

CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.', '')[:3] if torch.version.cuda else 'cpu')")
TORCH_VERSION="2.1.0"
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html"

pip install .
