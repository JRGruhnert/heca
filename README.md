# _HECA_ (Hierarchical Entity-Centric Agents)

Heca is a framework for training, testing and evaluating hierarchical agents that operate on a graph-based world representation. This makes it applicable for federated use.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/JRGruhnert/master-project.git
cd master-project
```

### 2. Create a Virtual Environment (Recommended)

The project was tested with python version 3.10

```bash
conda create -n heca_env python=3.10
conda activate heca_env
pip install -e .
```

You may also have to install pytorch_geometric packages after installing pytorch:

```bash
chmod +x install.sh
bash install.sh
```

## Acknowledgments

This project is built upon the following projects:

- [TAPAS](https://github.com/robot-learning-freiburg/TAPAS)
- [Molmo2](https://github.com/allenai/molmo2)
- [DinoV3](https://github.com/facebookresearch/dinov3)
- [CALVIN](https://github.com/mees/calvin)
- [OGBench](https://github.com/seohongpark/ogbench)
- [riepybdlib](https://github.com/vonHartz/riepybdlib)

## How to use it

- TODO
