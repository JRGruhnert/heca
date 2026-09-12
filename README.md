# _HECA_ (Hierarchical Entity-Centric Agent)

Heca is a framework for training, testing and evaluating hierarchical agents that operate on a graph-based world representation. This makes it applicable for federated use.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/JRGruhnert/heca.git
cd heca
```

### 2. Create a Virtual Environment (Recommended)

The project was tested and developed with python version 3.10

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

## Data

The `data/` tree is git-ignored and shipped separately as a single zip (the raw
`.h5` demos alone are >120 GB, so they stay out of the bundle):

```bash
./download_data.sh          # fetch and unpack ./data, sha256-verified
./download_data.sh --list   # show what is inside the archive
```

| bundle | contents | size |
|---|---|---|
| default | everything except `*.h5` and `*.zip` | ~32 GB, 261k files |
| built with `--trim` | additionally drops `data/old`, `data/scenes/calvin`, `*.png` | ~6.5 GB, ~900 files |

`download_data.sh` needs no credentials when the bundle is published publicly
(`ZENODO_RECORD` / `DATA_URL` in `data_bundle.env`), and it can also read from an
rclone remote or a local file:

```bash
./download_data.sh --record 1234567                  # Zenodo record id
./download_data.sh --url https://.../data.zip        # any direct link
./download_data.sh --remote gdrive:heca-data         # rclone remote
./download_data.sh --from-file /media/usb/data.zip   # USB stick
```

### Publishing a new bundle (maintainer)

```bash
export ZENODO_TOKEN=...     # token with deposit:write + deposit:actions
./upload_data.sh            # builds data.zip, uploads, publishes, prints the links
```

Paste the printed `ZENODO_RECORD` / `DATA_URL` into `data_bundle.env` and commit
it, so a fresh clone can fetch the data with `./download_data.sh`.

Google Drive, Dropbox or OneDrive work as well, through rclone:

```bash
./upload_data.sh --backend rclone --remote dropbox:heca-data
```

Both scripts read their shared settings from `data_bundle.env` (what to include,
where it goes), and every value can be overridden with a flag:
`--data-dir`, `--out`, `--trim`, `--backend`, `--remote`, `--no-upload`, …

> Free-tier note: the default bundle (~32 GB) is bigger than the free Google
> Drive (15 GB) and Dropbox (2 GB) quotas; Zenodo allows 50 GB per record, and a
> `--trim` bundle (~6.5 GB) fits on a free Drive.
>
> The `.h5` demos are intentionally not part of the bundle — fetch or regenerate
> `data/scenes/ogbench/scene*/experts/*/demos.h5` separately if you want to
> (re)fit conditions on a new machine.

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
