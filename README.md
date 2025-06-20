# Repository for Research Paper "Image Style Transfer Model Retrieval using Transformer Encoder and Learning to Rank"

## Project Directory Structure

```
data/
├── content-images.xlsx               # Metadata for content images
├── data.csv                          # Main data CSV
├── models.xlsx                       # Metadata for models
├── content-images/                   # Directory for content images
├── generated-images/                 # Directory for generated images
├── models/                           # Directory for downloaded models

istmr/
├── __init__.py
├── dataset.py                        # Dataset utilities
├── models.py                         # Proposed method implementation
├── utils.py                          # Utility functions (download, image gen, training, etc.)

allRank/                              # Third-party module for neuralNDCG implementation

main.py                               # Entrypoint for CLI commands (download, generate, etc.)
pyproject.toml                        # Python project metadata (for uv)
uv.lock                               # uv lock file
.gitignore                            # Git ignore rules
.gitmodules                           # Git submodules
.python-version                       # Python version file
README.md                             # Project documentation
```

## How to install

### Clone repository

```bash
git clone --recursive https://github.com/ohshimalab/istmr.git
```

### Install uv

We use uv as the Python package manager.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install libraries

```bash
uv sync
```

## How to run codes

### Download content images

```bash
uv run python main.py download-content-images \
  --content-image-list-excel ./data/content-images.xlsx \
  --output-dir ./data/content-images/
```

### Download Civitai models

Some models require a Civitai API token for downloading.
Civitai API token can be issued with the [following steps](https://education.civitai.com/civitais-guide-to-downloading-via-api/)

**Note**: Require ~20GB of free disk space

```bash
uv run python main.py download-civitai-models \
  --model-list-excel ./data/models.xlsx \
  --output-dir ./data/models/ \
  --token <your_civitai_api_token>
```

### Generate image for every image-model pair

**Note**: Required GPU for generating. Generating 1 image take about ~3 seconds depends on the GPU.

```bash
uv run python main.py generate-pair \
  --content-image-dir ./data/content-images/ \
  --model-dir ./data/models/ \
  --output-dir ./data/generated-images/
  --data-csv ./data/data.csv
  --model-excel ./data/models.xslx
```

### Train and evaluate the proposed method model

```bash
uv run python main.py train-and-evaluate-proposed-method \
  --generated-image-dir ./data/generated-images/ \
  --alpha 0.5 \
  --vit-name "vit_base_patch16_224" \
  --num-transformer-layers 2 \
  --num-heads 8 \
  --batch-size 16 \
  --max-epochs 30 \
  --num-workers 4 \
  --data-csv ./data/data.csv \
  --model-save-path ./trained-model.pth \
  --vit-save-path ./vit-model.pth \
  --lr 1e-5 \
  --seed 0 \
  --test-results-csv ./test-result.csv
```
