# 🚀 GPU-Accelerated Entity Resolver

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.5%2B-green)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A high-performance, production-ready Python package for entity resolution and deduplication at scale. Built on NVIDIA RAPIDS (`cudf`, `cuML`, `cupy`), the project keeps the entire pipeline on the GPU to minimize transfers and maximize throughput.

## 📋 Table of Contents

- [Features](#features)
- [Performance](#performance)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Advanced Usage](#advanced-usage)
- [Configuration](#configuration)
- [Pipeline Architecture](#pipeline-architecture)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## ✨ Features

### Core Capabilities
- **🏃‍♂️ GPU-Accelerated Pipeline**: End-to-end GPU execution minimizes data transfer bottlenecks
- **🔍 Intelligent Address Processing**: Leverages `libpostal` for robust address parsing and standardization
- **💾 Model Persistence**: Save and load fitted models for production deployment
- **📊 Confidence Scoring**: Multi-factor scoring system for match quality assessment
- **🔄 Incremental Processing**: Transform new data using pre-fitted models

### Multi-Stream Vectorization
Creates rich entity embeddings by combining three complementary approaches:

| Stream       | Method        | Purpose                  | Key Technique                 |
|--------------|---------------|--------------------------|-------------------------------|
| **Syntactic**| TF-IDF        | Character-level patterns | N-gram analysis (3–5 chars)   |
| **Phonetic** | Metaphone     | Sound-based matching     | Phonetic encoding             |
| **Semantic** | Transformers  | Contextual meaning       | `all-MiniLM-L6-v2` embeddings |

### Advanced Clustering
- **HDBSCAN + SNN Ensemble**: Combines density-based clustering with graph-based rescue mechanism
- **UMAP Manifold Learning**: Ensemble of models with randomized hyperparameters for robust dimensionality reduction
- **Automatic Chain Detection**: Identifies and numbers entities with multiple locations (e.g., “Starbucks – 1”, “Starbucks – 2”)

## ⚡ Performance

- Designed for **hundreds of thousands to millions** of rows on a single GPU.
- Memory scales primarily with TF-IDF vocabulary size and UMAP neighbor graphs.
- **Performance depends on hardware and configuration.** On an NVIDIA A100 40GB, end-to-end runs with the default settings typically process ~200k rows in **tens of minutes to a few hours**. Lighter settings (smaller vocabularies / fewer UMAP epochs) run faster; aggressive settings (larger vocabularies, more ensemble members) improve quality at the cost of runtime and memory.

## 💻 Requirements

### System Requirements
- **GPU**: NVIDIA GPU with Compute Capability 7.0+ (e.g., V100, T4, RTX 2080+)
- **CUDA**: Version 11.5 or higher
- **Memory**:
  - GPU: Minimum 8 GB VRAM (16 GB+ recommended for large datasets)
  - System RAM: ~2× your dataset size

### Software Dependencies
- Python 3.8+
- RAPIDS Suite (`cudf`, `cuml`, `cupy`)
- Additional CPU libraries: `sentence-transformers`, `phonetics`, `python-postal`

## 📦 Installation

### Step 1: Install RAPIDS

Conda is recommended:

```bash
# Create a new environment with RAPIDS
conda create -n rapids-24.12 -c rapidsai -c conda-forge -c nvidia \
    rapids=24.12 python=3.11 cuda-version=11.8

# Activate the environment
conda activate rapids-24.12
```

For detailed instructions, see the RAPIDS installation guide.

### Step 2: Install the Package

```bash
# Option A: Install directly from GitHub (Recommended)
pip install git+https://github.com/andrewjordan3/entity-resolver.git

# Option B: Install from a local clone
git clone https://github.com/andrewjordan3/entity-resolver.git
cd entity-resolver
pip install .
```

### Step 3: Verify Installation

```python
import entity_resolver
print(entity_resolver.__version__)  # Should print: 0.1.0
```

## 🚀 Quick Start

```python
import pandas as pd
from entity_resolver import EntityResolver

# Load your data
df = pd.DataFrame({
    'company_name': [
        'Acme Corp', 'ACME Corporation', 'Acme Corp.',
        'TechStart Inc', 'Tech Start LLC'
    ],
    'address': [
        '123 Main St, Chicago IL 60601',
        '123 Main Street, Chicago, IL',
        '123 Main St Chicago IL',
        '456 Oak Ave, Hoffman Estates, IL',
        '456 Oak Avenue Hoffman Estates IL'
    ]
})

# Initialize the resolver. 
# This uses the default settings. For custom settings, provide a path
# to your YAML file, e.g., EntityResolver(config_path='my_config.yaml')
resolver = EntityResolver()

# Fit and transform in one step
resolved_df = resolver.fit_transform(df)

# Examine results
print(resolved_df[['company_name', 'canonical_name', 'confidence_score']])

# Generate a detailed report
report = resolver.generate_report(df)
print(f"Reduced {report['summary']['unique_entities_before']} entities "
      f"to {report['summary']['unique_entities_after']}")

# Get records flagged for review
review_df = resolved_df[resolved_df['needs_review']]
print(f"Records needing review: {len(review_df)}")
```

## 🔧 Advanced Usage

### Saving and Loading Models

```python
# Train and save
resolver = EntityResolver(config)
resolver.fit(training_df)
resolver.save_model('models/my_resolver')

# Load and use
resolver = EntityResolver.load_model('models/my_resolver')
new_resolved_df = resolver.transform(new_df)
```

### Custom Configuration

To customize the pipeline, create a `config.yaml` file and pass its path to the resolver. You only need to specify the parameters you want to override.

**Example `my_config.yaml`:**
```yaml
# my_config.yaml
output:
  review_confidence_threshold: 0.80
  split_address_components: true

clusterer:
  hdbscan_params:
    min_cluster_size: 2
  umap_params:
    n_neighbors: 30

validation:
  name_fuzz_ratio: 85
```

**Usage in Python:
```python
from entity_resolver import EntityResolver

# Load the resolver with your custom configuration
resolver = EntityResolver(config_path='my_config.yaml')
```

### Processing Large Datasets

```python
# For datasets larger than GPU memory, process in chunks
chunk_size = 100_000
chunks = []

for i in range(0, len(large_df), chunk_size):
    chunk = large_df.iloc[i:i+chunk_size]
    resolved_chunk = resolver.transform(chunk)
    chunks.append(resolved_chunk)

final_df = pd.concat(chunks, ignore_index=True)
```
**Note**: The fit() or fit_transform() methods must be trained on a dataset that can fit into GPU memory, as they build global models (e.g., vectorizer vocabularies). Once the resolver is fitted, the transform() method can be used to process new, unseen data in chunks

## ⚙️ Configuration

The `ResolverConfig` object controls all aspects of the pipeline:

| Component       | Key Parameters                      | Description                         |
|----------------|-------------------------------------|-------------------------------------|
| `columns`      | `entity_col`, `address_cols`        | Input column mapping                |
| `normalization`| `replacements`, `suffixes_to_remove`| Text cleaning rules                 |
| `vectorizer`   | `encoders`, `stream_proportions`    | Feature extraction settings         |
| `clusterer`    | `umap_params`, `hdbscan_params`     | Clustering hyperparameters          |
| `validation`   | `name_fuzz_ratio`, `address_fuzz_ratio` | Match validation thresholds     |
| `output`       | `review_confidence_threshold`       | Output formatting options           |

See the **Configuration Guide** for detailed documentation.

## 🏗️ Pipeline Architecture

```mermaid
graph LR
    A[Input DataFrame] --> B[Text Normalization]
    B --> C[Address Processing]
    C --> D[Multi-Stream Vectorization]
    D --> E[UMAP Ensemble]
    E --> F[HDBSCAN + SNN Clustering]
    F --> G[Validation & Merging]
    G --> H[Canonical Mapping]
    H --> I[Confidence Scoring]
    I --> J[Output DataFrame]
```

### Pipeline Stages

**Preprocessing (Steps 1–2)**
- Text normalization and standardization  
- Address parsing with libpostal  
- Name consolidation for shared addresses

**Vectorization (Step 3)**
- TF-IDF character n-grams  
- Phonetic encoding  
- Semantic embeddings  
- Energy balancing across streams

**Clustering (Steps 4–5)**
- UMAP ensemble for dimensionality reduction  
- HDBSCAN for core clustering  
- SNN graph for noise rescue  
- Cluster validation and merging

**Finalization (Steps 6–7)**
- Canonical entity selection  
- Chain entity numbering  
- Confidence scoring  
- Review flagging

## 📚 API Reference

### Main Classes
- `EntityResolver`: Main orchestrator class  
- `ResolverConfig`: Configuration management  
- Component classes: `TextNormalizer`, `AddressProcessor`, `MultiStreamVectorizer`, `EntityClusterer`, `ClusterValidator`, `ClusterMerger`, `ClusterRefiner`, `ConfidenceScorer`

### Key Methods

```python
# EntityResolver methods
resolver.fit(df)                 # Train on data
resolver.transform(df)           # Apply to new data
resolver.fit_transform(df)       # Train and transform
resolver.save_model(path)        # Save fitted model
resolver.load_model(path)        # Load saved model
resolver.get_review_dataframe()  # Get review summary
resolver.generate_report(df)     # Generate statistics
```

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```python
# Reduce batch sizes
config.vectorizer.semantic_batch_size = 256
config.clusterer.cosine_consensus_batch_size = 1024
```

**Import Errors**
```bash
# Verify RAPIDS installation
python -c "import cudf; print(cudf.__version__)"
```

**Poor Clustering Results**
```python
# Adjust sensitivity
config.clusterer.hdbscan_params['min_cluster_size'] = 2
config.validation.name_fuzz_ratio = 80
```

See the **Troubleshooting Guide** for more solutions.

## 🤝 Contributing

Contributions are welcome!

```bash
# Development setup
git clone https://github.com/andrewjordan3/entity-resolver.git
cd entity-resolver
pip install -e ".[dev]"
pytest tests/
```

## 📝 Citation

If you use this package in your research, please cite:

```bibtex
@software{entity_resolver,
  author = {Andrew Jordan},
  title = {GPU-Accelerated Entity Resolver},
  year = {2025},
  url = {https://github.com/andrewjordan3/entity-resolver}
}
```

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA RAPIDS team for the GPU acceleration framework  
- OpenCage for libpostal address parsing  
- Hugging Face for sentence-transformers  

Maintained by: **Andrew Jordan**  
Contact: **andrewjordan3@gmail.com**  
Last Updated: **August 2025**

