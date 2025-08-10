# entity_resolver/persistence.py
"""
This module provides functions for saving and loading the state of a fitted
EntityResolver, including its configuration, trained models, and canonical map.
"""

import pickle
from pathlib import Path
import logging
from typing import Dict, Any, Optional

# Import necessary libraries for model serialization
import cudf
import cuml
import cupy
from sentence_transformers import SentenceTransformer

# Set up a logger for this module
logger = logging.getLogger(__name__)

# --- PUBLIC API ---

def save_model(resolver_instance, directory_path: str) -> None:
    """
    Saves all components of a fitted resolver to a specified directory.

    This function creates a structured directory with subfolders for each major
    component of the resolver, ensuring a clean and organized model artifact.

    Args:
        resolver_instance: The fitted EntityResolver instance to save.
        directory_path: The path to the directory where model components will be saved.
    """
    if not resolver_instance._is_fitted:
        raise RuntimeError("Cannot save an unfitted model. Please call fit() or fit_transform() first.")
    
    logger.info(f"Saving trained resolver to directory: {directory_path}...")
    p = Path(directory_path)
    p.mkdir(parents=True, exist_ok=True)

    # 1. Save the main configuration file to the root of the model directory.
    with open(p / 'config.pkl', 'wb') as f:
        pickle.dump(resolver_instance.config, f)

    # 2. Save components from each sub-module into their own dedicated subdirectories.
    # This keeps the model structure clean and avoids filename collisions.
    _save_vectorizer_components(resolver_instance.vectorizer, p / 'vectorizer')
    _save_clusterer_components(resolver_instance.clusterer, p / 'clusterer')
    _save_canonical_map(resolver_instance, p)
    
    logger.info("Model saved successfully.")


def load_model_components(directory_path: str) -> Dict[str, Any]:
    """
    Loads all resolver components from a directory on disk.

    This function reconstructs a dictionary of all the necessary components
    which can then be used to initialize a new EntityResolver instance.

    Args:
        directory_path: The path to the directory containing the saved model.
        
    Returns:
        A dictionary containing all the loaded components required to
        reconstruct a complete and functional EntityResolver instance.
    """
    logger.info(f"Loading trained resolver components from: {directory_path}")
    p = Path(directory_path)
    
    if not p.exists():
        raise FileNotFoundError(f"Model directory not found: {directory_path}")
    
    # This dictionary will hold all the reconstructed parts of the model.
    components = {}
    
    # 1. Load the main configuration file. This is a critical component.
    config_path = p / 'config.pkl'
    if not config_path.exists():
        raise FileNotFoundError(f"Required config file not found: {config_path}")
    
    with open(config_path, 'rb') as f:
        components['config'] = pickle.load(f)
    
    # 2. Load all other model components from their respective locations.
    components['vectorizer_models'] = _load_vectorizer_components(p / 'vectorizer')
    components['clusterer_models'] = _load_clusterer_components(p / 'clusterer')
    components['canonical_map'] = _load_canonical_map(p)
    
    logger.info("✓ All model components loaded successfully.")
    return components


# === PRIVATE HELPER FUNCTIONS FOR SAVING ===

def _save_vectorizer_components(vectorizer_instance, directory: Path) -> None:
    """Saves all models related to the vectorization process."""
    directory.mkdir(exist_ok=True)
    
    # Get the dictionaries of trained models from the vectorizer instance.
    models_to_save = vectorizer_instance.get_models()
    
    # Save the simple encoders (TF-IDF, CountVectorizer) using pickle.
    with open(directory / 'trained_encoders.pkl', 'wb') as f:
        pickle.dump(models_to_save['encoders'], f)
        
    # Save the dimensionality reduction models.
    with open(directory / 'reduction_models.pkl', 'wb') as f:
        pickle.dump(models_to_save['reduction_models'], f)
        
    # The SentenceTransformer model has its own saving method which creates a directory.
    if 'semantic_model' in models_to_save['encoders']:
        semantic_model = models_to_save['encoders']['semantic_model']
        semantic_model.save(str(directory / 'semantic_model'))


def _save_clusterer_components(clusterer_instance, directory: Path) -> None:
    """Saves all models related to the clustering process."""
    directory.mkdir(exist_ok=True)

    # Get the dictionaries of trained models from the clusterer instance.
    models_to_save = clusterer_instance.get_models()

    # Save the main HDBSCAN model using cuML's native save function.
    if models_to_save.get('cluster_model'):
        cuml.common.save(models_to_save['cluster_model'], directory / 'cluster_model.model')
    
    # Save the UMAP ensemble models individually.
    if 'umap_reducer_ensemble' in models_to_save:
        umap_ensemble = models_to_save['umap_reducer_ensemble']
        # Create a subdirectory for the ensemble to keep the parent directory clean.
        ensemble_dir = directory / 'umap_ensemble'
        ensemble_dir.mkdir(exist_ok=True)
        for i, umap_model in enumerate(umap_ensemble):
            cuml.common.save(umap_model, ensemble_dir / f"umap_model_{i}.model")


def _save_canonical_map(resolver_instance, directory: Path) -> None:
    """Saves the canonical map DataFrame to a Parquet file."""
    if resolver_instance.canonical_map_ is not None:
        resolver_instance.canonical_map_.to_parquet(directory / 'canonical_map.parquet')


# === PRIVATE HELPER FUNCTIONS FOR LOADING ===

def _load_vectorizer_components(directory: Path) -> Optional[Dict[str, Any]]:
    """Loads all models related to the vectorization process."""
    if not directory.exists():
        logger.warning(f"Vectorizer directory not found: {directory}. Skipping.")
        return None
        
    # Load the pickled dictionaries.
    with open(directory / 'trained_encoders.pkl', 'rb') as f:
        trained_encoders = pickle.load(f)
    with open(directory / 'reduction_models.pkl', 'rb') as f:
        reduction_models = pickle.load(f)
        
    # The SentenceTransformer was saved in its own directory, so we need to load it
    # back from there and place it back into the `trained_encoders` dictionary.
    semantic_model_path = directory / 'semantic_model'
    if semantic_model_path.exists():
        trained_encoders['semantic_model'] = SentenceTransformer(str(semantic_model_path))
        
    return {
        'trained_encoders': trained_encoders,
        'reduction_models': reduction_models
    }


def _load_clusterer_components(directory: Path) -> Optional[Dict[str, Any]]:
    """Loads all models related to the clustering process."""
    if not directory.exists():
        logger.warning(f"Clusterer directory not found: {directory}. Skipping.")
        return None
        
    components = {}
    
    # Load the main HDBSCAN model.
    cluster_model_path = directory / 'cluster_model.model'
    if cluster_model_path.exists():
        components['cluster_model'] = cuml.common.load(cluster_model_path)
    
    # Load the UMAP ensemble from its dedicated subdirectory.
    ensemble_dir = directory / 'umap_ensemble'
    if ensemble_dir.exists():
        ensemble = []
        # Sort the files to ensure they are loaded in the same order they were saved.
        for model_file in sorted(ensemble_dir.glob("*.model")):
            ensemble.append(cuml.common.load(model_file))
        if ensemble:
            components['umap_reducer_ensemble'] = ensemble
            
    return components


def _load_canonical_map(directory: Path) -> Optional[cudf.DataFrame]:
    """Loads the canonical map DataFrame from its Parquet file."""
    parquet_path = directory / 'canonical_map.parquet'
    if parquet_path.exists():
        return cudf.read_parquet(parquet_path)
    
    logger.warning(f"Canonical map file not found at {parquet_path}. Returning None.")
    return None
