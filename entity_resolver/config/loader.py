# entity_resolver/config/loader.py
"""
Loads and saves the configuration from/to a YAML file using a validated schema.
This module separates I/O operations from validation logic for flexibility.
"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import ValidationError

from .schema import ResolverConfig

# Set up a dedicated logger for this module.
logger = logging.getLogger(__name__)


def _validate_and_create_config(data: Dict[str, Any]) -> ResolverConfig:
    """
    Internal helper to validate a dictionary against the ResolverConfig schema.

    Args:
        data: A dictionary containing configuration data.

    Returns:
        A validated ResolverConfig instance.

    Raises:
        ValidationError: If the data fails Pydantic validation.
    """
    try:
        # Pydantic's model_validate handles merging user data with defaults.
        return ResolverConfig.model_validate(data)
    except ValidationError as e:
        # Log the critical error and re-raise to halt execution.
        logger.critical(f"Configuration validation failed:\n{e}")
        raise


def load_raw_config(config_path: Path | str) -> Dict[str, Any]:
    """
    Loads raw, unvalidated configuration from a YAML file into a dictionary.

    This function is intended for testing or inspection purposes where you need
    to access the configuration data without triggering validation.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary with the raw data from the YAML file.

    Raises:
        FileNotFoundError: If the provided config_path does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logger.error(f"Configuration file not found at: {config_path}")
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    logger.info(f"Loading raw, unvalidated configuration from: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: Optional[Path | str] = None) -> ResolverConfig:
    """
    Loads, validates, and merges configuration from a YAML file.

    This function always returns a fully validated ResolverConfig object. If no
    path is provided, it returns the default configuration. If the file data
    is invalid, it raises a ValidationError.

    Args:
        config_path: Optional path to the YAML configuration file.

    Returns:
        A validated ResolverConfig instance.

    Raises:
        FileNotFoundError: If the provided config_path does not exist.
        ValidationError: If the configuration data is invalid.
    """
    if not config_path:
        user_config_data = {}
        logger.info("No configuration path provided. Using default settings.")
    else:
        # For the main loading function, we can reuse the raw loader.
        user_config_data = load_raw_config(config_path)

    # Always validate the loaded data and return a ResolverConfig object.
    return _validate_and_create_config(user_config_data)


def save_config(config: ResolverConfig, path: Path | str) -> None:
    """
    Saves a ResolverConfig instance to a YAML file.

    Args:
        config: The validated ResolverConfig object to save.
        path: The output path for the YAML file.

    Raises:
        TypeError: If the provided config is not a ResolverConfig instance.
    """
    if not isinstance(config, ResolverConfig):
        raise TypeError("Input must be a valid ResolverConfig instance to save.")

    output_path = Path(path)
    # Ensure the parent directory exists before writing the file.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use model_dump to get a serializable dictionary. We exclude the internal
    # `_log_level_int` field as it's a derived value not meant for user config.
    config_dict = config.model_dump(exclude={'_log_level_int'})

    logger.info(f"Saving configuration to: {output_path}")
    with open(output_path, 'w') as f:
        yaml.safe_dump(
            config_dict,
            f,
            default_flow_style=False,
            sort_keys=False,
            indent=2
        )