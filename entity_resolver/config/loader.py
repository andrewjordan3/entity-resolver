# entity_resolver/config/loader.py
"""
Loads configuration from a YAML file into a validated Pydantic model.
"""
import yaml
from pathlib import Path
from pydantic import ValidationError

# Import the main Pydantic model from your schema
from .schema import ResolverConfig  

def load_config(config_path: str | Path | None = None) -> ResolverConfig:
    """
    Loads the resolver configuration from a YAML file.

    It starts with the default configuration defined in the Pydantic schema,
    overrides it with any values from the external YAML file, and then
    validates the final configuration.

    Args:
        config_path: Path to the user-defined YAML configuration file.
                     If None, only the default configuration is returned.

    Returns:
        The fully validated ResolverConfig instance.

    Raises:
        FileNotFoundError: If the provided config_path does not exist.
        ValidationError: If the final configuration fails Pydantic validation.
                         The error message will contain detailed information
                         about what is wrong.
    """
    # If no path is provided, simply return an instance of the config
    # with all the default values from the schema.
    if not config_path:
        return ResolverConfig()

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    # Load the user's YAML file into a dictionary.
    with open(config_path, 'r') as f:
        user_config_data = yaml.safe_load(f) or {}

    try:
        # Let Pydantic create, validate, and merge the configuration.
        # Pydantic's `model_validate` takes a dict, merges it with the defaults
        # defined in the model, and runs all validators.
        # If validation fails, it raises a very informative ValidationError.
        return ResolverConfig.model_validate(user_config_data)
    except ValidationError as e:
        # Re-raise the Pydantic error directly. It's very descriptive and
        # will tell the user exactly which parameter is wrong and why.
        print(f"--- Configuration Error ---\n{e}\n---------------------------")
        raise
