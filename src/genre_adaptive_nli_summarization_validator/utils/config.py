"""Configuration management utilities."""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

from ..exceptions import ConfigurationError


class Config:
    """Configuration manager for the genre-adaptive NLI summarization validator.

    Provides centralized configuration management with support for YAML files,
    environment variables, and programmatic overrides.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.

        Args:
            config_path: Path to configuration YAML file. If None, uses default.
        """
        self._config: Dict[str, Any] = {}
        self._setup_logging()

        if config_path is None:
            config_path = Path(__file__).parent.parent.parent.parent / "configs" / "default.yaml"

        self.load_config(config_path)

    def _setup_logging(self) -> None:
        """Setup basic logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file.

        Raises:
            ConfigurationError: If configuration file doesn't exist, cannot be read,
                or contains invalid YAML syntax.
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigurationError(
                "Configuration file not found",
                details={"config_path": str(config_path)}
            )

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
            logging.info(f"Loaded configuration from {config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(
                "Failed to parse YAML configuration",
                details={"config_path": str(config_path)},
                original_exception=e
            )
        except (IOError, OSError) as e:
            raise ConfigurationError(
                "Failed to read configuration file",
                details={"config_path": str(config_path)},
                original_exception=e
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Supports nested keys using dot notation (e.g., 'model.name').
        Environment variables override config file values.

        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        # Check environment variables first
        env_key = key.upper().replace(".", "_")
        env_value = os.getenv(env_key)
        if env_value is not None:
            return self._parse_env_value(env_value)

        # Navigate nested dictionary
        value = self._config
        for k in key.split("."):
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def _parse_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Parse environment variable value to appropriate type.

        Args:
            value: Raw environment variable value.

        Returns:
            Parsed value with appropriate type.
        """
        # Boolean values
        if value.lower() in ("true", "yes", "1"):
            return True
        elif value.lower() in ("false", "no", "0"):
            return False

        # Numeric values
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # String value
        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value.

        Args:
            key: Configuration key (supports dot notation).
            value: Value to set.
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with dictionary of values.

        Args:
            updates: Dictionary of configuration updates.
        """
        for key, value in updates.items():
            self.set(key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary.

        Returns:
            Complete configuration dictionary.
        """
        return self._config.copy()

    def create_output_dir(self, base_dir: Optional[str] = None) -> Path:
        """Create and return output directory for experiments.

        Args:
            base_dir: Base directory for outputs. Uses config if None.

        Returns:
            Path to created output directory.
        """
        if base_dir is None:
            base_dir = self.get("training.output_dir", "./outputs")

        output_dir = Path(base_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        return output_dir

    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_config = self.get("logging", {})

        # Create logs directory if specified
        log_file = log_config.get("file")
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "INFO")),
            format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            filename=log_file,
            filemode="a" if log_file else None,
            force=True
        )

    def get_device_config(self) -> Dict[str, Any]:
        """Get device configuration for PyTorch.

        Returns:
            Dictionary with device configuration.
        """
        import torch

        device_config = self.get("device", {})
        use_cuda = device_config.get("use_cuda", True)

        if use_cuda and torch.cuda.is_available():
            device = "cuda"
            device_count = torch.cuda.device_count()
        else:
            device = "cpu"
            device_count = 1

        return {
            "device": device,
            "device_count": device_count,
            "mixed_precision": device_config.get("mixed_precision", True) and device == "cuda",
            "compile_model": device_config.get("compile_model", False),
        }