"""Configuration management and utility functions.

This module provides configuration management utilities and helper functions
for the genre-adaptive NLI summarization validator, including YAML-based
configuration with environment variable override support.

Key Features:
    - Hierarchical YAML configuration loading
    - Environment variable override capabilities
    - Dot notation access for nested configurations
    - Automatic type parsing and validation
    - Device configuration management

Classes:
    Config: Central configuration manager with YAML loading and environment
        variable override support.

Example:
    >>> from genre_adaptive_nli_summarization_validator.utils import Config
    >>> config = Config('configs/default.yaml')
    >>> batch_size = config.get('training.batch_size', 8)
"""

from .config import Config

__all__ = ["Config"]