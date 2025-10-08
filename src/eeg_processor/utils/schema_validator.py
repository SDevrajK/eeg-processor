"""
JSON Schema validation utilities for EEG Processor configurations.

This module provides utilities for validating configuration files against
the generated JSON schemas.
"""

import json
import jsonschema
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import yaml

from .exceptions import ValidationError


class SchemaValidator:
    """Validator for EEG Processor configurations using JSON schemas."""
    
    def __init__(self, schema_dir: str = "schemas"):
        """Initialize validator with schema directory.
        
        Args:
            schema_dir: Directory containing JSON schema files
        """
        self.schema_dir = Path(schema_dir)
        self._schemas = {}
        self._load_schemas()
    
    def _load_schemas(self):
        """Load all JSON schemas from the schema directory."""
        if not self.schema_dir.exists():
            return
        
        for schema_file in self.schema_dir.glob("*.json"):
            schema_name = schema_file.stem
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    self._schemas[schema_name] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load schema {schema_file}: {e}")
    
    def validate_config_file(self, config_path: str) -> Tuple[bool, List[str]]:
        """Validate a configuration file against the main schema.
        
        Args:
            config_path: Path to the configuration file (YAML or JSON)
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            # Load configuration
            config_file = Path(config_path)
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yml', '.yaml']:
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            return self.validate_config(config)
            
        except Exception as e:
            return False, [f"Error loading configuration file: {e}"]
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a configuration dictionary against the main schema.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if 'eeg_processor_config' not in self._schemas:
            return True, ["Main configuration schema not available"]
        
        try:
            schema = self._schemas['eeg_processor_config']
            jsonschema.validate(config, schema)
            return True, []
            
        except jsonschema.ValidationError as e:
            return False, [self._format_validation_error(e)]
        except Exception as e:
            return False, [f"Validation error: {e}"]
    
    def validate_stage(self, stage_name: str, stage_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a stage configuration against its schema.
        
        Args:
            stage_name: Name of the processing stage
            stage_config: Stage configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        schema_name = f"stage_{stage_name}"
        
        if schema_name not in self._schemas:
            return True, [f"Schema for stage '{stage_name}' not available"]
        
        try:
            schema = self._schemas[schema_name]
            jsonschema.validate(stage_config, schema)
            return True, []
            
        except jsonschema.ValidationError as e:
            return False, [self._format_validation_error(e)]
        except Exception as e:
            return False, [f"Stage validation error: {e}"]
    
    def validate_preset(self, preset_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a preset configuration against the preset schema.
        
        Args:
            preset_config: Preset configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if 'preset_schema' not in self._schemas:
            return True, ["Preset schema not available"]
        
        try:
            schema = self._schemas['preset_schema']
            jsonschema.validate(preset_config, schema)
            return True, []
            
        except jsonschema.ValidationError as e:
            return False, [self._format_validation_error(e)]
        except Exception as e:
            return False, [f"Preset validation error: {e}"]
    
    def _format_validation_error(self, error: jsonschema.ValidationError) -> str:
        """Format a validation error into a human-readable message."""
        path = " -> ".join(str(p) for p in error.absolute_path) if error.absolute_path else "root"
        return f"At {path}: {error.message}"
    
    def get_available_schemas(self) -> List[str]:
        """Get list of available schema names."""
        return list(self._schemas.keys())
    
    def has_schema(self, schema_name: str) -> bool:
        """Check if a specific schema is available."""
        return schema_name in self._schemas


def validate_config_with_schema(config_path: str, schema_dir: str = "schemas") -> Tuple[bool, List[str]]:
    """Convenience function to validate a configuration file.
    
    Args:
        config_path: Path to configuration file
        schema_dir: Directory containing schema files
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    validator = SchemaValidator(schema_dir)
    return validator.validate_config_file(config_path)


def validate_stage_config(stage_name: str, stage_config: Dict[str, Any], 
                         schema_dir: str = "schemas") -> Tuple[bool, List[str]]:
    """Convenience function to validate a stage configuration.
    
    Args:
        stage_name: Name of the processing stage
        stage_config: Stage configuration dictionary
        schema_dir: Directory containing schema files
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    validator = SchemaValidator(schema_dir)
    return validator.validate_stage(stage_name, stage_config)