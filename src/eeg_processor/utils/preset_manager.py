"""
Preset management system for EEG Processor.

This module handles loading, validation, and management of configuration presets.
Presets provide pre-configured processing pipelines for common EEG analysis scenarios.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from loguru import logger
import copy

from .exceptions import ValidationError, ConfigurationError


@dataclass
class PresetInfo:
    """Information about a configuration preset."""
    name: str
    description: str
    category: str
    version: str
    author: Optional[str] = None
    created: Optional[str] = None
    tags: List[str] = None
    use_cases: List[str] = None
    data_formats: List[str] = None
    recommended_channels: Optional[str] = None
    file_path: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.use_cases is None:
            self.use_cases = []
        if self.data_formats is None:
            self.data_formats = []


class PresetManager:
    """Manages configuration presets for EEG processing pipelines."""
    
    def __init__(self):
        self.presets_dir = Path(__file__).parent.parent / "presets"
        self.schema_path = self.presets_dir / "schema.yml"
        self._preset_cache = {}
        self._load_schema()
    
    def _load_schema(self):
        """Load and parse the preset schema definition."""
        try:
            with open(self.schema_path, 'r') as f:
                self.schema = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Preset schema not found at {self.schema_path}")
            self.schema = {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing preset schema: {e}")
            self.schema = {}
    
    def get_available_presets(self, category: Optional[str] = None) -> Dict[str, List[PresetInfo]]:
        """Get all available presets, optionally filtered by category.
        
        Args:
            category: Optional category filter ('basic', 'advanced', 'research')
            
        Returns:
            Dictionary mapping categories to lists of PresetInfo objects
        """
        presets_by_category = {}
        
        # Scan preset directories
        for category_dir in self.presets_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith('.'):
                continue
                
            cat_name = category_dir.name
            if category and cat_name != category:
                continue
                
            presets_by_category[cat_name] = []
            
            # Find all YAML files in category directory
            for preset_file in category_dir.glob("*.yml"):
                if preset_file.name == "schema.yml":
                    continue
                    
                try:
                    preset_info = self._load_preset_info(preset_file)
                    presets_by_category[cat_name].append(preset_info)
                except Exception as e:
                    logger.warning(f"Failed to load preset {preset_file}: {e}")
        
        return presets_by_category
    
    def _load_preset_info(self, preset_path: Path) -> PresetInfo:
        """Load preset metadata without full configuration."""
        try:
            with open(preset_path, 'r') as f:
                preset_data = yaml.safe_load(f)
            
            metadata = preset_data.get('metadata', {})
            
            return PresetInfo(
                name=metadata.get('name', preset_path.stem),
                description=metadata.get('description', 'No description available'),
                category=metadata.get('category', preset_path.parent.name),
                version=metadata.get('version', '1.0.0'),
                author=metadata.get('author'),
                created=metadata.get('created'),
                tags=metadata.get('tags', []),
                use_cases=metadata.get('use_cases', []),
                data_formats=metadata.get('data_formats', []),
                recommended_channels=metadata.get('recommended_channels'),
                file_path=str(preset_path)
            )
            
        except Exception as e:
            raise ValidationError(f"Failed to load preset metadata from {preset_path}: {e}")
    
    def load_preset(self, preset_name: str, category: Optional[str] = None) -> Dict[str, Any]:
        """Load a complete preset configuration.
        
        Args:
            preset_name: Name of the preset (without .yml extension)
            category: Optional category to search in
            
        Returns:
            Complete preset configuration dictionary
            
        Raises:
            ConfigurationError: If preset not found or invalid
        """
        # Check cache first
        cache_key = f"{category or 'all'}:{preset_name}"
        if cache_key in self._preset_cache:
            return copy.deepcopy(self._preset_cache[cache_key])
        
        # Find preset file
        preset_path = self._find_preset_file(preset_name, category)
        if not preset_path:
            available = self._get_available_preset_names()
            raise ConfigurationError(
                f"Preset '{preset_name}' not found. Available presets: {', '.join(available)}"
            )
        
        # Load and validate preset
        try:
            with open(preset_path, 'r') as f:
                preset_data = yaml.safe_load(f)
            
            # Basic validation
            self._validate_preset(preset_data, preset_path)
            
            # Cache the result
            self._preset_cache[cache_key] = preset_data
            
            return copy.deepcopy(preset_data)
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing preset {preset_name}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading preset {preset_name}: {e}")
    
    def _find_preset_file(self, preset_name: str, category: Optional[str] = None) -> Optional[Path]:
        """Find preset file by name and optional category."""
        preset_filename = f"{preset_name}.yml"
        
        if category:
            # Search in specific category
            category_path = self.presets_dir / category / preset_filename
            if category_path.exists():
                return category_path
        else:
            # Search in all categories
            for category_dir in self.presets_dir.iterdir():
                if not category_dir.is_dir() or category_dir.name.startswith('.'):
                    continue
                    
                preset_path = category_dir / preset_filename
                if preset_path.exists():
                    return preset_path
        
        return None
    
    def _get_available_preset_names(self) -> List[str]:
        """Get list of all available preset names."""
        names = []
        for category_dir in self.presets_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith('.'):
                continue
            for preset_file in category_dir.glob("*.yml"):
                if preset_file.name != "schema.yml":
                    names.append(preset_file.stem)
        return sorted(names)
    
    def _validate_preset(self, preset_data: Dict[str, Any], preset_path: Path):
        """Validate preset against schema and common rules."""
        # Check required sections
        required_sections = ['metadata', 'config_template']
        for section in required_sections:
            if section not in preset_data:
                raise ValidationError(f"Preset {preset_path.name} missing required section: {section}")
        
        # Validate metadata
        metadata = preset_data['metadata']
        required_metadata = ['name', 'description', 'category', 'version']
        for field in required_metadata:
            if not metadata.get(field):
                raise ValidationError(f"Preset {preset_path.name} missing required metadata: {field}")
        
        # Validate category
        valid_categories = ['basic', 'advanced', 'research']
        if metadata['category'] not in valid_categories:
            raise ValidationError(
                f"Invalid category '{metadata['category']}' in {preset_path.name}. "
                f"Valid categories: {', '.join(valid_categories)}"
            )
        
        # Validate version format
        version = metadata['version']
        version_parts = version.split('.')
        if len(version_parts) != 3 or not all(part.isdigit() for part in version_parts):
            raise ValidationError(f"Invalid version format '{version}' in {preset_path.name}. Use 'X.Y.Z' format.")
        
        # Validate config template
        config_template = preset_data['config_template']
        if 'stages' not in config_template:
            raise ValidationError(f"Preset {preset_path.name} missing 'stages' in config_template")
        
        if not isinstance(config_template['stages'], list) or len(config_template['stages']) == 0:
            raise ValidationError(f"Preset {preset_path.name} must have at least one processing stage")
        
        # Validate stage names
        valid_stages = self.schema.get('validation_rules', {}).get('valid_stages', [])
        if valid_stages:
            for stage_config in config_template['stages']:
                if isinstance(stage_config, dict):
                    stage_name = list(stage_config.keys())[0]
                elif isinstance(stage_config, str):
                    stage_name = stage_config
                else:
                    raise ValidationError(f"Invalid stage format in {preset_path.name}: {stage_config}")
                
                if stage_name not in valid_stages:
                    raise ValidationError(f"Unknown stage '{stage_name}' in {preset_path.name}")
    
    def create_config_from_preset(self, preset_name: str, 
                                 user_overrides: Optional[Dict[str, Any]] = None,
                                 category: Optional[str] = None) -> Dict[str, Any]:
        """Create a complete configuration by merging preset with user overrides.
        
        Args:
            preset_name: Name of the preset to use
            user_overrides: User-provided configuration overrides
            category: Optional category to search in
            
        Returns:
            Complete configuration dictionary ready for processing
        """
        # Load base preset
        preset = self.load_preset(preset_name, category)
        config_template = preset['config_template']
        
        # Start with preset configuration
        final_config = copy.deepcopy(config_template)
        
        # Apply user overrides if provided
        if user_overrides:
            final_config = self._merge_configurations(final_config, user_overrides)
        
        # Validate final configuration has required fields
        self._validate_final_config(final_config, preset_name)
        
        return final_config
    
    def _merge_configurations(self, base_config: Dict[str, Any], 
                            overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user overrides with base configuration."""
        merged = copy.deepcopy(base_config)
        
        for key, value in overrides.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursive merge for nested dictionaries
                merged[key] = self._merge_configurations(merged[key], value)
            else:
                # Direct override
                merged[key] = copy.deepcopy(value)
        
        return merged
    
    def _validate_final_config(self, config: Dict[str, Any], preset_name: str):
        """Validate that final configuration has all required fields."""
        required_fields = ['stages']
        
        for field in required_fields:
            if field not in config:
                raise ConfigurationError(
                    f"Configuration from preset '{preset_name}' missing required field: {field}"
                )
        
        # Validate paths if provided
        if 'paths' in config:
            paths = config['paths']
            path_fields = ['raw_data_dir', 'results_dir', 'file_extension']
            missing_paths = [field for field in path_fields if not paths.get(field)]
            if missing_paths:
                logger.warning(
                    f"Preset '{preset_name}' missing path configuration: {', '.join(missing_paths)}. "
                    "These must be provided by user."
                )
    
    def list_presets_by_tag(self, tag: str) -> List[PresetInfo]:
        """Find presets that match a specific tag."""
        matching_presets = []
        all_presets = self.get_available_presets()
        
        for category_presets in all_presets.values():
            for preset_info in category_presets:
                if tag.lower() in [t.lower() for t in preset_info.tags]:
                    matching_presets.append(preset_info)
        
        return matching_presets
    
    def get_preset_info(self, preset_name: str, category: Optional[str] = None) -> PresetInfo:
        """Get detailed information about a specific preset."""
        preset_path = self._find_preset_file(preset_name, category)
        if not preset_path:
            raise ConfigurationError(f"Preset '{preset_name}' not found")
        
        return self._load_preset_info(preset_path)
    
    def validate_preset_file(self, preset_path: Path) -> List[str]:
        """Validate a preset file and return list of issues found."""
        issues = []
        
        try:
            with open(preset_path, 'r') as f:
                preset_data = yaml.safe_load(f)
            
            try:
                self._validate_preset(preset_data, preset_path)
            except ValidationError as e:
                issues.append(str(e))
                
        except yaml.YAMLError as e:
            issues.append(f"YAML parsing error: {e}")
        except Exception as e:
            issues.append(f"Error loading file: {e}")
        
        return issues