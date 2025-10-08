"""
JSON Schema generator for EEG Processor stage parameters and validation.

This module creates comprehensive JSON schemas for all processing stages,
enabling programmatic validation, IDE support, and API documentation.
"""

import json
import inspect
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

from .stage_documentation import StageDocumentationExtractor, StageInfo


class JSONSchemaGenerator:
    """Generator for JSON schemas of EEG processing stages and configurations."""
    
    def __init__(self):
        self.extractor = StageDocumentationExtractor()
        
    def generate_complete_schema(self, output_dir: str = "schemas") -> Dict[str, Any]:
        """Generate complete JSON schema for EEG Processor configurations.
        
        Args:
            output_dir: Directory to save schema files
            
        Returns:
            Dictionary containing all generated schemas
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        schemas = {}
        
        # Generate main configuration schema
        main_schema = self._generate_main_config_schema()
        main_file = output_path / "eeg_processor_config.json"
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(main_schema, f, indent=2)
        schemas["main_config"] = main_schema
        
        # Generate individual stage schemas
        stage_schemas = self._generate_all_stage_schemas()
        for stage_name, schema in stage_schemas.items():
            stage_file = output_path / f"stage_{stage_name}.json"
            with open(stage_file, 'w', encoding='utf-8') as f:
                json.dump(schema, f, indent=2)
        schemas.update(stage_schemas)
        
        # Generate combined stages schema
        stages_schema = self._generate_stages_schema(stage_schemas)
        stages_file = output_path / "stages_schema.json"
        with open(stages_file, 'w', encoding='utf-8') as f:
            json.dump(stages_schema, f, indent=2)
        schemas["stages"] = stages_schema
        
        # Generate preset schema (if not already exists)
        preset_schema = self._generate_preset_schema()
        preset_file = output_path / "preset_schema.json"
        with open(preset_file, 'w', encoding='utf-8') as f:
            json.dump(preset_schema, f, indent=2)
        schemas["preset"] = preset_schema
        
        return schemas
    
    def _generate_main_config_schema(self) -> Dict[str, Any]:
        """Generate the main configuration file schema.
        
        Supports both legacy and new configuration formats:
        - Legacy: top-level raw_data_dir, results_dir, participants, stages, conditions
        - New: study, paths, participants, processing, conditions structure
        """
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": "https://eeg-processor.org/schemas/config.json",
            "title": "EEG Processor Configuration",
            "description": "Configuration schema for EEG Processor pipeline (supports legacy and new formats)",
            "type": "object",
            "oneOf": [
                {
                    "title": "New Configuration Format",
                    "required": ["study", "participants", "paths", "processing"],
                    "properties": {
                        "study": {
                            "type": "object",
                            "required": ["name"],
                            "properties": {
                                "name": {"type": "string"},
                                "dataset": {"type": "string"},
                                "description": {"type": "string"},
                                "researcher": {"type": "string"}
                            },
                            "additionalProperties": True
                        },
                        "paths": {
                            "type": "object",
                            "required": ["raw_data", "results"],
                            "properties": {
                                "raw_data": {"type": "string"},
                                "results": {"type": "string"},
                                "file_extension": {"type": "string", "default": ".vhdr"}
                            },
                            "additionalProperties": True
                        },
                        "participants": {
                            "type": "object",
                            "patternProperties": {
                                "^[a-zA-Z0-9_-]+$": {
                                    "oneOf": [
                                        {"type": "string"},
                                        {
                                            "type": "object",
                                            "required": ["file"],
                                            "properties": {
                                                "file": {"type": "string"},
                                                "age": {"type": "integer"},
                                                "gender": {"type": "string"},
                                                "group": {"type": "string"}
                                            },
                                            "additionalProperties": True
                                        }
                                    ]
                                }
                            }
                        },
                        "processing": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "patternProperties": {
                                    "^[a-zA-Z0-9_]+$": {
                                        "type": "object"
                                    }
                                },
                                "additionalProperties": False
                            }
                        },
                        "conditions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["name"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "triggers": {
                                        "type": "object",
                                        "description": "Trigger definitions for condition"
                                    },
                                    "markers": {
                                        "oneOf": [
                                            {"type": "null"},
                                            {
                                                "type": "array",
                                                "items": {
                                                    "oneOf": [
                                                        {"type": "string"},
                                                        {"type": "integer"}
                                                    ]
                                                }
                                            },
                                            {
                                                "type": "object",
                                                "patternProperties": {
                                                    "^[a-zA-Z0-9_]+$": {
                                                        "type": "array",
                                                        "items": {
                                                            "oneOf": [
                                                                {"type": "string"},
                                                                {"type": "integer"}
                                                            ]
                                                        }
                                                    }
                                                }
                                            }
                                        ]
                                    }
                                },
                                "additionalProperties": True
                            }
                        },
                        "output": {
                            "type": "object",
                            "description": "Output configuration settings"
                        }
                    },
                    "additionalProperties": True
                },
                {
                    "title": "Legacy Configuration Format",
                    "required": ["raw_data_dir", "results_dir", "participants", "stages"],
                    "properties": {
                        "raw_data_dir": {"type": "string"},
                        "results_dir": {"type": "string"},
                        "file_extension": {"type": "string", "default": ".vhdr"},
                        "participants": {
                            "oneOf": [
                                {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                {
                                    "type": "object",
                                    "patternProperties": {
                                        "^[a-zA-Z0-9_-]+$": {
                                            "oneOf": [
                                                {"type": "string"},
                                                {
                                                    "type": "object",
                                                    "required": ["file"],
                                                    "properties": {
                                                        "file": {"type": "string"},
                                                        "conditions": {
                                                            "type": "array",
                                                            "items": {"type": "string"}
                                                        }
                                                    },
                                                    "additionalProperties": True
                                                }
                                            ]
                                        }
                                    }
                                }
                            ]
                        },
                        "stages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "patternProperties": {
                                    "^[a-zA-Z0-9_]+$": {
                                        "type": "object"
                                    }
                                },
                                "additionalProperties": False
                            }
                        },
                        "conditions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["name", "condition_markers"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "condition_markers": {
                                        "type": "array",
                                        "items": {
                                            "oneOf": [
                                                {"type": "string"},
                                                {"type": "integer"}
                                            ]
                                        }
                                    },
                                    "description": {"type": "string"}
                                },
                                "additionalProperties": True
                            }
                        },
                        "study_info": {"type": "object"},
                        "output_config": {"type": "object"},
                        "dataset_name": {"type": "string"}
                    },
                    "additionalProperties": True
                }
            ]
        }
    
    def _generate_all_stage_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Generate schemas for all individual stages."""
        stage_schemas = {}
        categorized_stages = self.extractor.get_all_stages()
        
        for category, stages in categorized_stages.items():
            for stage_name in stages:
                stage_info = self.extractor.get_stage_info(stage_name)
                if stage_info:
                    schema = self._generate_stage_schema(stage_info)
                    stage_schemas[stage_name] = schema
        
        return stage_schemas
    
    def _generate_stage_schema(self, stage_info: StageInfo) -> Dict[str, Any]:
        """Generate JSON schema for a single stage."""
        properties = {}
        required = []
        
        for param_name, param_info in stage_info.parameters.items():
            # Skip internal parameters
            if param_name in ['raw', 'data', 'epochs', 'self']:
                continue
            
            # Skip **kwargs parameters (they're not user-configurable)
            if param_name.endswith('_kwargs') or param_name == 'kwargs':
                continue
            
            # Skip condition parameter (provided by configuration system)
            if param_name == 'condition':
                continue
            
            param_schema = self._convert_param_to_schema(param_info)
            properties[param_name] = param_schema
            
            if param_info.get('required', False):
                required.append(param_name)
        
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": f"https://eeg-processor.org/schemas/stage_{stage_info.name}.json",
            "title": f"EEG Processor - {stage_info.name} Stage",
            "description": stage_info.description,
            "type": "object",
            "properties": properties
        }
        
        if required:
            schema["required"] = required
        
        # Add examples if available
        examples = self._get_stage_examples(stage_info.name)
        if examples:
            schema["examples"] = examples
        
        schema["additionalProperties"] = False
        
        return schema
    
    def _convert_param_to_schema(self, param_info: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parameter information to JSON schema property."""
        param_type = param_info.get('type', 'Any')
        default_value = param_info.get('default', 'Required')
        
        schema = {}
        
        # Handle type conversion
        if param_type == 'bool':
            schema["type"] = "boolean"
        elif param_type == 'int':
            schema["type"] = "integer"
        elif param_type == 'float':
            schema["type"] = "number"
        elif param_type == 'str':
            schema["type"] = "string"
        elif param_type == 'list':
            schema["type"] = "array"
        elif param_type == 'dict':
            schema["type"] = "object"
        elif '|' in param_type:
            # Handle union types like "float | None", "str | int"
            types = [t.strip() for t in param_type.split('|')]
            json_types = []
            
            for t in types:
                if t == 'None':
                    json_types.append("null")
                elif t == 'bool':
                    json_types.append("boolean")
                elif t == 'int':
                    json_types.append("integer")
                elif t == 'float':
                    json_types.append("number")
                elif t == 'str':
                    json_types.append("string")
                elif t == 'list':
                    json_types.append("array")
                elif t == 'dict':
                    json_types.append("object")
            
            if len(json_types) == 1:
                schema["type"] = json_types[0]
            else:
                schema["type"] = json_types
        else:
            # Default to allowing any type
            schema["type"] = ["string", "number", "boolean", "array", "object", "null"]
        
        # Add default value if not required
        if default_value != "Required" and default_value != "None":
            try:
                # Try to parse the default value
                if default_value == "True":
                    schema["default"] = True
                elif default_value == "False":
                    schema["default"] = False
                elif default_value.startswith('"') and default_value.endswith('"'):
                    schema["default"] = default_value[1:-1]  # Remove quotes
                elif default_value.replace('.', '').replace('-', '').isdigit():
                    if '.' in default_value:
                        schema["default"] = float(default_value)
                    else:
                        schema["default"] = int(default_value)
                else:
                    schema["default"] = default_value
            except (ValueError, AttributeError):
                # If parsing fails, store as string
                schema["default"] = str(default_value)
        
        # Add parameter-specific constraints
        constraints = self._get_parameter_constraints(param_info['name'])
        if constraints:
            schema.update(constraints)
        
        return schema
    
    def _get_parameter_constraints(self, param_name: str) -> Dict[str, Any]:
        """Get parameter-specific constraints for validation."""
        constraints = {}
        
        # Common parameter constraints
        if param_name in ['l_freq', 'h_freq']:
            constraints.update({
                "type": "number",
                "minimum": 0.001,
                "maximum": 1000,
                "description": "Frequency in Hz"
            })
        elif param_name == 'notch':
            constraints.update({
                "oneOf": [
                    {"type": "null"},
                    {
                        "type": "number",
                        "minimum": 30,
                        "maximum": 100,
                        "description": "Notch frequency in Hz"
                    },
                    {
                        "type": "array",
                        "items": {
                            "type": "number",
                            "minimum": 30,
                            "maximum": 100
                        },
                        "description": "Multiple notch frequencies"
                    }
                ]
            })
        elif param_name in ['tmin', 'tmax']:
            constraints.update({
                "type": "number",
                "description": "Time in seconds"
            })
            if param_name == 'tmin':
                constraints.update({"minimum": -10.0, "maximum": 5.0})
            else:  # tmax
                constraints.update({"minimum": 0.1, "maximum": 60.0})
        elif param_name == 'threshold':
            constraints.update({
                "type": "number",
                "minimum": 0.1,
                "maximum": 10.0,
                "description": "Detection threshold"
            })
        elif param_name == 'n_neighbors':
            constraints.update({
                "type": "integer",
                "minimum": 3,
                "maximum": 50,
                "description": "Number of neighboring channels"
            })
        elif param_name == 'cutoff':
            constraints.update({
                "type": "number",
                "minimum": 5,
                "maximum": 100,
                "description": "ASR cutoff parameter"
            })
        elif param_name == 'method':
            # This depends on context, but we can provide common values
            constraints.update({
                "type": "string",
                "enum": ["average", "median", "ica", "euclid", "riemann", "eog_regression", "gratton_coles"]
            })
        
        return constraints
    
    def _get_stage_examples(self, stage_name: str) -> List[Dict[str, Any]]:
        """Get example configurations for a stage."""
        examples = {
            "filter": [
                {"l_freq": 0.1, "h_freq": 40},
                {"l_freq": 1.0, "h_freq": 30, "notch": 50},
                {"h_freq": 40, "notch": [50, 100]}
            ],
            "detect_bad_channels": [
                {"threshold": 1.5, "n_neighbors": 8},
                {"threshold": 2.0, "interpolate": True}
            ],
            "rereference": [
                {"method": "average"},
                {"method": "median"}
            ],
            "epoch": [
                {"tmin": -0.2, "tmax": 0.8, "baseline": [-0.2, 0]},
                {"tmin": -0.5, "tmax": 2.0, "baseline": None}
            ],
            "clean_rawdata_asr": [
                {"cutoff": 20, "method": "euclid"},
                {"cutoff": 15, "method": "riemann"}
            ],
            "remove_blinks_emcp": [
                {"method": "eog_regression", "eog_channels": ["HEOG", "VEOG"]},
                {"method": "gratton_coles", "eog_channels": ["HEOG"], "subtract_evoked": True}
            ]
        }
        
        return examples.get(stage_name, [])
    
    def _generate_stages_schema(self, stage_schemas: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate schema for the stages array configuration."""
        stage_definitions = {}
        
        for stage_name, schema in stage_schemas.items():
            stage_definitions[f"{stage_name}_stage"] = {
                "type": "object",
                "properties": {
                    stage_name: {
                        "oneOf": [
                            {"type": "null"},
                            schema
                        ]
                    }
                },
                "additionalProperties": False,
                "minProperties": 1,
                "maxProperties": 1
            }
        
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": "https://eeg-processor.org/schemas/stages.json",
            "title": "EEG Processor Stages Schema",
            "description": "Schema for processing stages configuration",
            "definitions": {
                "stage_item": {
                    "oneOf": [
                        {
                            "type": "string",
                            "enum": list(stage_schemas.keys()),
                            "description": "Stage name (uses default parameters)"
                        },
                        {
                            "type": "object",
                            "oneOf": [stage_definitions[f"{name}_stage"] for name in stage_schemas.keys()],
                            "description": "Stage with custom parameters"
                        }
                    ]
                }
            }
        }
    
    def _generate_preset_schema(self) -> Dict[str, Any]:
        """Generate schema for preset configuration files."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": "https://eeg-processor.org/schemas/preset.json",
            "title": "EEG Processor Preset Schema",
            "description": "Schema for EEG Processor preset configuration files",
            "type": "object",
            "required": ["metadata", "config_template"],
            "properties": {
                "metadata": {
                    "type": "object",
                    "required": ["name", "version", "description", "category"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "pattern": "^[a-z0-9_-]+$",
                            "description": "Preset identifier (lowercase, alphanumeric, hyphens, underscores)"
                        },
                        "version": {
                            "type": "string",
                            "pattern": "^\\d+\\.\\d+\\.\\d+$",
                            "description": "Semantic version (e.g., 1.0.0)"
                        },
                        "description": {
                            "type": "string",
                            "minLength": 10,
                            "description": "Human-readable description of the preset"
                        },
                        "category": {
                            "type": "string",
                            "enum": ["basic", "advanced", "research"],
                            "description": "Preset category"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorizing and searching presets"
                        },
                        "use_cases": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Typical use cases for this preset"
                        },
                        "author": {
                            "type": "string",
                            "description": "Preset author"
                        },
                        "created_date": {
                            "type": "string",
                            "format": "date",
                            "description": "Creation date (YYYY-MM-DD)"
                        },
                        "recommended_channels": {
                            "type": "string",
                            "description": "Recommended number/type of channels"
                        }
                    },
                    "additionalProperties": False
                },
                "config_template": {
                    "$ref": "eeg_processor_config.json",
                    "description": "The actual configuration template"
                }
            },
            "additionalProperties": False
        }


def generate_json_schemas(output_dir: str = "schemas") -> Dict[str, Any]:
    """Generate all JSON schemas for EEG Processor.
    
    Args:
        output_dir: Directory to save schema files
        
    Returns:
        Dictionary containing all generated schemas
    """
    generator = JSONSchemaGenerator()
    return generator.generate_complete_schema(output_dir)