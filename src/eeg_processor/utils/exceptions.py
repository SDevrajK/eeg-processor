"""Custom exceptions for EEG Processor."""


class EEGProcessorError(Exception):
    """Base exception for EEG Processor."""
    pass


class ConfigurationError(EEGProcessorError):
    """Raised when there are issues with configuration."""
    pass


class DataLoadError(EEGProcessorError):
    """Raised when data loading fails."""
    pass


class ProcessingError(EEGProcessorError):
    """Raised during EEG processing operations."""
    pass


class QualityControlError(EEGProcessorError):
    """Raised during quality control operations."""
    pass


class FileFormatError(DataLoadError):
    """Raised when file format is not supported or corrupted."""
    pass


class ValidationError(EEGProcessorError):
    """Raised when data validation fails."""
    pass