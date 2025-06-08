# Usage example for generating mock report

from pathlib import Path
from quality_control.quality_reporter import QualityReporter
from loguru import logger

# Method 1: Create mock reporter and generate report
output_directory = Path("./test_reports")
mock_reporter = QualityReporter.create_mock_reporter(output_directory)
report_path = mock_reporter.generate_mock_report(output_directory)

print(f"Mock report generated at: {report_path}")


# Method 2: If you want to add this to your pipeline class
# Add this method to your EEGPipeline class:

def generate_mock_quality_report(self, output_dir: Path = None):
    """Generate a mock quality report for testing layout and design"""

    if output_dir is None:
        output_dir = Path("./mock_reports")

    mock_reporter = QualityReporter.create_mock_reporter(output_dir)
    report_path = mock_reporter.generate_mock_report(output_dir)

    logger.info(f"Mock quality report generated at: {report_path}")
    return report_path

# Then you can call it like:
# pipeline = EEGPipeline()
# pipeline.generate_mock_quality_report()