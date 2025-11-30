"""
@bref Custom logger configuration for technical department performance analysis
Provides centralized logging functionality using file handlers.
"""


import logging
import os
from datetime import datetime


class AnalysisLogger:
    """
    @brief Custom logger class for analysis operations
    Handles log file creation and management for different analysis types
    """

    def __init__(self, log_directory="logs"):
        """
        @brief Initialize the analysis logger
        Creates log directory and configures logging handlers

        @param log_directory: Directory to store log files
        """

        self.log_directory = log_directory


    def _ensure_log_directory(self):
        """
        @brief Create log directory if it doesn't exist
        Ensures proper directory structure for log files
        """
        try:
            if not os.path.exists(self.log_directory):
                os.makedirs(self.log_directory)
        except Exception as error:
            print(f"Error creating log directory: {error}")

    def _configure_root_logger(self):
        """
        @brief Configure root logger with basic settings
        Sets up logging format and level for all loggers
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def get_analysis_logger(self, analysis_name):
        """
        @brief Create and configure a dedicated logger for specific analysis
        Each analysis gets its own log file and logger instance

        @param analysis_name: Name of the analysis for log file naming
        @return: Configured logger instance
        """
        logger = logging.getLogger(analysis_name)
        logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create file handler for this analysis
        log_filename = f"{analysis_name.lower()}_{datetime.now().strftime('%Y%m%d')}.log"
        log_filepath = os.path.join(self.log_directory, log_filename)

        try:
            file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
            file_handler.setLevel(logging.INFO)

            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
        except Exception as error:
            print(f"Error creating log file handler: {error}")

        return logger

# Global logger instance
analysis_logger = AnalysisLogger()