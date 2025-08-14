import os
import logging
from datetime import datetime

class CustomLogger:
    def __init__(self, log_dir="logs"):
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)

        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        log_file_path = os.path.join(self.logs_dir, log_file)

        # Create a logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter("[ %(asctime)s ] %(levelname)s %(name)s (line:%(lineno)d) - %(message)s")

        # File Handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)

        # Stream Handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # Add handlers if not already added
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)

    def get_logger(self, name=__file__):
        return logging.getLogger(os.path.basename(name))

# Usage example
if __name__ == "__main__":
    logger = CustomLogger().get_logger(__file__)
    logger.info("Custom logger initialized - file + console logging enabled.")
