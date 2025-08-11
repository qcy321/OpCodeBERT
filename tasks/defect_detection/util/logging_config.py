import logging
import threading


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'CRITICAL': '\033[91m'
    }
    RESET_COLOR = '\033[0m'

    def format(self, record):
        levelname = record.levelname
        color = self.COLORS.get(levelname, self.RESET_COLOR)
        thread_name = threading.current_thread().name
        formatted_message = super().format(record)
        return f"{color}{thread_name} - {formatted_message}{self.RESET_COLOR}"


class Log:
    def __init__(self, level=logging.DEBUG):
        self.logger = logging.getLogger()
        logger = logging.getLogger()
        logger.setLevel(level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        formatter = ColoredFormatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
