import sys
from enum import Enum
from typing import Union

import dotenv
from dotenv import load_dotenv
import os
from loguru import logger


class LogLevel(Enum):
    NONE = "NONE"
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    WARNING = "WARNING"
    INFO = "INFO"
    ERROR = "ERROR"


LOG_LEVELS = [loader.value for loader in LogLevel]


def configure_logger(log_level: Union[str, LogLevel]):
    level = log_level.value if isinstance(log_level, LogLevel) else log_level
    # yes, it is duplicate but it is nice to avoid cross-module dependencies here
    if level.upper() != LogLevel.NONE.value:
        logger.add(sys.stdout, level=level.upper())

def load_environment_keys(debug: bool = True):
    e = dotenv.find_dotenv()
    if debug:
        print(f"environment found at {e}")
    has_env: bool = load_dotenv(e, verbose=True, override=True)
    if not has_env:
        print("Did not found environment file, using system OpenAI key (if exists)")
    openai_key = os.getenv('OPENAI_API_KEY')
    return openai_key