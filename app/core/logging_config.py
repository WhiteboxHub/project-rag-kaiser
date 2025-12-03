"""Logging configuration."""
import logging
import logging.config


def setup_logging():
    """Configure logging for the application."""
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": "kaiser_rag.log",
                "encoding": "utf-8"
            }
        },
        "loggers": {
            "": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": True
            }
        }
    }
    
    logging.config.dictConfig(logging_config)
