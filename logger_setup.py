import logging
import logging.handlers
import os
import time
from config import LOG_DIR


class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """A RotatingFileHandler that retries on PermissionError (Windows file lock)."""
    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename("%s.%d" % (self.baseFilename, i))
                dfn = self.rotation_filename("%s.%d" % (self.baseFilename, i + 1))
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    # Retry up to 5 times with 0.1s delay
                    for attempt in range(5):
                        try:
                            os.rename(sfn, dfn)
                            break
                        except PermissionError:
                            time.sleep(0.1)
            dfn = self.rotation_filename(self.baseFilename + ".1")
            if os.path.exists(dfn):
                os.remove(dfn)
            # Retry renaming the current log
            for attempt in range(5):
                try:
                    self.rotate(self.baseFilename, dfn)
                    break
                except PermissionError:
                    time.sleep(0.1)
        if not self.delay:
            self.stream = self._open()


def setup_logging(debug: bool = False) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    detailed_format = "[%(asctime)s] [%(name)s] [%(levelname)s] [%(threadName)s] %(message)s"
    simple_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    detailed_formatter = logging.Formatter(fmt=detailed_format, datefmt=date_format)
    simple_formatter = logging.Formatter(fmt=simple_format, datefmt=date_format)

    # Main log: 10 MB per file, keep 5 backups
    log_file_path = os.path.join(LOG_DIR, "app.log")
    file_handler = SafeRotatingFileHandler(
        filename=log_file_path,
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding="utf-8",
        delay=True
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Error log: 10 MB per file, keep 5 backups
    error_log_path = os.path.join(LOG_DIR, "error.log")
    error_handler = SafeRotatingFileHandler(
        filename=error_log_path,
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding="utf-8",
        delay=True
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)

    console_handler = logging.StreamHandler()
    console_level = logging.DEBUG if debug else logging.INFO
    console_handler.setLevel(console_level)
    console_handler.setFormatter(simple_formatter)

    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    for noisy in ("pymongo", "urllib3", "requests", "urllib3.connectionpool"):
        logging.getLogger(noisy).setLevel(logging.WARNING)