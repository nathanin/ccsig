import logging


def make_logger(logfile=None):
  # Call this one at the op of scripts
  logger = logging.getLogger("ccsig")
  logger.setLevel(logging.INFO)
  logger.propagate = False

  # remove all default handlers
  for handler in logger.handlers:
      logger.removeHandler(handler)

  # create console handler and set level to debug
  console_handle = logging.StreamHandler()
  console_handle.setLevel(logging.INFO)

  # create formatter
  formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(module)s %(lineno)s: %(message)s')
  console_handle.setFormatter(formatter)

  if logfile is not None:
    file_handle = logging.FileHandler(logfile)
    file_handle.setLevel(logging.INFO)
    file_handle.setFormatter(formatter)
    logger.addHandler(file_handle)

  # now add new handler to logger
  logger.addHandler(console_handle)
  return logger


def get_logger(app_name='ccsig'):
  # Call this one inside modules
  logger = logging.getLogger(app_name)
  return logger