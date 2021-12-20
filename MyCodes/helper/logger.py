import logging

logging_level = {'debug': logging.DEBUG,
                 'info': logging.INFO,
                 'warning': logging.WARNING,
                 'error': logging.ERROR,
                 'critical': logging.CRITICAL}

def debug(msg):
    logging.debug(msg)
    print("DEBUG: " + msg)

def info(msg):
    logging.info(msg)
    print("INFO: " + msg)

def warning(msg):
    logging.warning(msg)
    print("WARNING: " + msg)

def error(msg):
    logging.error(msg)
    print("ERROR: " + msg)

def critical(msg):
    logging.critical(msg)
    print("CRITICAL: " + msg)

class Logger(object):
    def __init__(self, config):
        super(Logger, self).__init__()
        assert config.dict['log']['level'] in logging_level.keys()
        logging.getLogger('').handlers = []
        logging.basicConfig(filename=config.dict['log']['filename'],
                            level=logging_level[config.dict['log']['level']],
                            format='%(asctime)s - %(levelname)s : %(message)s',
                            datefmt='%Y/%m/%d %H:%M:%S')