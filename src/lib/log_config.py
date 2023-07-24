import logging


# config the root logger to print output to both stdout and `filename`
def log_config(filename, level=logging.DEBUG):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(filename, mode='w'),
            logging.StreamHandler()
        ]
    )
