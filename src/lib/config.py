import configparser
import os.path

from definitions import CONFIG_DIR


class ConfigManager:
    """
    A singleton class that read the config file only once at the first time of initialization.
    """
    _instance = None
    _is_initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if ConfigManager._is_initialized:
            return

        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(CONFIG_DIR, 'settings.ini'))

        ConfigManager._is_initialized = True
