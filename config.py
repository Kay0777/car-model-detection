from configparser import ConfigParser
from argparse import ArgumentParser
from typing import Any

from os.path import exists, join, dirname, isfile


class SettingsBase(type):
    def __call__(cls, *args: tuple, **kwargs: dict) -> Any:
        if not hasattr(cls, 'instance'):
            cls.instance = super().__call__(*args, **kwargs)
        return cls.instance


class Settings(metaclass=SettingsBase):
    def __init__(self, default_config_file: str = 'config.ini'):
        if not exists(default_config_file) or not isfile(default_config_file):
            raise FileNotFoundError(
                f'Config file: [{default_config_file}] file is not found')

        if isfile(default_config_file):
            default_config_file = join(dirname(__file__), default_config_file)

        parser = ArgumentParser(
            description='Example script with argparse and config file.')

        # Add the -c/--config parameter
        parser.add_argument('-c', '--config',
                            default=default_config_file,
                            help='Specify the config file path')

        args = parser.parse_args()

        config = ConfigParser()
        config.read(args.config)

        # Access values from the configuration file
        self.__settings = {
            # Getting Main Path
            "PATH": dirname(__file__),

            # Getting Server Settings
            "HOST": self.variable(config.get('App', 'HOST')),
            "PORT": int(self.variable(config.get('App', 'PORT'))),
            "WORKERS": int(self.variable(config.get('App', 'WORKERS'))),

            # Getting Redis Settings
            "REDIS_HOST": self.variable(config.get('Redis', 'REDIS_HOST')),
            "REDIS_PORT": int(self.variable(config.get('Redis', 'REDIS_PORT'))),
            "REDIS_DB": int(self.variable(config.get('Redis', 'REDIS_DB'))),
            "IMAGE_WAIT_TTL": int(self.variable(config.get('Redis', 'IMAGE_WAIT_TTL'))),
            "IMAGE_BATCH_NAME": self.variable(config.get('Redis', 'IMAGE_BATCH_NAME')),

            # Getting Image Settings
            "IMAGE_HEIGHT_TO_MODEL": int(self.variable(config.get('Settings', 'IMAGE_HEIGHT_TO_MODEL'))),
            "IMAGE_WIDTH_TO_MODEL": int(self.variable(config.get('Settings', 'IMAGE_WIDTH_TO_MODEL'))),
            "IMAGE_SIMILARITY_K": int(self.variable(config.get('Settings', 'IMAGE_SIMILARITY_K'))),
            "IMAGE_HANDLE_SIZE": int(self.variable(config.get('Settings', 'IMAGE_HANDLE_SIZE'))),
            "IMAGE_BATCH_SIZE": int(self.variable(config.get('Settings', 'IMAGE_BATCH_SIZE'))),

            # Getting Classes Files
            'CLASS_NAMES': self.variable(config.get('Files', 'CLASS_NAMES')),
            'CLASS_NAMES_WITH_IDS': self.variable(config.get('Files', 'CLASS_NAMES_WITH_IDS')),

            # Getting Using Model Settings
            'OPENVINO_MODEL_DEVICE': self.variable(config.get('Model', 'OPENVINO_MODEL_DEVICE')),
            'OPENVINO_MODEL_XML': self.variable(config.get('Model', 'OPENVINO_MODEL_XML')),
            'OPENVINO_MODEL_WEIGHTS': self.variable(config.get('Model', 'OPENVINO_MODEL_WEIGHTS')),
        }

    def variable(self, word: str) -> str:
        if '#' in word:
            word = word.split('#')[0]
        return word.strip()

    def __str__(self) -> str:
        return str(self.__settings)

    def __repr__(self) -> str:
        return str(self.__settings)

    def __getitem__(self, __name: str) -> Any:
        return self.__settings.get(__name, None)


CONF = Settings()
