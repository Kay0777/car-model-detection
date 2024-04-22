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
            raise FileNotFoundError(f'Config file: [{default_config_file}] file is not found')

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
            "path": dirname(__file__),

            "HOST": config.get('App', 'HOST').split('#')[0].strip(),
            "PORT": int(config.get('App', 'PORT').split('#')[0].strip()),

            # Getting Saving Files Formats
            "COUNT_OF_PROCCESSES_IMAGES": int(config.get('Settings', 'COUNT_OF_PROCCESSES_IMAGES').split('#')[0].strip()),
            "COUNT_OF_TASKS": int(config.get('Settings', 'COUNT_OF_TASKS').split('#')[0].strip()),
            "UPPER_SCORE": float(config.get('Settings', 'UPPER_SCORE').split('#')[0].strip()),

            # Getting Acceptings
            "ANALYZE_WITH_OPENVINO": True if config.get('Accept', 'ANALYZE_WITH_OPENVINO').split('#')[0].strip() == 'y' else False,
            
            # Getting Classes Files
            "CLASS_NAMES": config.get('Classfiles', 'CLASS_NAMES').split('#')[0].strip(),
            "CLASS_NAMES_WITH_IDS": config.get('Classfiles', 'CLASS_NAMES_WITH_IDS').split('#')[0].strip(),

            # Getting Using Model Settings
            "CAR_TORCH_MODEL": config.get('Models', 'CAR_TORCH_MODEL').split('#')[0].strip(),
            "CAR_OPENVINO_MODEL": config.get('Models', 'CAR_OPENVINO_MODEL').split('#')[0].strip(),

            # Getting Using Device Type
            "DEVICE": config.get('Device', 'DEVICE').split('#')[0].strip(),
        }

    def __str__(self) -> str:
        return str(self.__settings)

    def __repr__(self) -> str:
        return str(self.__settings)

    def __getitem__(self, __name: str) -> Any:
        return self.__settings.get(__name, None)

CONF = Settings()
