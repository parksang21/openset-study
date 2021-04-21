from cutils import get_yaml
import yaml


def get_config(path):
    return Config(get_yaml(path))


class Config:
    def __init__(self, initial_data):
        if not isinstance(initial_data, dict):
            raise Exception('param must be dictionary')

        for key, value in initial_data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                if value == "None":
                    value = None
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def dct(self):
        x = {}
        for n, v in self.__dict__.items():
            if isinstance(v, Config):
                x[n] = v.dct()
            else:
                x[n] = v
        return x

    def save(self, path):
        with open(path, 'w+') as f:
            yaml.dump(self.dct(), f)

        print(f"config file has been saved at [[ {path} ]]")

    def __call__(self):
        return self.__dict__

