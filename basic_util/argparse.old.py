import argparse 
import json 
from typing import get_origin, Any 


class Arguments(dict):
    __getattr__ = dict.__getitem__ 
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    @staticmethod 
    def parse_bool_argument(value: str) -> bool:
        value = value.strip().lower() 

        if value == 'true':
            return True 
        elif value == 'false':
            return False 
        else:
            raise ValueError 
        
    @staticmethod 
    def parse_list_argument(value: str) -> list:
        obj = json.loads(value)

        if isinstance(obj, list):
            return obj 
        else:
            raise TypeError 
        
    @staticmethod 
    def parse_dict_argument(value: str) -> dict:
        obj = json.loads(value)

        if isinstance(obj, dict):
            return obj 
        else:
            raise TypeError 
    
    def add_argument(
        self,
        name: str,
        type: Any,
        default: Any = None,
    ):
        assert not name.startswith('-')
        required = default is None 

        if type in [str, int, float]:
            pass 
        elif type == bool:
            type = self.parse_bool_argument 
        elif type == list or get_origin(type) == list:
            type = self.parse_list_argument
        elif type == dict or get_origin(type) == dict:
            type = self.parse_dict_argument
        else:
            raise TypeError 

        self.parser.add_argument(
            f"--{name}",
            type = type,
            required = required, 
            default = default,
        )

    def parse_args(self) -> Arguments:
        args = self.parser.parse_args()

        return Arguments(vars(args))

    def parse_known_args(self) -> tuple[Arguments, list[str]]:
        args, remaining_args = self.parser.parse_known_args()

        return Arguments(vars(args)), remaining_args 
