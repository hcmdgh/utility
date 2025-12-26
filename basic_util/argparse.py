from pydantic import BaseModel 
from pydantic_core import PydanticUndefined
import argparse 
import json 
from typing import Any, get_origin 


def str_2_bool(s: str) -> bool: 
    s = s.strip().lower()

    if s == 'true':
        return True
    elif s == 'false':
        return False
    else:
        raise ValueError
    

def str_2_list(s: str) -> list:
    list_obj = json.loads(s) 
    assert isinstance(list_obj, list) 

    return list_obj 


def str_2_dict(s: str) -> dict:
    dict_obj = json.loads(s) 
    assert isinstance(dict_obj, dict) 

    return dict_obj 


class BaseArguments(BaseModel, validate_default=True, validate_assignment=True, protected_namespaces=()):
    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser()

        for arg_name in cls.model_fields:
            arg_type = cls.model_fields[arg_name].annotation
            arg_default = cls.model_fields[arg_name].default

            if arg_type in [str, int, float]: 
                pass 
            elif arg_type == bool:
                arg_type = str_2_bool 
            elif arg_type == list or get_origin(arg_type) == list:
                arg_type = str_2_list
            elif arg_type == dict or get_origin(arg_type) == dict:
                arg_type = str_2_dict
            else:
                raise TypeError 
            
            arg_default = arg_default if arg_default != PydanticUndefined else None

            parser.add_argument(
                f"--{arg_name}",
                type = arg_type,
                required = arg_default is None,
                default = arg_default,
            )

        args = parser.parse_args()

        return cls(**vars(args))
