import json 
import pickle 
import os 
from typing import Any, Optional


def json_dump(
    data: Any,
    path: str,
    ensure_ascii: bool = False,
    indent: Optional[int] = 4,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'wt', encoding='utf-8') as w:
        json.dump(
            data,
            w, 
            ensure_ascii = ensure_ascii,
            indent = indent,
        )


def json_dumps(
    data: Any,
    ensure_ascii: bool = False,
    indent: Optional[int] = None,
) -> str:
    return json.dumps(
        data, 
        ensure_ascii = ensure_ascii,
        indent = indent,
    )


def json_load(
    path: str,
) -> Any:
    with open(path, 'rt', encoding='utf-8') as r:
        data = json.load(r) 

    return data 


def json_loads(
    data: str,
) -> Any:
    return json.loads(data)


def pickle_dump(
    data: Any,
    path: str,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'wb') as w:
        pickle.dump(
            data,
            w, 
        )
