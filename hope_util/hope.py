import json 
import re 
from typing import Any 

from ..basic_util import run_cmd 


def _extract_hope_run_id(output: str) -> int:
    for line in output.splitlines():
        line = line.strip() 

        if line.startswith('INFO: run_id:') and line.endswith('Use `hope killjob/stop` or `hopeml kill` to stop job properly!'):
            match = re.search(r"run_id:\s*(\d+)", line)

            if match:
                return int(match.group(1)) 
            else:
                return -1

    return -1  


def hope_run(
    *,
    hope_file: str = 'template.hope',
    app_name: str,
    script: str,
    gpu: int,
    memory: int,
    cpu: int,
    image: str,
    queue: str = 'root.zw05_training_cluster.hadoop-waimaiadrd.gpu_job',
    args: dict[str, Any] = dict(),
) -> int:
    cmd = [
        'hope', 'run', hope_file,
        f"-Dafo.app.name={app_name}",
        f"-Dworker.gcores80g={gpu}",
        f"-Dworker.memory={memory * 1024}",
        f"-Dworker.vcore={cpu}",
        f"-Dworker.script={script}",
        f"-Dafo.docker.image.name={image}",
        f"-Dqueue={queue}",
    ]

    for key, value in args.items():
        if isinstance(value, (str, int, float, bool)):
            value = str(value)
        elif isinstance(value, (tuple, list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        else:
            raise TypeError 
        
        cmd.append(f"-Dargs.{key}={value}")

    _, stdout_stderr_text = run_cmd(
        cmd,
        capture_stdout_stderr = True,
    )

    run_id = _extract_hope_run_id(output=stdout_stderr_text)

    return run_id 
