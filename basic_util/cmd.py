import subprocess 
import json 
import os 
import shlex 
import sys 
from datetime import datetime 
from typing import Optional, Any 


def echo(
    data: Any,
    cmd_type: str = 'echo',
):
    def get_datetime_prefix(
        cmd_type: str,
    ) -> str:
        datetime_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if cmd_type:
            return f"[{datetime_str} {cmd_type}] "
        else:
            return f"[{datetime_str}] "
        
    msg = get_datetime_prefix(cmd_type=cmd_type) + str(data)
    print(msg)


def exit_(
    exit_code: int,
):
    echo(f"exit {exit_code}", cmd_type='exit')
    exit(exit_code)


def run_bash_cmd(
    cmd: list[Any],
    env: Optional[dict[str, Any]] = None,
    log_path: Optional[str] = None,
    verbose: bool = True,
    capture_stdout_stderr: bool = False,
    exit_on_error: bool = True,
    discard_stdout: bool = False,
) -> tuple[int, str]:
    cmd = [str(item) for item in cmd]

    cmd_str = shlex.join(cmd)

    if verbose:
        echo(cmd_str, cmd_type='run')
        print()

    if env:
        full_env = os.environ.copy() 
        full_env.update({ k: str(v) for k, v in env.items() })
    else:
        full_env = None 

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        if not discard_stdout:
            cmd_str += ' 2>&1 | tee ' + shlex.quote(log_path)  
        else:
            cmd_str += ' 2>&1 1>/dev/null | tee ' + shlex.quote(log_path)

    process = subprocess.Popen(
        ['/bin/bash', '-c', cmd_str],
        env = full_env,
        stdout = subprocess.PIPE, 
        stderr = subprocess.STDOUT, 
        text = True,
        bufsize = 1,
    )

    stdout_stderr_text = ''

    while True:
        line = process.stdout.readline()
        
        if line == '' and process.poll() is not None:
            break
        
        if line:
            sys.stdout.write(line)
            sys.stdout.flush()

            if capture_stdout_stderr:
                stdout_stderr_text += line 

    return_code = process.returncode

    if exit_on_error and return_code != 0:
        exit_(return_code)

    return return_code, stdout_stderr_text


run_cmd = run_bash_cmd 


def run_python(
    path: str,
    module: bool = False,
    args: Optional[dict[str, Any] | list[str]] = None,
    env: Optional[dict[str, Any]] = None,
    log_path: Optional[str] = None,
    exit_on_error: bool = True,
    discard_stdout: bool = False,
):
    if not module:
        cmd = ['python3', '-u']
    else:
        cmd = ['python3', '-u', '-m']

    cmd.append(path)

    if not args:
        pass 
    elif isinstance(args, list):
        cmd.extend(args)
    elif isinstance(args, dict):
        for key, value in args.items():
            cmd.append(f"--{key}")

            if isinstance(value, (list, tuple, dict)):
                value = json.dumps(value, ensure_ascii=False)

            cmd.append(value)
    else:
        raise TypeError

    run_bash_cmd(
        cmd,
        env = env,
        log_path = log_path,
        exit_on_error = exit_on_error,
        discard_stdout = discard_stdout,
    )


def rm_f(
    path: str,
):
    run_bash_cmd(['rm', '-f', path])


def rm_rf(
    path: str,
):
    run_bash_cmd(['rm', '-rf', path])


def cp_r(
    src_path: str,
    dest_path: str,
):
    run_bash_cmd(['cp', '-r', src_path, dest_path])


def pwd():
    echo(os.getcwd(), cmd_type='pwd')


def cd(
    path: str,
):
    echo(path, cmd_type='cd') 
    os.chdir(path)
    pwd()


def get_script_dir(
    script_path: str, 
) -> str:
    return os.path.abspath(os.path.dirname(script_path))


def cd_script_dir(
    script_path: str,
) -> str:
    script_dir = os.path.abspath(os.path.dirname(script_path))
    cd(script_dir)

    return script_dir 


def get_env(
    name: str,
) -> str:
    value = os.getenv(name)

    if not value:
        raise ValueError 
    
    return value 


def set_env(
    name: str,
    value: Any,
):
    if value is None:
        value = ''

    value = str(value)
    echo(f"{name} = {value}", cmd_type='env')

    os.environ[name] = value 
