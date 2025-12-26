import os 
import sys 

from .cmd import run_cmd, exit_ 


class Tee:
    def __init__(
        self,
        log_path: str, 
    ):
        self.stdout = sys.stdout 
        self.fp = open(log_path, 'wt', encoding='utf-8')

    def write(
        self, 
        text: str,
    ):
        self.stdout.write(text)
        self.fp.write(text)
        self.flush() 

    def flush(self):
        self.stdout.flush()
        self.fp.flush()
    
    def isatty(self) -> bool:
        return self.stdout.isatty()
    

def capture_stdout_stderr(
    log_path: str,
):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    tee = Tee(log_path=log_path)

    sys.stdout = tee 
    sys.stderr = tee 


def capture_stdout_stderr_and_restart(
    script_path: str,
    log_path: str,
):
    if os.getenv('CAPTURE_STDOUT_STDERR_AND_RESTART'):
        return 
    
    os.environ['CAPTURE_STDOUT_STDERR_AND_RESTART'] = '1'

    script_path = os.path.abspath(script_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    tee = Tee(log_path=log_path)

    sys.stdout = tee 
    sys.stderr = tee 

    run_cmd([
        'python3', '-u', script_path, *sys.argv[1:],
    ])

    exit_(0)
