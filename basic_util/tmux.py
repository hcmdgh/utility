import os 

from .cmd import run_cmd


def tmux_run_self(
    script_path: str,
):
    flag = os.getenv('TMUX') 

    if flag:
        return 
    
    script_name = os.path.basename(script_path)
    script_dir = os.path.abspath(os.path.dirname(script_path))
    os.chdir(script_dir)

    run_cmd(['tmux', 'new-session', '-d'])
    run_cmd(['tmux', 'send-keys', f'python3 -u "{script_name}"', 'C-m'])
    run_cmd(['tmux', 'attach'])

    exit(0) 
