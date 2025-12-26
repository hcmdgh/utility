import threading 
import queue 
from typing import Any, Optional

from .cmd import run_python


class BackgroundTaskManager:
    def __init__(
        self,
        available_device_ids: list[int],
    ):
        self.available_device_ids = available_device_ids 

        self.device_queue = queue.Queue()
        self.thread_list = []

        for device_id in available_device_ids:
            self.device_queue.put(device_id) 

    def thread_func(
        self,
        python_path: str,
        args: Optional[dict[str, Any]],
    ):
        device_id = self.device_queue.get()
        env = { 'CUDA_VISIBLE_DEVICES': str(device_id) }

        run_python(
            path = python_path,
            args = args,
            env = env, 
        )

        self.device_queue.put(device_id)

    def submit(
        self,
        python_path: str,
        args: Optional[dict[str, Any]],
    ):
        thread = threading.Thread(
            target = self.thread_func,
            kwargs = dict(
                python_path = python_path,
                args = args, 
            ),
        )
        thread.start()
        self.thread_list.append(thread)

    def join(self):
        for thread in self.thread_list:
            thread.join() 
