import time
from datetime import datetime


class Logger:

    def __init__(self):
        pass

    def timed_print(self, *args, **kwargs):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{current_time}]", *args, **kwargs)
