"""
autopush.py — watches the project directory and auto-commits + pushes
any changes to GitHub after a short debounce delay.

Usage:
    python autopush.py

Stop with Ctrl+C.
Requires: pip install watchdog
"""

import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

WATCH_DIR   = Path(__file__).parent
DEBOUNCE_S  = 5          # seconds to wait after last change before committing
IGNORE_DIRS = {'.git', '__pycache__', '.ipynb_checkpoints', '.claude'}
IGNORE_EXT  = {'.pyc', '.pyo'}


def run(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, cwd=WATCH_DIR,
                            capture_output=True, text=True)
    return (result.stdout + result.stderr).strip()


class ChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self._pending = False
        self._last_event = 0.0

    def on_any_event(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if any(part in IGNORE_DIRS for part in path.parts):
            return
        if path.suffix in IGNORE_EXT:
            return
        self._pending = True
        self._last_event = time.time()

    def flush_if_ready(self):
        if not self._pending:
            return
        if time.time() - self._last_event < DEBOUNCE_S:
            return

        self._pending = False
        status = run('git status --porcelain')
        if not status:
            return  # nothing to commit

        stamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f'\n[{stamp}] Changes detected — committing...')
        print(run('git add -A'))
        print(run(f'git commit -m "auto: save {stamp}"'))
        print(run('git push'))
        print('Done.\n')


if __name__ == '__main__':
    handler  = ChangeHandler()
    observer = Observer()
    observer.schedule(handler, str(WATCH_DIR), recursive=True)
    observer.start()
    print(f'Watching {WATCH_DIR}  (Ctrl+C to stop)')
    try:
        while True:
            time.sleep(1)
            handler.flush_if_ready()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
