"""Cross-process file locking for pipeline runs."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TextIO

from ..errors import PipelineLockError
from .io import ensure_dir


class InterProcessFileLock:
    """Non-blocking advisory file lock backed by the operating system."""

    def __init__(self, lock_path: str | Path):
        self.lock_path = Path(lock_path)
        self._handle: TextIO | None = None

    def acquire(self) -> None:
        """Acquire the lock or fail immediately if it is already held."""
        if self._handle is not None:
            return

        ensure_dir(self.lock_path)
        handle = open(self.lock_path, "a+", encoding="utf-8")

        try:
            self._acquire_os_lock(handle)
            self._write_metadata(handle)
            self._handle = handle
        except Exception:
            handle.close()
            raise

    def release(self) -> None:
        """Release the lock if currently held."""
        if self._handle is None:
            return

        handle = self._handle
        self._handle = None

        try:
            try:
                handle.seek(0)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
            except OSError:
                pass
            self._release_os_lock(handle)
        finally:
            handle.close()

    def __enter__(self) -> "InterProcessFileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.release()

    def _acquire_os_lock(self, handle: TextIO) -> None:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write("\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                try:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]
                except ImportError:
                    # fcntl not available on Windows, but we should never reach here
                    # because os.name == "nt" should be True
                    pass
        except OSError as exc:
            raise PipelineLockError(
                f"Another longtext pipeline process is already running for this input. "
                f"Lock file: {self.lock_path}"
            ) from exc

    def _release_os_lock(self, handle: TextIO) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
            except ImportError:
                # fcntl not available on Windows, but we should never reach here
                # because os.name == "nt" should be True
                pass

    def _write_metadata(self, handle: TextIO) -> None:
        metadata = {
            "pid": os.getpid(),
            "acquired_at": datetime.now().isoformat(timespec="seconds"),
            "lock_path": str(self.lock_path),
        }
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps(metadata, ensure_ascii=False))
        handle.flush()
        os.fsync(handle.fileno())
