"""
Run logs, phase timings and progress lines for every pipeline step.

``start_run_log(mode)``
    Puts a tee on the process's stdout and stderr file descriptors. Every line written to
    either (loguru messages, ``print``, Python warnings, text from compiled libraries, and a
    crash's traceback) is prefixed with ``YYYY-MM-DD HH:MM:SS |``, shown on the terminal and
    appended to ``logs/run_<YYYYmmdd_HHMMSS>_<mode>.log``. loguru is reconfigured without its
    own timestamp, so its lines read ``<time> | INFO | message`` like every other line. Lines
    are written as they come, so a run in the background can be followed while it works.

``begin_phase(name)`` / ``finish_run()``
    A phase lasts from its begin_phase call to the next one, or to finish_run. Each phase is
    logged with its duration, and finish_run logs a table of phases (seconds and share of the
    run).

``Progress``
    Progress lines for long loops, in one format: ``LABEL | 34/56 | detail | 3m12s elapsed |
    ~2m05s left``, written at most PROGRESS_LINES times per loop (plus its last step).

Adapted from the stcif project's runlog module, without its cloud upload, VM cost and experiment
tracking.

Input: none.

Output: the log file in ``logs/`` (not committed: logs differ per computer and run).
"""

import atexit
from datetime import datetime
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any
import warnings

from loguru import logger

from autoregressive_experiments.config import LOGS_DIR

# Timestamp format of every log line.
STAMP = "%Y-%m-%d %H:%M:%S"
# A long loop writes about this many progress lines: enough to follow it, few enough to read the
# log. Its last step is always written.
PROGRESS_LINES = 10


class Tee:
    """
    Timestamping tee on file descriptors 1 and 2.

    The original descriptors are duplicated, both are pointed at one pipe, and a background
    thread reads the pipe line by line, stamping each line before writing it to the original
    stdout and to the log file.

    Attributes
    ----------
    path : Path
        The log file.
    file : TextIO
        The open log file, line-buffered.
    """

    def __init__(self, path: Path) -> None:
        """
        Open the log file and redirect stdout and stderr through the pipe.

        Parameters
        ----------
        path : Path
            The log file to append to.
        """
        self.path = path
        # Open for the whole run and closed in close(), so no with block.
        self.file = open(path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115
        self.original_out = os.dup(1)
        self.original_err = os.dup(2)
        self.read_end, self.write_end = os.pipe()
        os.dup2(self.write_end, 1)
        os.dup2(self.write_end, 2)
        self.thread = threading.Thread(target=self.pump, name="runlog-tee", daemon=True)
        self.thread.start()

    def pump(self) -> None:
        """Read the pipe and emit each complete line, until the write end closes."""
        pending = b""
        while True:
            chunk = os.read(self.read_end, 65536)
            if not chunk:
                break
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                self.emit(line)
        if pending:
            self.emit(pending)
        self.file.flush()

    def emit(self, raw: bytes) -> None:
        """
        Stamp one line and write it to the terminal and the log file.

        Parameters
        ----------
        raw : bytes
            One line, without its newline.
        """
        text = raw.decode("utf-8", errors="replace").rstrip("\r")
        line = f"{time.strftime(STAMP)} | {text}\n"
        os.write(self.original_out, line.encode("utf-8"))
        self.file.write(line)

    def close(self) -> None:
        """Restore the descriptors, let the pump drain the pipe, and close the file."""
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (OSError, ValueError):
                pass
        os.dup2(self.original_out, 1)
        os.dup2(self.original_err, 2)
        os.close(self.write_end)
        # Bounded, so shutdown never hangs on a worker that still holds the write end.
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            # A child process (such as a joblib worker or its resource tracker) inherited the
            # write end and still holds it, so the pump is blocked reading. On Windows, closing
            # a descriptor another thread is reading blocks as well, so the pipe, the copies of
            # the terminal and the file are left for process exit to close; the pump is a
            # daemon thread and doesn't keep the process alive.
            self.file.flush()
            return
        os.close(self.read_end)
        os.close(self.original_out)
        os.close(self.original_err)
        self.file.close()


# The running tee and the phases of this run.
TEE: Tee | None = None
PHASES: list[dict[str, Any]] = []
CURRENT: dict[str, Any] | None = None


def start_run_log(mode: str, log_dir: Path | None = None) -> Path:
    """
    Start the timestamped run log and send loguru's output through it.

    Parameters
    ----------
    mode : str
        The step, for the file name, e.g. "evaluate" or "garch".
    log_dir : Path or None, default=None
        Folder for the log; LOGS_DIR (``logs/`` at the project root) when None.

    Returns
    -------
    Path
        The log file being written.

    Notes
    -----
    Processing:
    1. Make the Python streams line-buffered, so prints and logger lines reach the pipe in
       order and at once, whether or not the output is a terminal.
    2. Start the tee and register its closing at exit (after a crash's traceback is printed).
    3. Replace loguru's handlers with one plain handler on stderr: level and message only,
       since the tee adds the timestamp.
    4. Show each distinct warning once per code location (Python's default): statsmodels
       resets its convergence warnings to "always", which floods the log; failed fits are
       counted in the reports anyway.

    Calling it again in the same process returns the running log.
    """
    global TEE
    if TEE is not None:
        return TEE.path
    folder = LOGS_DIR if log_dir is None else log_dir
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"run_{datetime.now():%Y%m%d_%H%M%S}_{mode}.log"
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(line_buffering=True)
    TEE = Tee(path)
    atexit.register(stop_run_log)
    logger.remove()
    logger.add(sys.stderr, format="{level} | {message}", colorize=False)
    warnings.simplefilter("default")
    logger.info(f"Run log: {path}")
    return path


def stop_run_log() -> None:
    """
    End the run log: restore stdout and stderr and close the file.

    Returns
    -------
    None
        Runs at exit; tests call it to restore the descriptors.
    """
    global TEE
    if TEE is not None:
        TEE.close()
        TEE = None


def format_duration(seconds: float) -> str:
    """
    A duration as "1h 02m 03s", "4m 05s" or "6.7s".

    Parameters
    ----------
    seconds : float
        The duration.

    Returns
    -------
    str
        The formatted duration.
    """
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int(seconds % 3600 // 60):02d}m {int(seconds % 60):02d}s"
    if seconds >= 60:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    return f"{seconds:.1f}s"


def close_phase() -> None:
    """Log the running phase's duration and add it to PHASES."""
    global CURRENT
    if CURRENT is None:
        return
    CURRENT["seconds"] = time.perf_counter() - CURRENT["start"]
    logger.info(f"DONE  {CURRENT['phase']} in {format_duration(CURRENT['seconds'])}")
    PHASES.append(CURRENT)
    CURRENT = None


def begin_phase(name: str) -> None:
    """
    Start timing a phase, ending the one that was running.

    Parameters
    ----------
    name : str
        The phase's label in the log and the timings table.
    """
    global CURRENT
    close_phase()
    CURRENT = {"phase": name, "start": time.perf_counter()}
    logger.info(f"START {name}")


def finish_run() -> list[dict[str, Any]]:
    """
    Close the running phase and log the table of phases.

    Returns
    -------
    list of dict of str to Any
        One entry per phase: "phase", "seconds" and "share" (of the run's total).
    """
    close_phase()
    total = sum(phase["seconds"] for phase in PHASES)
    # Phases too short to measure share the run equally rather than dividing by zero.
    table = [
        {
            "phase": p["phase"],
            "seconds": p["seconds"],
            "share": p["seconds"] / total if total > 0 else 1 / len(PHASES),
        }
        for p in PHASES
    ]
    if table:
        width = max(len(row["phase"]) for row in table)
        lines = [
            f"  {row['phase']:<{width}}  {row['seconds']:>8.1f}s  {row['share']:6.1%}"
            for row in table
        ]
        logger.info("Phase timings:\n" + "\n".join(lines))
        logger.info(f"Run total: {format_duration(total)}")
    PHASES.clear()
    return table


class Progress:
    """
    Progress lines for one long loop: "LABEL | step/total | detail | elapsed | ~left".

    Attributes
    ----------
    label : str
        Names the loop, e.g. "RANDOM FOREST (3 lags) | validation | total assets".
    total : int
        Steps in the loop.
    every : int
        A line is written every `every` steps (and at the last step).
    """

    def __init__(self, label: str, total: int) -> None:
        """
        Start timing a loop.

        Parameters
        ----------
        label : str
            Names the loop in every line.
        total : int
            Steps in the loop.
        """
        self.label = label
        self.total = total
        # About PROGRESS_LINES lines per loop, however long it is.
        self.every = max(1, math.ceil(total / PROGRESS_LINES))
        self.start = time.perf_counter()

    def step(self, done: int, detail: str = "") -> str | None:
        """
        Report that `done` steps are finished; write a line if it's time.

        Parameters
        ----------
        done : int
            Steps finished so far, 1..total.
        detail : str, default=""
            What the step was, e.g. the origin month.

        Returns
        -------
        str or None
            The line written, or None when this step writes none.

        Notes
        -----
        The time left assumes the remaining steps take as long, on average, as the finished
        ones did.
        """
        if done % self.every and done != self.total:
            return None
        elapsed = time.perf_counter() - self.start
        left = elapsed / done * (self.total - done)
        parts = [self.label, f"{done}/{self.total}", detail, f"{format_duration(elapsed)} elapsed"]
        if done < self.total:
            parts.append(f"~{format_duration(left)} left")
        line = " | ".join(part for part in parts if part)
        logger.info(line)
        return line
