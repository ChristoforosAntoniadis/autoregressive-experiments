"""
Tests for the run log, phase timings and progress lines in autoregressive_experiments.runlog.
"""

from pathlib import Path
import re
import subprocess
import sys

import pytest

from autoregressive_experiments.runlog import Progress, begin_phase, finish_run, format_duration


def test_format_duration_by_hand() -> None:
    """3723 s is 1 h 2 min 3 s; 245 s is 4 min 5 s; under a minute keeps one decimal."""
    assert format_duration(3723) == "1h 02m 03s"
    assert format_duration(245) == "4m 05s"
    assert format_duration(6.7) == "6.7s"


def test_progress_writes_about_ten_lines_and_always_the_last() -> None:
    """56 origins: a line every ceil(56 / 10) = 6 steps, so at 6, 12, ..., 54 (9 lines) and at
    the last step, 56: 10 lines. Only unfinished loops show the time left."""
    progress = Progress("validation | total assets", 56)

    lines = [progress.step(done, f"origin {done}") for done in range(1, 57)]
    written = [line for line in lines if line is not None]

    assert len(written) == 10
    assert written[0] is not None and written[0].startswith("validation | total assets | 6/56")
    assert "left" in written[0]
    assert written[-1] is not None and "56/56" in written[-1] and "left" not in written[-1]


def test_phases_are_timed_and_their_shares_add_up() -> None:
    """Two phases give two rows, in order, whose shares of the run add up to 1."""
    begin_phase("first")
    begin_phase("second")

    table = finish_run()

    assert [row["phase"] for row in table] == ["first", "second"]
    assert sum(row["share"] for row in table) == pytest.approx(1.0)


def test_the_run_log_captures_prints_logger_lines_and_warnings(tmp_path: Path) -> None:
    """In a fresh process (the tee rewires its stdout and stderr, which pytest also captures):
    a print, a loguru line and a warning all reach the log file, each with the timestamp."""
    script = (
        "import warnings, sys\n"
        "from pathlib import Path\n"
        "from loguru import logger\n"
        "from autoregressive_experiments.runlog import start_run_log\n"
        f"start_run_log('check', Path(r'{tmp_path}'))\n"
        "print('a printed line')\n"
        "logger.info('a logger line')\n"
        "warnings.warn('a warning line')\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)

    logs = list(tmp_path.glob("run_*_check.log"))
    assert len(logs) == 1
    text = logs[0].read_text(encoding="utf-8")
    stamp = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| "
    assert re.search(stamp + "a printed line", text)
    assert re.search(stamp + "INFO \\| a logger line", text)
    assert re.search(stamp + ".*a warning line", text)


def test_the_run_log_lets_the_process_exit_while_a_child_still_holds_its_pipe(
    tmp_path: Path,
) -> None:
    """A child process inherits the pipe the run log reads (as joblib's workers do) and outlives
    its parent: the parent still exits within seconds, with its own lines in the log, instead of
    blocking on the pipe at shutdown."""
    script = (
        "import subprocess, sys\n"
        "from pathlib import Path\n"
        "from autoregressive_experiments.runlog import start_run_log\n"
        f"start_run_log('child', Path(r'{tmp_path}'))\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "print('parent done')\n"
    )

    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, timeout=25)

    text = next(tmp_path.glob("run_*_child.log")).read_text(encoding="utf-8")
    assert "parent done" in text
