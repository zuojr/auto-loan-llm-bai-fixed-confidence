import importlib.util
import time
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "02_score_openai_gpt_api.py"
    spec = importlib.util.spec_from_file_location("score_openai_gpt_api", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_format_duration_uses_hh_mm_ss():
    mod = _load_script()

    assert mod.format_duration(0) == "00:00:00"
    assert mod.format_duration(65) == "00:01:05"
    assert mod.format_duration(3661) == "01:01:01"


def test_progress_line_includes_elapsed_eta_and_batch_timing():
    mod = _load_script()
    started_at = time.monotonic() - 100

    line = mod.progress_line(
        done=20,
        total=100,
        out_path="predictions/out.csv",
        started_at=started_at,
        last_batch_seconds=12.3,
    )

    assert "wrote 20/100 (20.0%)" in line
    assert "elapsed" in line
    assert "eta" in line
    assert "last_call 12.3s" in line
    assert "avg_row" in line
    assert "predictions/out.csv" in line
