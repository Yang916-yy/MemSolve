from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    expected_package = {"__init__.py", "ball", "py.typed"}
    actual_package = {
        path.name for path in (ROOT / "memsolve").iterdir() if path.name != "__pycache__"
    }
    if actual_package != expected_package:
        raise SystemExit(
            f"memsolve/ contract mismatch: expected {sorted(expected_package)}, "
            f"got {sorted(actual_package)}"
        )

    expected_ball = {"__init__.py", "config.py", "cuda.py", "model.py", "reference.py"}
    actual_ball = {
        path.name
        for path in (ROOT / "memsolve" / "ball").iterdir()
        if path.name != "__pycache__"
    }
    if actual_ball != expected_ball:
        raise SystemExit(
            f"memsolve/ball contract mismatch: expected {sorted(expected_ball)}, "
            f"got {sorted(actual_ball)}"
        )

    forbidden = (
        "legacy",
        "rrlsso",
        "modules_v2",
        "mathdx_backend",
        "core_controller",
        "static_core",
        "learn_eta",
        "rotary_base",
        "rotary_scale",
    )
    for path in (ROOT / "memsolve").rglob("*.py"):
        lowered = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            if token in lowered:
                raise SystemExit(f"forbidden compatibility token {token!r} in {path}")

    definitions = []
    for path in (ROOT / "memsolve").rglob("*.py"):
        if ".git" in path.parts:
            continue
        if "class MemSolve(" in path.read_text(encoding="utf-8"):
            definitions.append(path.relative_to(ROOT).as_posix())
    if definitions != ["memsolve/ball/model.py"]:
        raise SystemExit(f"MemSolve must have one implementation, got {definitions}")

    print("repository contract: ok")


if __name__ == "__main__":
    main()
