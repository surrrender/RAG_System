from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import textwrap
import venv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VENV = REPO_ROOT / ".venv"
WORKSPACE_REQUIREMENTS = REPO_ROOT / "requirements-workspace.txt"
CONSTRAINTS_DIR = REPO_ROOT / ".workspace"
CONSTRAINTS_PATH = CONSTRAINTS_DIR / "constraints-from-subvenvs.txt"
SUBPROJECTS = (
    ("Crawler", "wechat-framework-crawler"),
    ("Embedding_Indexing", "embedding-indexing"),
    ("LLM", "llm-rag"),
)
SKIP_PACKAGES = {
    "pip",
    "setuptools",
    "wheel",
    "pkg_resources",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a single top-level virtual environment for all Python subprojects."
    )
    parser.add_argument(
        "--venv-path",
        type=Path,
        default=DEFAULT_VENV,
        help=f"Path to the unified virtual environment. Defaults to {DEFAULT_VENV}.",
    )
    parser.add_argument(
        "--constraints-only",
        action="store_true",
        help="Only generate constraints from existing subproject virtual environments.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Create the virtual environment and constraints, but skip pip install.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    constraints = collect_constraints()
    write_constraints_file(constraints)

    if args.constraints_only:
        print(f"Wrote {len(constraints)} constraints to {CONSTRAINTS_PATH}")
        return 0

    ensure_venv(args.venv_path)

    if args.skip_install:
        print(
            textwrap.dedent(
                f"""
                Unified virtual environment is ready at {args.venv_path}
                Constraints were written to {CONSTRAINTS_PATH}
                Skipped installation because --skip-install was provided.
                """
            ).strip()
        )
        return 0

    python_executable = venv_python(args.venv_path)
    run(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ]
    )

    install_cmd = [
        str(python_executable),
        "-m",
        "pip",
        "install",
        "-r",
        str(WORKSPACE_REQUIREMENTS),
    ]
    if constraints:
        install_cmd.extend(["-c", str(CONSTRAINTS_PATH)])
    run(install_cmd)

    print(
        textwrap.dedent(
            f"""
            Unified environment created at {args.venv_path}
            Activate it with:
              source {args.venv_path}/bin/activate

            Then you can run from the repository root:
              crawler --help
              embedding-indexing --help
              python -m llm --help
            """
        ).strip()
    )
    return 0


def collect_constraints() -> dict[str, str]:
    constraints: dict[str, str] = {}
    local_packages = {normalize_name(package_name) for _, package_name in SUBPROJECTS}
    for subdir, _local_package in SUBPROJECTS:
        subvenv = REPO_ROOT / subdir / ".venv"
        site_packages = find_site_packages(subvenv)
        if site_packages is None:
            continue
        for dist in importlib.metadata.distributions(path=[str(site_packages)]):
            name = normalize_name(dist.metadata.get("Name", ""))
            version = dist.version
            if not name or name in SKIP_PACKAGES or name in local_packages:
                continue
            constraints.setdefault(name, version)
    return dict(sorted(constraints.items()))


def write_constraints_file(constraints: dict[str, str]) -> None:
    CONSTRAINTS_DIR.mkdir(exist_ok=True)
    lines = [f"{name}=={version}" for name, version in constraints.items()]
    content = "\n".join(lines)
    if content:
        content += "\n"
    CONSTRAINTS_PATH.write_text(content, encoding="utf-8")


def ensure_venv(venv_path: Path) -> None:
    if venv_python(venv_path).exists():
        return
    print(f"Creating unified virtual environment at {venv_path}")
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(venv_path)


def find_site_packages(venv_path: Path) -> Path | None:
    if not venv_path.exists():
        return None
    version_dir = venv_path / "lib"
    if not version_dir.exists():
        return None
    matches = sorted(version_dir.glob("python*/site-packages"))
    return matches[0] if matches else None


def venv_python(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
