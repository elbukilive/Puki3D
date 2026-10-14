"""Environment diagnostic entry point; application UI will be added later."""

from importlib.metadata import PackageNotFoundError, version
import sys


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "no instalado"


def main() -> int:
    print(f"Puki3D { _package_version('puki3d') }")
    print(f"Python {sys.version.split()[0]}")
    for distribution in ("PySide6", "PyOpenGL", "numpy"):
        print(f"{distribution} {_package_version(distribution)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
