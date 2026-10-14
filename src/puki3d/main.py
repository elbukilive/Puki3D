"""Puki3D desktop application entry point."""


def main() -> int:
    from .ui import run_app

    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
