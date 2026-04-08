"""Server entry point for OpenEnv multi-mode deployment."""

import uvicorn
from app import app  # noqa: F401 – re-export the FastAPI app


def main() -> None:
    """Start the uvicorn server (used as [project.scripts] entry point)."""
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
