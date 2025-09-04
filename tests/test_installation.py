"""Simply ensure `lydata` is installed and pytest can proceed with doctests."""

import os


def test_env_vars() -> None:
    """Check that the .env file is loaded and the Github token is accessible."""
    token_env_var: str = os.environ.get("GITHUB_TOKEN", "nope")
    assert "github" in token_env_var, "GITHUB_TOKEN env var not accessible"


def test_is_installed() -> None:
    """Check that `lydata` can be imported (and is therefore installed)."""
    import lydata  # noqa: F401

    assert True, "lydata is not installed or cannot be imported."  # noqa: S101
