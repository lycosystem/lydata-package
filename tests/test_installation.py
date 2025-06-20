"""Simply ensure `lydata` is installed and pytest can proceed with doctests."""

def test_is_installed() -> None:
    """Check that `lydata` can be imported (and is therefore installed)."""
    import lydata  # noqa: F401
    assert True, "lydata is not installed or cannot be imported."  # noqa: S101
