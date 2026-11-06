"""Shared pytest configuration."""


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running dataset test (deselect with -m 'not slow')")
