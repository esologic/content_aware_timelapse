"""
Configures pytest, adding custom stuff.
"""

from pathlib import Path

import pytest
from pytest import FixtureRequest, Parser, TempPathFactory

_TEST_DIRECTORY = Path(__file__).parent.resolve()


def pytest_addoption(parser: Parser) -> None:
    """
    Adds a command-line option to control debug/test asset writing.
    :parser: pytest parser
    :return: None
    """
    parser.addoption(
        "--write-assets",
        action="store_true",
        default=True,
        help="Write local debug/test assets (images, logs, etc.)",
    )


@pytest.fixture
def artifact_root(request: FixtureRequest, tmp_path_factory: TempPathFactory) -> Path:
    """
    Returns a writable directory path for test artifacts. If the config option `--write-assets` is
    set, the directory will be in the project, if it is not set the directory will be a temporary
    directory.
    :param request: The pytest request object.
    :param tmp_path_factory: The temporary path factory.
    """
    write_assets: bool = request.config.getoption("--write-assets")

    if write_assets:
        # Persistent directory under ./test_artifacts
        root_dir = _TEST_DIRECTORY / Path("output_artifacts")

        assert root_dir.exists()

        test_dir: Path = root_dir / request.node.name
        test_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Temporary directory that pytest cleans up automatically
        test_dir = Path(tmp_path_factory.mktemp(request.node.name))

    return test_dir
