import re
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def data_dir() -> Path:
    """Path to the tests/data directory."""
    return (Path(__file__) / ".." / "data").resolve()


@pytest.fixture(scope="session")
def raises_context() -> Callable[[Exception], pytest.RaisesExc]:
    """
    Wrapper around pytest.raises for convenience.

    Allows for direct passing of an `Exception` instance into a `pytest.raises` context,
    rather than needing to provide the `Exception` class and regex-escaped string
    separately, as arguments.

    `pytest.raises` is typically used in the following manner:

    ```python
    import re
    import pytest

    with pytest.raises(Exception, match=re.escape("error text")):
      ...
    ```

    raises_context allows us to instead write:

    ```python
    with raises_context(Exception("error text")):
      ...
    ```

    without the need to import the other two modules. Use `raises_context` as a fixture
    in the test definition to gain access to it.
    """

    def _inner(exception: Exception) -> pytest.RaisesExc:
        return pytest.raises(type(exception), match=re.escape(str(exception)))

    return _inner
