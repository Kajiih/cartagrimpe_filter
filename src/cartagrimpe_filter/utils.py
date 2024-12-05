"""General utils."""

from collections.abc import Iterable, Mapping
from typing import Any, Literal, overload


@overload
def get_first[K, V, D](d: Mapping[K, V], keys: Iterable[K], default: D = None) -> V | D: ...


@overload
def get_first[K, V, D](
    d: Mapping[K, V], keys: Iterable[K], default: Any = None, no_default: Literal[True] = True
) -> V: ...


@overload
def get_first[K, V, D](
    d: Mapping[K, V], keys: Iterable[K], default: D = None, no_default: bool = True
) -> V | D: ...


def get_first[K, V, D](
    d: Mapping[K, V], keys: Iterable[K], default: D = None, no_default: bool = False
) -> V | D:
    """
    Return the first value for the first key that exists in the mapping.

    Args:
        d: The dictionary to search in.
        keys: The sequence of keys to look for.
        default: The value to return if no key is found.
        no_default: If `True`, raises a `KeyError` if no key is found.

    Returns:
        The value associated with the first found key, or the default value if not found.

    Raises:
        KeyError: If `no_default` is `True` and none of the keys are found.

    Example:
        >>> d = {"a": 1, "b": 2, "c": 3}
        >>> get_first(d, ["x", "a", "b"])  # Returns: 1
        >>> get_first(d, ["x", "y"], default=0)  # Returns: 0
        >>> get_first(d, ["x", "y"], no_default=True)  # Raises: KeyError
    """
    for key in keys:
        if key in d:
            return d[key]

    if no_default:
        raise KeyError(f"None of the keys {list(keys)} were found in the mapping.")  # noqa: TRY003

    return default
