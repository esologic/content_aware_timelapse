"""
Tries to do `itertools.tee`, but on disk instead of in memory.

Thank u: https://stackoverflow.com/a/70917416
"""

import pickle
import shutil
import threading
from pathlib import Path
from queue import Queue
from tempfile import NamedTemporaryFile
from typing import Any, Iterator, List, NamedTuple, Tuple, TypeVar, cast
from typing_extensions import Protocol

import h5py
import more_itertools
import numpy as np
from sentinels import NOTHING

from content_aware_timelapse.viderator.viderator_types import RGBInt8ImageType

T = TypeVar("T")


class SerializeItem(Protocol):
    """
    Describes a function that writes a given item out to disk.
    """

    def __call__(self, path: Path, items: List[Any]) -> None:  # type: ignore[explicit-any]
        """
        :param path: Path to write the serialized object to on disk.
        :param items: List of objects to serialize.
        :return: None
        """


class DeSerializeItem(Protocol):
    """
    Describes a function that loads an item from disk back into memory.
    """

    def __call__(self, path: Path) -> List[Any]:  # type: ignore[explicit-any]
        """
        :param path: Path to the object on disk.
        :return: Items loaded back into memory.
        """


class Serializer(NamedTuple):
    """
    A pair of functions, one to write and one to load items from disk.
    """

    serialize: SerializeItem
    deserialize: DeSerializeItem


def serialize_pickle(path: Path, items: List[T]) -> None:
    """
    Writes an item to disk using the built-in pickle module.
    :param path: Path to write the serialized object to on disk.
    :param items: List of objects to serialize.
    :return: None
    """

    with open(str(path), "wb") as p:
        pickle.dump(items, p)


def deserialize_pickle(path: Path) -> List[Any]:  # type: ignore[explicit-any]
    """
    Loads a pickled item from disk using the built-in pickle module.
    :param path: Path to the object on disk.
    :return: Item loaded back into memory.
    """

    with open(str(path), "rb") as p:
        return cast(List[Any], pickle.load(p))  # type: ignore[explicit-any]


PICKLE_SERIALIZER = Serializer(serialize=serialize_pickle, deserialize=deserialize_pickle)

HDF5_DATASET_NAME = "item_dataset"


def serialize_hdf5(path: Path, items: List[RGBInt8ImageType]) -> None:
    """
    Writes an item to disk using hdf5, a format for storing data arrays on disk.
    :param path: Path to write the serialized object to on disk.
    :param items: List of objects to serialize.
    :return: None
    """

    items_array = np.stack(items, axis=0)
    with h5py.File(name=str(path), mode="w") as f:
        f.create_dataset(
            HDF5_DATASET_NAME,
            shape=items_array.shape,
            dtype=items_array.dtype,
            data=items_array,
        )


def deserialize_hdf5(path: Path) -> List[RGBInt8ImageType]:
    """
    Loads an item to disk using hdf5, a format for storing data arrays on disk.
    :param path: Path to the object on disk.
    :return: Items loaded back into memory.
    """

    with h5py.File(name=str(path), mode="r", libver="latest") as f:
        return [RGBInt8ImageType(frame) for frame in np.array(f[HDF5_DATASET_NAME])]


HDF5_SERIALIZER = Serializer(serialize=serialize_hdf5, deserialize=deserialize_hdf5)


def load_queue_items(queue: "Queue[Path]", deserialize: DeSerializeItem) -> Iterator[T]:
    """
    Iterate over the items in a queue.
    Load the objects on disk back into memory and yield them.
    Before yielding the objects, deletes their source file.
    :param queue: To consume.
    :param deserialize: Function to load the items from disk.
    :return: An iterator of the items stored in the queue.
    """

    for path in iter(queue.get, NOTHING):
        output: List[T] = deserialize(path)
        path.unlink()
        yield from output


def tee_disk_cache(
    iterator: Iterator[T],
    copies: int,
    serializer: Serializer = PICKLE_SERIALIZER,
) -> Tuple[Iterator[T], ...]:
    """
    Caches the results from an input iterator onto disk rather than into memory for re-iteration
    later. Kind of like `itertools.tee`, but instead of going into memory with the copies, the
    intermediate objects are stored on disk.
    :param iterator: The iterator to duplicate.
    :param copies: The number of secondary iterators to make. Think of this like the `n` argument
    to `itertools.tee`.
    :param serializer: Defines how the objects will be stored on disk.
    :return: A tuple:
        (
            The primary iterator. Consume this one to populate the values in the secondary
            iterators.,
            The secondary iterators. When one of these is incremented, its next object
            is loaded from disk and yielded. Note that if you iterate on these past the head of
            `primary`, then the iteration will block.
        )
    """

    path_queues: List["Queue[Path]"] = [Queue() for _ in range(copies)]

    def forward_iterator() -> Iterator[T]:
        """
        Works through the input iterator, and as new times are produced, saves
        them to disk, and fills the queues with their locations.
        :return: Yields the original items from the input iterator.
        """

        for item in iterator:

            # These will get deleted after being loaded into memory later.
            with NamedTemporaryFile(mode="wb", delete=True) as primary_dump:

                primary_path = Path(primary_dump.name)
                serializer.serialize(path=primary_path, items=[item])

                for queue in path_queues:
                    with NamedTemporaryFile(mode="wb", delete=False) as secondary_dump:
                        secondary_path = Path(secondary_dump.name)
                        shutil.copy(src=primary_path, dst=secondary_path)
                        queue.put(secondary_path)

            yield item

        # Tells the queues that no more items will be coming out.
        for queue in path_queues:
            queue.put(NOTHING)

    return (forward_iterator(),) + tuple(
        load_queue_items(queue, deserialize=serializer.deserialize) for queue in path_queues
    )


def disk_buffer(
    source: Iterator[T],
    buffer_size: int,
    serializer: Serializer = HDF5_SERIALIZER,
) -> Iterator[T]:
    """
    Uses a worker thread to read from the source iterator and write the contents to disk. When
    output is required, items are deserialized and written out. Should be used to maintain pipeline
    if input iterator is slow and items are large, disk is used rather than memory to cache the
    items.

    :param source: To buffer and forward.
    :param buffer_size: The target size of the buffer of items on disk.
    :param serializer: Defines how the objects will be stored on disk.
    :return: `source` but having been buffered.
    """

    path_queue: "Queue[Path | object]" = Queue(maxsize=buffer_size)
    sentinel = object()  # Unique object to signal the end of the iterator

    def worker() -> None:
        """
        Read items from the input and serialize them to disk, putting the locations in the queue.
        :return: None
        """

        for chunk_of_items in more_itertools.chunked(source, 250):

            # Don't delete here, we'll delete after consumption.
            with NamedTemporaryFile(mode="wb", delete=False) as dump:
                dump_path = Path(dump.name)
                serializer.serialize(path=dump_path, items=chunk_of_items)
                path_queue.put(dump_path)

        # input has been exhausted.
        path_queue.put(sentinel)

    # Start the background thread to read from the input and fill the queue.
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        serialized_path = path_queue.get()
        if serialized_path is sentinel:
            break
        yield from serializer.deserialize(path=serialized_path)  # type: ignore

        serialized_path.unlink()  # type: ignore[attr-defined]

    thread.join()
