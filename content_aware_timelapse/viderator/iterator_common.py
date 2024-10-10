"""
Common operations on iterators, which are the canonical representation of videos, data sources etc
throughout this application.
"""

import collections
import datetime
import itertools
import logging
import threading
from queue import Queue
from typing import Deque, Iterator, Tuple, TypeVar

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


def first_item_from_iterator(iterator: Iterator[T]) -> Tuple[T, Iterator[T]]:
    """
    Get the first item out of an iterator to surmise some properties of the rest of the items.
    :param iterator: To preview.
    :return: A tuple:
    (the first item in the iterator, the FULL iterator,
    note that the first item is added back onto this output iterator )
    """

    try:
        first_item = next(iterator)
    except StopIteration:
        LOGGER.error("Iterator source was empty, nothing to preview.")
        raise

    return first_item, itertools.chain([first_item], iterator)


def items_per_second(source: Iterator[T], queue_size: int = 60) -> Iterator[T]:
    """
    Prints logging around how often items are being extracted.
    :param source: To forward.
    :param queue_size: Average is computed across this many items.
    :return: The input iterator.
    """

    queue_count = itertools.count()
    item_queue: Deque[datetime.datetime] = collections.deque(maxlen=queue_size)

    def yield_item(item: T) -> T:
        """
        Return the input item, printing speed logging along the way.
        :param item: To forward.
        :return: The input item, unmodified.
        """
        item_queue.append(datetime.datetime.now())
        if next(queue_count) >= queue_size:
            LOGGER.info(
                f"The last {queue_size} items were consumed at a rate of: "
                f"{queue_size / ((item_queue[-1] - item_queue[0]).total_seconds())} items per "
                "second."
            )

        # Don't do anything to the input item.
        return item

    return map(yield_item, source)


def preload_into_memory(source: Iterator[T], buffer_size: int) -> Iterator[T]:
    """
    Iterators that involve reading from disk can be slow. To make sure that consumers always
    have a nice chunk of items to consume from RAM, this function creates a worker thread that
    reads items from source and puts them in a Queue. When the output is iterated on, it is
    getting items from that queue, not from the input iterator.
    :param source: Input iterator.
    :param buffer_size: Size of the output queue. If the objects in `source` are big, this is going
    to decide how much RAM is allocated for this process.
    :return: An iterator of the buffered items from `source`.
    """

    item_queue: "Queue[T | object]" = Queue(maxsize=buffer_size)
    sentinel = object()  # Unique object to signal the end of the iterator

    def worker() -> None:
        """
        Read items from `source`, putting it into the Queue.
        `.put` operations block if the queue is full, which is used to make sure it is topped up.
        :return: None
        """

        for input_item in source:
            item_queue.put(input_item)

        item_queue.put(sentinel)

    # Start the background thread to read from the input and fill the queue.
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # Iterator that consumes from the queue
    while True:
        output_item = item_queue.get()
        if output_item is sentinel:
            break
        yield output_item  # type: ignore
