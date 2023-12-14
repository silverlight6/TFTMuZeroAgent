from config import ITEM_POSITIONING_BUFFER_SIZE
from ray.util.queue import Queue


class QueueStorage(object):
    def __init__(self, threshold=ITEM_POSITIONING_BUFFER_SIZE * 0.75, size=ITEM_POSITIONING_BUFFER_SIZE):
        """Queue storage
        Parameters
        ----------
        threshold: int
            if the current size if larger than threshold, the data won't be collected
        size: int
            the size of the queue
        """
        self.threshold = threshold
        self.queue = Queue(maxsize=size)

    def push(self, batch):
        if self.queue.qsize() <= self.threshold:
            self.queue.put(batch)

    def pop(self):
        if self.queue.qsize() > 0:
            return self.queue.get()
        else:
            return None

    def q_size(self):
        return self.queue.qsize()
