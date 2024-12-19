# pylint: disable=E1101

import queue
import time
from threading import Thread
from typing import List

import imageio.v2 as imageio
import torch
import numpy as np


def read_image(image_path: str, num_channels: int = 3) -> torch.Tensor:
    """
    Reading an image given its absolute path.
    """
    image = torch.from_numpy(imageio.imread(image_path)).to(torch.uint8)
    image = (image / 255.0).clamp(0.0, 1.0)

    if num_channels == 4:
        background = np.array([0, 0, 0])
        image = image.numpy()
        image = image[:, :, :3] * image[:, :, 3:4] + \
            background * (1 - image[:, :, 3:4])
        image = torch.from_numpy(image).float()

    return image


class TaskQueue(queue.Queue):
    """
    A simple queue for executing tasks in multi-threading.
    """

    def __init__(
        self,
        max_size: int = 100,
        max_num_threads: int = 8
    ):
        queue.Queue.__init__(self)

        self.max_size = max_size
        self.max_num_threads = max_num_threads
        self.threads = []

        self.start_workers()

    def empty_threads(self):
        """
        Enforcing release all resources occupied by the threads.
        """
        for _ in range(self.max_num_threads):
            self.put((None, None, None))

        for t in self.threads:
            t.join()

        self.threads = None

    def worker(self):
        while True:
            item, args, kwargs = self.get()
            if item is None:
                break

            item(*args, **kwargs)
            self.task_done()

    def start_workers(self):
        """
        Add threads.
        """
        for _ in range(self.max_num_threads):
            t = Thread(target=self.worker)
            # t.daemon = True
            t.start()

            self.threads.append(t)

    def add_task(self, task, *args, **kwargs):
        """
        Add a task to the task queue.
        """
        args = args or ()
        kwargs = kwargs or {}
        self.put((task, args, kwargs))


class ImageReader(TaskQueue):
    """
    Reading images into a FIFO queue with multi-threading.
    """

    def __init__(
        self,
        max_size: int = 100,
        max_num_threads: int = 8,
        num_channels: int = 3,
        image_list: List[str] = None  # pylint: disable=W0621
    ):
        super().__init__(max_size, max_num_threads)

        self.image_list = image_list
        self.num_channels = num_channels
        self.image_queue = queue.Queue(maxsize=max_size)

    def add_task(self, task, *args, **kwargs):
        for i, image_path in enumerate(self.image_list):
            super().add_task(read_image, image_path, self.num_channels, index=i)

    def worker(self):
        while True:
            item, args, kwargs = self.get()
            if item is None:
                break

            index = kwargs["index"]  # pylint: disable=W0621
            image = item(*args)  # pylint: disable=W0621
            self.image_queue.put((index, image))

            self.task_done()

    def get_image(self):
        """
        Get an image from the queue.
        """
        index, image = self.image_queue.get()  # pylint: disable=W0621
        self.image_queue.task_done()

        return index, image

    def num_images(self):
        """
        The number of images remained in the queue.
        """
        return self.image_queue.qsize()

    def safe_exit(self):
        """
        Clear all unfinished tasks and safely terminate all threads.
        """

        num_images = self.num_images()
        while num_images > 0:
            time.sleep(0.01)
            self.get_image()  # Do nothing
            num_images = self.num_images()

        self.join()
        self.image_queue.join()

        self.empty_threads()


if __name__ == '__main__':
    image_list = []
    for i in range(1000):
        image_list.append(
            "/home/yuchen/datasets/internal/reception/images_2/img_000001.png")

    image_reader = ImageReader(image_list=image_list)
    image_reader.add_task(None)

    for i in range(1000):
        time.sleep(0.01)
        index, image = image_reader.get_image()
        print(f'{i}: image index: {index}, image shape: {image.shape}, ' +
              f'num images in queue: {image_reader.num_images()}')

    image_reader.safe_exit()
