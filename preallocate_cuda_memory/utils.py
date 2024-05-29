import time
import torch

from warnings import warn

from typing import List


class MemoryController(object):
    """
    Control the memory occupy of a CUDA device.
    """

    def __init__(self, cuda_device: int) -> None:
        """
        Initialize the class.

        Parameters
        ----------
        cuda_device : int
            The index of the CUDA device to occupy.

        Examples
        --------
        >>> mc = MemoryController(0)
        >>> mc.check_mem()
        16160
        """
        self.cuda_device = cuda_device
        self.xs: List[torch.Tensor] = []

    def check_memory(self) -> int:
        """
        Check the memory usage of a CUDA device.

        Returns
        -------
        available : int
            The available memory of the CUDA device. Unit: MB

        Examples
        --------
        >>> mc = MemoryController(0)
        >>> mc.check_mem()
        16160
        """
        return torch.cuda.mem_get_info(self.cuda_device)[0] >> 20

    def occupy_memory(self, block_mem: int) -> bool:
        """
        Occupy GPU memory on the specified CUDA device.

        Parameters
        ----------
        block_mem : int
            The amount of memory to occupy.
            Unit: MB

        Returns
        -------
        success : bool
            Whether the memory is successfully occupied.

        Examples
        --------
        >>> mc = MemoryController(0)
        >>> mc.occupy_memory(1024)
        """
        try:
            self.xs.append(
                torch.FloatTensor(256, 1024, block_mem).cuda(self.cuda_device)
            )
        except Exception:
            available = self.check_memory()
            warn(
                f"Cannot occupy memory on the CUDA device {self.cuda_device}.\n"
                f"Try to allocate {block_mem}MB, "
                f"but only {available}MB is available."
            )
            return False
        return True

    def occupy_all_available_memory(self) -> None:
        """
        Occupies GPU memory on the specified CUDA device.

        Parameters
        ----------
        cuda_device : int
            The index of the CUDA device to occupy memory on.

        Examples
        --------
        >>> from utils import occupy_mem
        >>> occupy_mem(0)
        """
        available = self.check_memory()
        ret = self.occupy_memory(int(available * 0.9))
        assert ret

        available = self.check_memory()

        while available > 2048:
            if not self.occupy_memory(1024):
                break
            available = self.check_memory()
        while available > 100:
            if not self.occupy_memory(100):
                break
            available = self.check_memory()
        self.occupy_memory(int(available * 0.9))

    def try_occupy_memory_in_time(self, block_mem: int, timeout: int) -> bool:
        """
        Try to occupy memory in a specified time.

        Parameters
        ----------
        block_mem : int
            The amount of memory to occupy.
            Unit: MB
        timeout : int
            The maximum time to occupy memory.
            Unit: second

        Returns
        -------
        success : bool
            Whether the memory is successfully occupied.

        Examples
        --------
        >>> mc = MemoryController(0)
        >>> mc.try_occupy_memory_in_time(1024, 10)
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.occupy_memory(block_mem):
                return True
            time.sleep(1)
        return False

    def free_memory(self) -> None:
        """
        Free the occupied memory on the specified CUDA device.

        Examples
        --------
        >>> mc = MemoryController(0)
        >>> mc.free_memory()
        """
        del self.xs
        self.xs = []


def occupy_specific_device_memory(cuda_device: int) -> None:
    mc = MemoryController(cuda_device)
    mc.occupy_all_available_memory()
    mc.free_memory()


def occupy_all_devices_memory() -> None:
    for cuda_device in list(range(torch.cuda.device_count())):
        mc = MemoryController(cuda_device)
        mc.occupy_all_available_memory()
        mc.free_memory()
