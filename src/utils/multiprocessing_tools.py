import multiprocessing
import time

def save_terminate_process(process: multiprocessing.Process,
                           shutdown_event = None,
                           timeout: float = 2.0) -> None:
    """ First request termination, then force-kill process."""
    if process.is_alive():
        if shutdown_event is not None:  # try first via send shutdown event:
            shutdown_event.set()
            time.sleep(timeout)

        if process.is_alive():  # terminate process (request)
            process.terminate()
            process.join(timeout=timeout)

        if process.is_alive():  # if still alive kill (force)
            process.kill()
            process.join()

    if process.pid is not None:  # then process was started at some point
        process.join()


class RobustEventManager:
    """ Triggers events and safely waits for triggers while preventing deadlocks through timeouts. """
    def __init__(self):
        self.event = multiprocessing.Event()
        self.lock = multiprocessing.Lock()
        self.trigger_count = multiprocessing.Value('i', 0)

    def set(self):
        with self.lock:
            self.trigger_count.value += 1
            self.event.set()

    def is_set(self):
        return self.event.is_set()

    def wait(self, timeout=None):
        initial_count = self.trigger_count.value

        if timeout is None:
            # Wait indefinitely
            while True:
                if self.event.wait(timeout=1):  # Short timeout for checks
                    with self.lock:
                        if self.trigger_count.value > initial_count:
                            return True
        else:
            # Wait with timeout
            while timeout > 0:
                if self.event.wait(timeout=min(1, timeout)):  # Short timeout for checks
                    with self.lock:
                        if self.trigger_count.value > initial_count:
                            return True
                timeout -= 1
                if timeout <= 0:
                    return False
            return False

    def clear(self):
        with self.lock:
            self.event.clear()
            self.trigger_count.value = 0


class SharedString:
    """
    Thread-safe wrapper for shared string storage using multiprocessing.Array.

    Creates an instance object that can be passed between processes and provides
    instance methods for safe read/write operations with automatic lock management.

    Attributes:
        buffer (multiprocessing.Array): Shared character buffer
        lock (multiprocessing.Lock): Synchronization lock
        max_size (int): Maximum buffer capacity
    """

    def __init__(self, size: int, initial_value: str = ""):
        """
        Initialize shared string instance with specified size.

        Parameters:
            size (int): Maximum buffer size in bytes (includes null terminator)
            initial_value (str): Optional initial string value

        Raises:
            ValueError: If initial_value exceeds size limit
            TypeError: If size is not positive integer
        """
        # Validate inputs
        if not isinstance(size, int) or size <= 0:
            raise TypeError(f"size must be positive integer, got {size}")

        if not isinstance(initial_value, str):
            raise TypeError(f"initial_value must be str, got {type(initial_value)}")

        # Check overflow
        encoded_init = initial_value.encode('utf-8')
        if len(encoded_init) >= size:
            raise ValueError(
                f"initial_value too long: {len(encoded_init)} bytes "
                f"exceeds buffer size {size}"
            )

        # Create shared buffer and lock
        self.buffer = multiprocessing.Array('c', size)
        self.lock = multiprocessing.Lock()
        self.max_size = size

        # Write initial value
        if initial_value:
            self.write(initial_value)

    def write(self, value: str) -> None:
        """
        Safely write string to shared buffer with null termination.

        Parameters:
            value (str): String to write

        Raises:
            ValueError: If value exceeds buffer capacity
            TypeError: If value is not string
        """
        if not isinstance(value, str):
            raise TypeError(f"value must be str, got {type(value)}")

        # Encode and validate size
        encoded = value.encode('utf-8')
        if len(encoded) >= self.max_size:
            raise ValueError(
                f"value too long: {len(encoded)} bytes "
                f"exceeds buffer capacity {self.max_size}"
            )

        # Write to buffer with lock
        with self.lock:
            # Clear previous data
            self.buffer[:] = [0] * self.max_size

            # Write encoded string as list of byte integers
            self.buffer[:len(encoded)] = list(encoded)

            # Add null terminator at end of string
            self.buffer[len(encoded)] = 0

    def read(self) -> str:
        """
        Safely read string from shared buffer with null-termination handling.

        Returns:
            str: Decoded string with null bytes stripped

        Raises:
            UnicodeDecodeError: If buffer contains invalid UTF-8
        """
        # Read from buffer with lock
        with self.lock:
            # Convert buffer slice to bytes
            raw_bytes = bytes(self.buffer[:])

            # Strip null bytes and decode
            try:
                decoded = raw_bytes.rstrip(b'\x00').decode('utf-8')
            except UnicodeDecodeError as e:
                raise UnicodeDecodeError(
                    e.encoding,
                    e.object,
                    e.start,
                    e.end,
                    f"Invalid UTF-8 in shared buffer: {e.reason}"
                ) from e

        return decoded

    def get_lock(self) -> multiprocessing.Lock:
        """
        Retrieve the synchronization lock for manual context management.

        Returns:
            multiprocessing.Lock: Lock object
        """
        return self.lock

    def get_size(self) -> int:
        """
        Get maximum buffer capacity.

        Returns:
            int: Max size in bytes
        """
        return self.max_size
