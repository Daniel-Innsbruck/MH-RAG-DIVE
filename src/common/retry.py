"""Retry logic with exponential backoff for robust API calls."""
from functools import wraps
import time
import asyncio
import logging

logger = logging.getLogger(__name__)

def retry_with_backoff(exceptions, max_retries: int = 5, backoff_base: float = 2.0):
    if isinstance(exceptions, list):
        exceptions = tuple(exceptions)

    def decorator(func):
        """
        A decorator that retries a function with exponential backoff upon encountering specific exceptions.

        Supports both synchronous and asynchronous functions.

        Args:
            exceptions (type or tuple): The exception class or a tuple of exception classes that trigger a retry.
            max_retries (int): The maximum number of retry attempts. Default is 5.
            backoff_base (float): The base multiplier for the exponential backoff sleep time (base^attempt). Default is 2.0.

        Returns:
            Callable: The decorated function incorporating the retry logic.
        """
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                for attempt in range(max_retries):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        if attempt < max_retries - 1:
                            sleep_time = backoff_base ** attempt
                            logger.warning(f"[Retry {attempt+1}/{max_retries}] {func.__name__} failed: {e}, retrying in {sleep_time}s")
                            await asyncio.sleep(sleep_time)
                        else:
                            raise
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        if attempt < max_retries - 1:
                            sleep_time = backoff_base ** attempt
                            logger.warning(f"[Retry {attempt+1}/{max_retries}] {func.__name__} failed: {e}, retrying in {sleep_time}s")
                            time.sleep(sleep_time)
                        else:
                            raise
            return sync_wrapper
    return decorator