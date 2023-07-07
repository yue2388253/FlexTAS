import time


def timing_decorator(log_func):
    def real_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            log_func(f"Execution time of {func.__name__}(): {end_time - start_time} seconds")
            return result
        return wrapper
    return real_decorator
