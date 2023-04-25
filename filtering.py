import numpy as np


def shift_by_one_element(array: np.ndarray, fill_value=-1) -> np.ndarray:
    result = np.empty_like(array)
    result[:1] = fill_value
    result[1:] = array[:-1]
    return result


def get_most_frequent_value_in_queue(queue: np.ndarray) -> np.int64:
    freq = np.bincount(queue)
    return np.argmax(freq)


def push(queue: np.ndarray, new_value: int) -> np.ndarray:
    if len(queue) < 10:
        queue = np.append(queue, new_value)
    else:
        queue = shift_by_one_element(queue)
        queue[0] = new_value
    return queue
