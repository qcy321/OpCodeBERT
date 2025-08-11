import logging
import multiprocessing

from tqdm import tqdm
from typing import TypeVar, Generic

T = TypeVar('T')
logger = logging.getLogger(__name__)


class FunctionInf(Generic[T]):

    def __init__(self, func: callable, args: tuple) -> None:
        self.func = func
        self.args = args

    def run(self) -> T:
        return self.func(*self.args)


def process_task(func_inf: FunctionInf):
    """
    Tasks to be performed
    :return: Task return value
    """
    # 任务执行
    try:
        result = func_inf.run()
        return result
    except Exception as e:
        logger.error(e)
        logger.warning("This task has no return value")
        return None


def worker(func_inf: FunctionInf):
    """
    Worker process function, which executes a single task
    """
    result = process_task(func_inf)
    return result


def split_task(data: T, chunk_size: int = 10000) -> list[T]:
    """
    Data partitioning
    :param data: Original data
    :param chunk_size: Partition size
    :return:
    """
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i + chunk_size])
    return chunks


def run(num_processes: int, tasks: list[FunctionInf], task_name: str = "task") -> list[T]:
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = []

        logger.info("-----Multiple processes started successfully-----")

        with tqdm(total=len(tasks), desc=f'Progress - {task_name}', ncols=80) as pbar:
            for result in pool.imap_unordered(worker, tasks):
                results.append(result)
                pbar.update(1)

        logger.info("-----All tasks completed-----")
        return results
