# -*- coding: utf-8 -*-
# @Date    : 2023-10-24 16:21:46
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"""wrapper on an iterable to allow interruption & auto resume, retrying and multiprocessing"""

from __future__ import annotations
import os
import traceback
import json
import logging
from typing import Any, BinaryIO, Callable, Iterable, Iterator, Literal, TextIO, TypeVar, IO, Sequence
from glob import glob
from itertools import product
from functools import wraps
from multiprocessing import Lock, Process, synchronize

# package info
__name__ = __file__.split("/")[-1].split(".")[0]
__version__ = "0.1.3"
__author__ = "Starreeze"
__license__ = "GPLv3"
__url__ = "https://github.com/starreeze/server-tools"
_logger = logging.getLogger(__name__)

# default file path
output_tmpl = "{name}_p{id}.output"
ckpt_tmpl = "{name}.ckpt"

# type
DataType = TypeVar("DataType")


def check_unfinished(run_name: str):
    "To check if the run is unfinished, according to whether the checkpoint file and the cache file exists"
    ckpt = ckpt_tmpl.format(name=run_name)
    if os.path.exists(ckpt):
        num_cache = len(glob(output_tmpl.format(name=run_name, id="*")))
        num_ckpt = len(open(ckpt, "r").readlines())
        if num_cache == num_ckpt:
            return True
        _logger.warning(
            f"unmatched: {num_cache} unfinished files vs {num_ckpt} checkpoints, restart from the beginning"
        )
    return False


class IterateWrapper:
    def __init__(
        self,
        *data: Any,
        mode: Literal["product", "zip"] = "product",
        restart=False,
        bar=0,
        total_items: int | None = None,
        convert_type=list,
        run_name=__name__,
    ):
        """wrap some iterables to provide automatic resuming on interruption, no retrying and limited to sequence

        Args:
            data: iterables to be wrapped
            mode: how to combine iterables. 'product' means Cartesian product, 'zip' means zip()
            restart: whether to restart from the beginning
            bar: the position of the progress bar. -1 means no bar
            total_items: total items to be iterated
            convert_type: convert the data to this type
            run_name: name of the run to identify the checkpoint and output files
        """
        if len(data) == 1:
            if convert_type is not None:
                self.data = convert_type(data[0])
            else:
                self.data = data[0]
        elif mode == "product":
            self.data = list(product(*data))
        elif mode == "zip":
            self.data = list(zip(*data))
        else:
            raise ValueError("mode must be 'product' or 'zip'")
        total_items = total_items if total_items is not None else len(self.data)
        checkpoint_path = ckpt_tmpl.format(name=run_name)
        if restart:
            os.remove(checkpoint_path)
        try:
            with open(checkpoint_path, "r") as checkpoint_file:
                checkpoint = int(checkpoint_file.read().strip())
        except FileNotFoundError:
            checkpoint = 0
        if bar >= 0:
            from tqdm import tqdm

            self.wrapped_range = tqdm(
                range(checkpoint, total_items),
                initial=checkpoint,
                total=total_items,
                position=bar,
            )
        else:
            self.wrapped_range = range(checkpoint, total_items)
        self.wrapped_iter = iter(self.wrapped_range)
        self.index = 0
        self.checkpoint_path = checkpoint_path

    def __iter__(self):
        return self

    def __next__(self):
        with open(self.checkpoint_path, "w") as checkpoint_file:
            checkpoint_file.write(str(self.index))
        try:
            self.index = next(self.wrapped_iter)
        except StopIteration:
            try:
                os.remove(self.checkpoint_path)
            except FileNotFoundError:
                pass
            raise StopIteration()
        return self.data[self.index]


def retry_dec(retry=5, on_error: Literal["raise", "continue"] = "raise"):
    "decorator for retrying a function on exception; on_error could be raise or continue"

    def decorator(func):
        def wrapper(*args, **kwargs):
            if retry <= 1:
                return func(*args, **kwargs)
            for j in range(retry):
                try:
                    return func(*args, **kwargs)
                except KeyboardInterrupt as e:
                    _logger.error(traceback.format_exc())
                    raise e
                except BaseException as e:
                    if j == retry - 1:
                        if on_error == "raise":
                            _logger.error("All retry failed:")
                            _logger.info(traceback.format_exc())
                            raise e
                        elif on_error == "continue":
                            _logger.warning(
                                f"{type(e).__name__}: {e}, all retry failed. Continue due to on_error policy."
                            )
                            return
                    _logger.warning(f"{type(e).__name__}: {e}, retrying [{j + 1}]...")

        return wrapper

    return decorator


def _load_ckpt(path: str, restart: bool) -> list[int] | None:
    checkpoint = None
    if restart:
        if os.path.exists(path):
            os.remove(path)
    try:
        with open(path, "r") as checkpoint_file:
            checkpoint = list(map(int, checkpoint_file.read().splitlines()))
    except FileNotFoundError:
        pass
    return checkpoint


def _write_ckpt(path: str, checkpoint: int, process_idx: int, lock: synchronize.Lock):
    with lock:
        with open(path, "r") as f:
            checkpoints = f.read().splitlines()
        checkpoints[process_idx] = str(checkpoint)
        with open(path, "w") as f:
            f.write("\n".join(checkpoints))


def _process_job(
    func: Callable[[IO, DataType, dict[str, Any]], None],
    data: Iterator[DataType] | Sequence[DataType],
    output_type: Literal["text", "binary", "none"],
    total_items: int,
    process_idx: int,
    num_workers: int,
    iterator_mode: bool,
    lock: synchronize.Lock,
    run_name,
    checkpoint,
    restart,
    retry,
    on_error,
    bar,
    envs: dict[str, str] | None,
    vars_factory: Callable[[], dict[str, Any]],
):
    if envs is not None:
        for k, v in envs.items():
            os.environ[k] = v
    vars = vars_factory()

    chunk_items = (total_items + num_workers - 1) // num_workers
    start_pos = process_idx * chunk_items
    end_pos = min(start_pos + chunk_items, total_items)

    range_to_process = range(checkpoint, end_pos)
    range_checkpointed = range(start_pos, checkpoint)
    if bar:
        from tqdm import tqdm

        range_to_process = tqdm(
            range_to_process,
            initial=checkpoint - start_pos,
            total=end_pos - start_pos,
            position=process_idx,
            desc=f"proc {process_idx}",
        )

    if iterator_mode:
        _logger.info(f"{process_idx}: skipping data until {checkpoint}")
        assert isinstance(data, Iterator)
        for i in range_checkpointed:
            next(data)

    retry_func = retry_dec(retry, on_error)(func)
    open_flags = ("w" if restart else "a") + ("b" if output_type == "binary" else "")
    output = open(output_tmpl.format(name=run_name, id=process_idx), open_flags) if output_type != "none" else None
    _logger.debug(f"{process_idx}: processing data from {checkpoint} to {end_pos}")
    for i in range_to_process:
        _logger.debug(f"{process_idx}: processing data {i}")
        if iterator_mode:
            assert isinstance(data, Iterator)
            try:
                item = next(data)
            except StopIteration:
                break
        else:
            assert isinstance(data, Sequence)
            item = data[i]
        retry_func(output, item, vars)
        _write_ckpt(ckpt_tmpl.format(name=run_name), i + 1, process_idx, lock)
    if output is not None:
        output.close()


def _merge_files(input_paths: list[str], output_stream: IO | None):
    for path in input_paths:
        if output_stream is not None:
            with open(path, "r") as f:
                output_stream.write(f.read())
        if os.path.exists(path):
            os.remove(path)


def iterate_wrapper(
    func: Callable[[IO, DataType, dict[str, Any]], None],
    data: Iterable[DataType],
    output: str | IO | None = None,
    *,
    restart=False,
    retry=5,
    on_error: Literal["raise", "continue"] = "raise",
    num_workers=1,
    bar=True,
    total_items: int | None = None,
    run_name=__name__,
    envs: list[dict[str, str]] = [],
    vars_factory: Callable[[], dict[str, Any]] = lambda: {},
) -> None:
    """Wrapper on a processor (func) and iterable (data) to support multiprocessing, retrying and automatic resuming.

    Args:
        func: The processor function. It should accept three arguments: output stream, data item and vars. Within func, the output stream can be used to save data in real time.
        data: The data to be processed. It can be an iterable or a sequence.
        output: The output stream. It can be a file path, a file object or None. If None, no output will be written.
        restart: Whether to restart from the beginning.
        retry: The number of retries for processing each data item.
        on_error: The action to take when an error occurs.
        num_workers: The number of workers to use. If set to 1, the processor will be run in the main process.
        bar: Whether to show a progress bar (package tqdm required).
        total_items: The total number of items in data. It is required when data is not a sequence.
        run_name: The name of the run. It is used to construct the checkpoint file path.
        envs: Additional environment variables for each worker. This will be set before spawning new processes.
        vars_factory: A function that returns a dictionary of variables to be passed to func. The factory will be called after each process is spawned and before entering the loop. For plain vars, one can simply use closure or functools.partial to pass into func.
    """
    # init vars
    if num_workers < 1 or len(envs) and len(envs) != num_workers:
        raise ValueError("num_workers must be a positive integer and envs must be a list of length num_workers")
    if isinstance(data, Sequence):
        iterator_mode = False
        if total_items is not None:
            assert total_items == len(data)
        else:
            total_items = len(data)
    else:
        iterator_mode = True
        data = iter(data)
        assert total_items is not None, "total_items must be provided when data is not a sequence"

    if num_workers > total_items:
        _logger.warning(
            f"num_workers {num_workers} is greater than total_items {total_items}, "
            "setting num_workers to total_items."
        )
        num_workers = total_items

    # load checkpoint
    checkpoint_path = ckpt_tmpl.format(name=run_name)
    checkpoint = _load_ckpt(checkpoint_path, restart)
    if checkpoint is None:
        checkpoint = [0] * num_workers
        with open(checkpoint_path, "w") as f:
            f.write("\n".join(map(str, checkpoint)))
    elif len(checkpoint) != num_workers:
        raise ValueError(f"checkpoint length {len(checkpoint)} does not match num_workers {num_workers}!")

    # get multiprocessing results
    lock = Lock()
    if isinstance(output, str) or isinstance(output, TextIO):
        output_type = "text"
    elif isinstance(output, BinaryIO):
        output_type = "binary"
    else:
        if output is not None:
            raise ValueError("output must be a string, a file-like object or None")
        output_type = "none"

    def _get_job_args(i: int):
        return (
            func,
            data,
            output_type,
            total_items,
            i,
            num_workers,
            iterator_mode,
            lock,
            run_name,
            checkpoint[i],
            restart,
            retry,
            on_error,
            bar,
            os.environ | (envs[i] if len(envs) else {}),
            vars_factory,
        )

    if num_workers > 1:
        pool = [Process(target=_process_job, args=_get_job_args(i)) for i in range(num_workers)]
        for p in pool:
            p.start()
        for p in pool:
            p.join()
            p.close()
    else:
        _process_job(*_get_job_args(0))

    # merge multiple files
    if isinstance(output, str):
        os.makedirs(os.path.dirname(output), exist_ok=True)
        f = open(output, "w" if restart else "a")
    elif isinstance(output, IO):
        f = output
    else:
        f = None
    _merge_files([output_tmpl.format(name=run_name, id=i) for i in range(num_workers)], f)

    # remove checkpoint file on complete
    try:
        os.remove(checkpoint_path)
    except FileNotFoundError:
        pass


def bind_cache_json(file_path_factory: Callable[[], str]):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            file_path = file_path_factory()
            if not os.path.exists(file_path):
                result = func(*args, **kwargs)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w") as f:
                    json.dump(result, f)
                return result
            else:
                _logger.info(f"Loading cached file {file_path} from disk...")
                with open(file_path, "r") as f:
                    return json.load(f)

        return wrapper

    return decorator


# testing
def _perform_operation(_, item: int, vars):
    from time import sleep

    global _tmp
    print(vars)
    sleep(1)
    if item == 0:
        if _tmp:
            _tmp = False
            raise ValueError("here")


def _test_fn():
    data = list(range(10))
    l = [0, 1]  # noqa: E741
    iterate_wrapper(
        _perform_operation,
        data,
        "output.txt",
        num_workers=3,
        envs=[{"id": str(i)} for i in range(3)],
        vars_factory=lambda: {"a": l + [os.environ["id"]]},
    )


def _test_wrapper():
    from time import sleep

    for i in IterateWrapper(range(10)):
        sleep(1)


if __name__ == "__main__":
    _tmp = True
    _test_fn()
