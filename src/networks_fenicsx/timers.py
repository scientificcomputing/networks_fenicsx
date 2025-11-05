# Copyright (C) Simula Research Laboratory and Cécile Daversin-Catty and Jørgen S. Dokken
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""
Timer utilities for profiling
"""

from mpi4py import MPI

from time import perf_counter
from pathlib import Path
from functools import wraps
from typing import Dict, List

from networks_fenicsx import config

__all__ = ["timeit", "timing_dict", "timing_table"]


def timeit(func):
    """
    Decorator that makes a function stores it interpolation time to `"profiling.txt"`.

    The function has to have an attribute `cfg` with the attribute `outdir` or
    `timing_dir` giving the path to the output directory.

    Note:
        In parallel, the average time over all processors is stored.
    
    Args:
        func: function to time
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        time = end - start

        # In parallel, reduce this into average of time on each processors
        sum_time = MPI.COMM_WORLD.reduce(time, op=MPI.SUM, root=0)

        # Write to profiling file
        if MPI.COMM_WORLD.rank == 0:
            avg_time = sum_time / MPI.COMM_WORLD.size
            avg_time_info = (
                f"{func.__name__}: {avg_time:.5e} s \n")
            try:
                p = Path(args[0].cfg.outdir)
            except AttributeError:
                p = Path(args[0].timing_dir)

            p.mkdir(exist_ok=True)
            with (p / "profiling.txt").open("a") as f:
                f.write(avg_time_info)

        return result

    return wrapper


def timing_dict(input_file: config.Config|str|Path)-> dict[str, list[float]]:
    """
    Read `"profiling.txt"` and create a dictionary out of it.

    Args:
       str : Configuration file or path to read from
    
    Returns:
         dict: Dictionary with function names as keys and the list of times as values.
    """
    if hasattr(input_file, "outdir"):
        p = Path(input_file.outdir)
    else:
        p = Path(input_file)
    timing_file = (p / "profiling.txt").open("r")
    timing_dict: Dict[str, List[float]] = dict()

    for line in timing_file:
        split_line = line.strip().split(":")
        keyword = split_line[0]
        value = float(split_line[1].split()[0])

        if keyword in timing_dict.keys():
            timing_dict[keyword].append(value)
        else:
            timing_dict[keyword] = [value]
    timing_file.close()
    return timing_dict