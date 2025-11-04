"""
This file is based on the graphnics project (https://arxiv.org/abs/2212.02916), https://github.com/IngeborgGjerde/fenics-networks - forked on August 2022
Copyright (C) 2022-2023 by Ingeborg Gjerde

You can freely redistribute it and/or modify it under the terms of the GNU General Public License, version 3.0, provided that the above copyright notice is kept intact and that the source code is made available under an open-source license.

Modified by Cécile Daversin-Catty - 2023
"""

from mpi4py import MPI

from time import perf_counter
from pathlib import Path
from functools import wraps
from typing import Dict, List

from networks_fenicsx import config


def timeit(func):
    """
    Prints and saves to 'profiling.txt' the execution time of func
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
                f"{func.__name__}: {avg_time:.5e} s \n"  # sum_time / MPI.COMM_WORLD.size
            )
            p = Path(args[0].cfg.outdir)
            p.mkdir(exist_ok=True)
            with (p / "profiling.txt").open("a") as f:
                f.write(avg_time_info)

        return result

    return wrapper


def timing_dict(config: config.Config):
    """
    Read 'profiling.txt' and create a dictionary out of it
    Args:
       str : outdir path
    """
    p = Path(config.outdir)
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

    return timing_dict


def timing_table(config: config.Config):
    """
    Read 'profiling.txt' and create a table data file
    Args:
       str : outdir path
    """
    t_dict = timing_dict(config)
    try:
        import pandas as pd

        if MPI.COMM_WORLD.rank == 0:
            df = pd.DataFrame(
                {
                    "n": t_dict["n"],
                    "forms": t_dict["compute_forms"],
                    "assembly": t_dict["assemble"],
                    "solve": t_dict["solve"],
                }
            )
            df.to_csv(config.outdir / "timings.txt", sep="\t", index=False)
        
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Pandas is required to create timing tables.")
        # file = open(config.outdir + '/timings.txt', 'w')
        # file.write("No timing information available.")
        # file.close()

    return t_dict
