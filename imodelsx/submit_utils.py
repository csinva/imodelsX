from typing import Any, Dict, List, Optional, Tuple, Union
import itertools
import subprocess
import random
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool, current_process, Queue
from functools import reduce
from itertools import repeat
import time
import traceback
import numpy as np
import os
from os.path import dirname, join
import yaml

from dict_hash import sha256
submit_utils_dir = dirname(__file__)
"""Handles utilities for job sweeps,
focused on embarassingly parallel sweeps on a single machine.
"""


def run_args_list(
    args_list: List[Dict[str, Any]],
    cmd_python: str = 'python',
    script_name: str = '02_train_suffix.py',
    actually_run: bool = True,
    debug_mode: bool = False,
    shuffle: bool = False,
    reverse: bool = False,
    unique_seeds: str = None,
    n_cpus: int = 1,
    gpu_ids: Union[List[int], List[List[int]]] = [],
    repeat_failed_jobs: bool = False,
    slurm: bool = False,
    slurm_kwargs: Optional[Dict] = None,
    amlt_kwargs: Optional[Dict] = None,
):
    """
    Params
    ------
    run_args_list
    cmd_python: str
        Command to run python
    script_name: str
        Name of script to run
    actually_run: bool
        Whether to actually run the script (otherwise just print the command)
    debug_mode: bool
        Whether to open debugger after failure (stops all parallelilization) 
    shuffle: bool
        Whether to shuffle the order of the script calls
    reverse: bool
        Whether to reverse the order of the script calls
    unique_seeds: str
        Whether to assign random, unique values to each parameter with this value
    n_cpus: int
        Number of cpus to use (if >1, parallelizes over local machine)
    gpu_ids: List[int], List[List[int]]
        Ids of GPUs to run on (e.g. [0, 1] for 2 gpus)
        If List[List[int]], then each inner list is a group of GPUs to run on, e.g. [[0, 1], [2, 3]] for 2 groups of 2 GPUs
    repeat_failed_jobs: bool
        Whether to repeatedly run failed jobs
    run_slurm: bool
        Whether to run on SLURM (defaults to False)
    slurm_kwargs: Optional[Dict]
        kwargs for slurm
    amlt_kwargs: Optional[Dict]
        kwargs for amlt (will override everything else)
    """
    if amlt_kwargs is not None:
        print('Running on AMLT with', amlt_kwargs)
    else:
        n_gpus = len(gpu_ids)
        _validate_run_arguments(n_cpus, gpu_ids)

    # adjust order
    if shuffle:
        random.shuffle(args_list)
    if reverse:
        args_list = args_list[::-1]

    # debug mode
    if debug_mode:
        cmd_python = 'python -m pdb -c continue'
        if n_cpus > 1 or n_gpus > 1:
            print('\n###\n### Debug mode, setting n_cpus=1 and n_gpus=0 ###\n###\n')
            n_cpus = 1
            n_gpus = 0

    # assign unique seeds
    if unique_seeds:
        for i, args in enumerate(args_list):
            args_list[i]['seed_stories'] = random.randint(1, int(1e6))

    # construct commands
    param_str_list = [_param_str_from_args(
        args, cmd_python, script_name) for args in args_list]

    # just print and exit
    if not actually_run:
        print('Not actually running the commands, just printing them.')
        for i, param_str in enumerate(param_str_list):
            print(
                f'\n\n-------------------{i + 1}/{len(param_str_list)}--------------------\n' + param_str)
        return

    failed_jobs = []

    if slurm:
        for i, param_str in enumerate(param_str_list):
            print(
                f'\n\n-------------------{i + 1}/{len(param_str_list)}--------------------\n' + param_str)
            run_slurm(param_str, slurm_kwargs=slurm_kwargs)
        return
    elif amlt_kwargs is not None:
        assert 'amlt_file' in amlt_kwargs
        sku = amlt_kwargs.get('sku', 'G1')
        process_count_per_node = amlt_kwargs.get('process_count_per_node', 1)
        amlt_dir = dirname(amlt_kwargs['amlt_file'])
        repo_dir = dirname(amlt_dir)
        script_name = script_name.replace(repo_dir, '').strip('/')
        param_str_list = [_param_str_from_args(
            args, cmd_python, script_name) for args in args_list]
        if 'mnt_rename' in amlt_kwargs:
            param_str_list = [
                param_str.replace(
                    amlt_kwargs['mnt_rename'][0], amlt_kwargs['mnt_rename'][1])
                for param_str in param_str_list
            ]

        # read and update amlt yaml file
        with open(amlt_kwargs['amlt_file'], 'r') as f:
            amlt_yaml = yaml.safe_load(f)

        if 'target___name' in amlt_kwargs:
            amlt_yaml['target']['name'] = amlt_kwargs['target___name']

        uai = ''
        if '_AZUREML_SINGULARITY_JOB_UAI' in amlt_kwargs:
            uai = amlt_kwargs['_AZUREML_SINGULARITY_JOB_UAI']
        elif '_AZUREML_SINGULARITY_JOB_UAI' in os.environ:
            uai = os.environ['_AZUREML_SINGULARITY_JOB_UAI']
        else:
            uai = '/subscriptions/2cd190bb-b42a-477c-b1bb-2f20932d8dc5/resourceGroups/chansingh/providers/Microsoft.ManagedIdentity/userAssignedIdentities/chansinghid'


        jobs = []
        for i, param_str in enumerate(param_str_list):
            jobs.append({
                'name': f'{sku}_job_{i}',
                'process_count_per_node': process_count_per_node,
                'sku': sku,
                'command': [f'echo "{param_str}"', param_str],
                'identity': 'managed',
                'submit_args': {'env': {'_AZUREML_SINGULARITY_JOB_UAI': uai}},
            })
        amlt_yaml['jobs'] = jobs
        amlt_text = yaml.dump(amlt_yaml, default_flow_style=False)
        amlt_text = amlt_text.replace('$CONFIG_DIR', '$CONFIG_DIR/..')


        # save yaml file in logs dir and run with amlt
        logs_dir = join(amlt_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        out_file = join(logs_dir, sha256({'s': str(param_str_list)}) + '.yaml')
        with open(out_file, 'w') as f:
            f.write(amlt_text)
        subprocess.run(
            f'amlt run {out_file}', shell=True, check=True,
        )
        return

    # run serial
    elif n_cpus == 1 and n_gpus == 0:
        for i, param_str in enumerate(param_str_list):
            print(
                f'\n\n-------------------{i + 1}/{len(param_str_list)}--------------------\n' + param_str)
            try:
                output = subprocess.run(
                    param_str, shell=True, check=True,
                )
            except KeyboardInterrupt:
                print('Keyboard interrupt, exiting...')
                exit(0)
            except subprocess.CalledProcessError as e:
                print('CalledProcessError', e)
                failed_jobs.append((i, param_str))
            except Exception as e:
                print(e)

    # run parallel on CPUs
    elif n_cpus > 1 and n_gpus == 0:
        def run_single_job(i, param_str):
            print(
                f'\n\n-------------------{i + 1}/{len(param_str_list)}--------------------\n' + param_str)
            try:
                output = subprocess.run(
                    param_str, shell=True, check=True,
                )
            except subprocess.CalledProcessError as e:
                print('CalledProcessError', e)
                failed_jobs.append((i, param_str))
            except KeyboardInterrupt:
                print('Keyboard interrupt, exiting...')
                exit(0)
            except Exception as e:
                print(e)
        pool = ThreadPool(n_cpus)
        for i, param_str in enumerate(param_str_list):
            pool.apply_async(run_single_job, (i, param_str, ))
        pool.close()
        pool.join()

    # run parallel on GPUs
    elif n_gpus > 0:
        # initialize the queue with the GPU ids
        global job_queue_multiprocessing
        job_queue_multiprocessing = Queue()
        for gpu_id in gpu_ids:
            job_queue_multiprocessing.put(gpu_id)

        # call the jobs
        pool = Pool(processes=n_gpus)
        n = len(param_str_list)
        indexes = [i for i in range(n)]
        args = zip(param_str_list, indexes, repeat(n))
        # time.sleep(0.1)
        for failed_job in pool.starmap(run_on_gpu, args, chunksize=1):
            failed_jobs.append(failed_job)
        failed_jobs = [x for x in failed_jobs if x is not None]
        pool.close()
        pool.join()
        print('failed_jobs', failed_jobs)

    # final printing
    print('\n\n\n*********************Done*********************')
    if len(failed_jobs) == 0:
        print('All jobs succeeded!')
    else:
        print(len(failed_jobs), 'Failed jobs\n\n')
        for (i, param_str) in failed_jobs:
            print('\t', param_str)
            # print('\t', repr(e))
        failed_args_list = [args_list[i] for (i, _) in failed_jobs]

        if repeat_failed_jobs:
            print('Repeating failed jobs...')
            run_args_list(
                failed_args_list,
                cmd_python=cmd_python,
                script_name=script_name,
                actually_run=actually_run,
                shuffle=shuffle,
                reverse=reverse,
                n_cpus=n_cpus,
                gpu_ids=gpu_ids,
                repeat_failed_jobs=repeat_failed_jobs,
            )


def run_slurm(param_str, slurm_kwargs):
    from slurmpy import Slurm
    slurm = Slurm(
        f"imodelsx_job_{time.time()}",
        slurm_kwargs=slurm_kwargs,
        slurm_flags=["requeue"],
    )
    slurm.run(
        f"""
        {param_str}
        """
    )


def _param_str_from_args(args, cmd_python, script_name):
    param_str = cmd_python + ' ' + script_name + ' '
    for k, v in args.items():
        if isinstance(v, list):
            param_str += '--' + k + ' ' + ' '.join(v) + ' '
        elif v is None:
            # skip: None means don't include this argument
            pass
        else:
            param_str += '--' + k + ' ' + str(v) + ' '
    return param_str


def run_on_gpu(param_str, i, n):
    gpu_id = job_queue_multiprocessing.get()
    failed_job = None
    try:
        # run on GPU <gpu_id>
        ident = current_process().ident
        print(f'{ident}: Starting process on GPU(s) {gpu_id}')
        if isinstance(gpu_id, list):
            gpu_str = ','.join([str(x) for x in gpu_id])
        else:
            gpu_str = str(gpu_id)
        prefix = f'CUDA_VISIBLE_DEVICES={gpu_str} '
        param_str = prefix + param_str
        print(
            f'\n\n-------------------{i + 1}/{n}--------------------\n' + param_str)
        subprocess.run(
            param_str, check=True, shell=True
        )
    except KeyboardInterrupt:
        print('Keyboard interrupt, exiting...')
        exit(0)
    except subprocess.CalledProcessError as e:
        print('CalledProcessError', e)
        print(f'{ident}: Finished on GPU(s) {gpu_id}')
        failed_job = (i, param_str)
    finally:
        job_queue_multiprocessing.put(gpu_id)
        return failed_job


def get_args_list(
    params_shared_dict: Dict[str, List],
    params_coupled_dict: Dict[Tuple[str], List[Tuple]] = {},
) -> List[Dict[str, Any]]:
    _validate_arguments(params_shared_dict, params_coupled_dict)

    def combos_collapse(l: List[List[Dict]]) -> List[Dict]:
        # get param combos as List[Tuple[Dict]] then convert to List[Dict]
        return [
            # convert List[Dict[Tuple]] -> List[Dict]
            reduce(lambda a, b: {**a, **b}, dict_tup)
            # get param combos as List[Tuple[Dict]]
            for dict_tup in list(itertools.product(*l))
        ]

    # Shared params as List[List[Dict]]
    shared_combos_dict_list = combos_collapse(
        [[{k: v} for v in params_shared_dict[k]]
         for k in params_shared_dict.keys()]
    )

    # Coupled params as List[List[Dict]]]
    coupled_combos_dict_list = [[
        {k_tup[x]: v[i][x] for x in range(len(k_tup))}
        for i in range(len(v))]
        for k_tup, v in params_coupled_dict.items()
    ]
    if coupled_combos_dict_list == []:
        return shared_combos_dict_list

    # Combine each coupled List[Dict] with the shared List[Dict]
    combined_combos_dict_list = [
        combos_collapse(
            [coupled_combos_dict_list[i], shared_combos_dict_list])
        for i in range(len(coupled_combos_dict_list))
    ]
    args_list = sum(combined_combos_dict_list, [])
    return args_list


def _validate_arguments(
    params_shared_dict: Dict[str, List],
    params_coupled_dict: Dict[Tuple[str], List[Tuple]],
):
    for k, v in params_shared_dict.items():
        if isinstance(v, range):
            v = list(v)
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        assert isinstance(
            k, str), f"params_shared_dict key {k} must be type list, got type {type(k)}"
        assert isinstance(
            v, list), f"params_shared_dict val {v} must be type list, got type {type(v)}"
    for k_tup, v_tup_list in params_coupled_dict.items():
        assert isinstance(
            k_tup, tuple), f"params_coupled_dict key {k_tup} must be type tuple, got type {type(k_tup)}"
        assert isinstance(
            v_tup_list, list), f"params_coupled_dict val {v_tup_list} must be type list, got type {type(v_tup_list)}"
        assert all([isinstance(x, str) for x in k_tup]
                   ), f"params_coupled_dict k {k_tup} must only contain strings"
        assert [len(
            k_tup) == x for x in v_tup_list], f"params_coupled_dict k and v must have same length but got {len(k_tup)} and {len(v_tup_list)} for {k_tup} and {v_tup_list} respectively"
        for k in k_tup:
            assert not k in params_shared_dict, f"params_coupled_dict key {k} should not be in params_shared_dict"
        for v_tup in v_tup_list:
            assert len(k_tup) == len(
                v_tup), f"params_coupled_dict k and v must have same length but got {len(k_tup)} and {len(v_tup)} for {k_tup} and {v_tup} respectively"


def _validate_run_arguments(
    n_cpus: int,
    gpu_ids: List[int],
):
    assert n_cpus > 0, f"n_cpus must be greater than 0, got {n_cpus}"
    assert not (n_cpus > 1 and len(gpu_ids) >
                0), 'Cannot parallelize over cpus and gpus'
    if len(gpu_ids) > 0:
        import torch.cuda
        num_gpus = torch.cuda.device_count()
        assert all([isinstance(x, int) for x in gpu_ids]) or all([isinstance(x, list) for x in gpu_ids]
                                                                 ), f'gpu_ids {gpu_ids} must be type int or type list'
        if all([isinstance(x, int) for x in gpu_ids]):
            assert all([x >= 0 and x < num_gpus for x in gpu_ids]
                       ), f'gpu_ids {gpu_ids} must be less than available gpus count {num_gpus}'
        elif all([isinstance(x, list) for x in gpu_ids]):
            gpu_ids_flattened = sum(gpu_ids, [])
            assert all([x >= 0 and x < num_gpus for x in gpu_ids_flattened]
                       ), f'gpu_ids {gpu_ids} must be less than available gpus count {num_gpus}'


if __name__ == '__main__':
    params_shared_dict = {
        'name': ['chandan', 'saloni', 'alice', 'albert', 'jessica', 'felicia', ],
    }

    # Single-tree sweep
    params_coupled_dict = {
        ('dataset_name',): [
            ('llm_tree', )
        ],
    }

    # Args list is a list of dictionaries
    args_list = get_args_list(
        params_shared_dict=params_shared_dict,
        params_coupled_dict=params_coupled_dict,
    )
    run_args_list(
        args_list,
        script_name=join(submit_utils_dir, '../tests/dummy_script.py'),
        actually_run=True,
        # n_cpus=3,
        gpu_ids=[[0, 3]],
        repeat_failed_jobs=True,
    )
