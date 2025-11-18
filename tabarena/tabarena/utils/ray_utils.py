from __future__ import annotations

from copy import deepcopy

import ray

DEFAULT_REMOTE_KWARGS = {
    "max_calls": 1,
    "retry_exceptions": True,
    "max_retries": 0,
}


def run_function_as_ray_task(
    *, func: callable, num_cpus: int, num_gpus: int, func_kwargs: dict
):
    """Run a function as a ray task with the given number of cpus and gpus. Blocks until the function is done."""
    remote_func = ray.remote(**DEFAULT_REMOTE_KWARGS)(func)

    return ray.get(
        remote_func.options(num_cpus=num_cpus, num_gpus=num_gpus).remote(**func_kwargs),
    )


def ray_map_list(
    list_to_map: list,
    *,
    func: callable,
    func_element_key_string: str,
    num_workers: int,
    num_cpus_per_worker: int,
    num_gpus_per_worker: int = 0,
    func_kwargs: dict | None = None,
    func_put_kwargs: dict | None = None,
    put_list_elements: bool = False,
    output_handler: callable | None = None,
    track_progress: bool = False,
    tqdm_kwargs: dict | None = None,
    ray_remote_kwargs: dict | None = None,
) -> list:
    """Map a function over a list using ray. Blocks until all functions are done.

    Arguments:
    ----------
    list_to_map: list
        The list to map the function over.
    func: callable
        The function to map over the list.
    func_element_key_string: str
        The key string to use to pass the element as a key-value pair to the function.
        That is, the function will for example use `{func_element_key_string: list_to_map[0]}` as kwargs for `func`.
    num_workers: int
        The number of workers to use.
    num_cpus_per_worker: int
        The number of cpus to use per worker.
    func_kwargs: dict, default=None
        Additional kwargs to pass to the function.
    func_put_kwargs: dict, default=None
        Additional kwargs to pass to the function, where the values are put into the object store.
    put_list_elements: bool, default=False
        If True, put the elements of `list_to_map` into the object store before passing them to the function.
    output_handler: callable, default=None
        If not None, this should be a function that takes the output of `func` as input and uses it.
        For example, this could log the output or save it to a file.
    track_progress: bool, default=False
        Track the progress of working on the list.
    tqdm_kwargs: dict | None, default=None
        Additional kwargs to pass to tqdm if `track_progress` is True.
        For example, the decription for the progress bar: {"desc": "Processing list"}.
    """
    assert num_workers > 0, "Number of workers must be at least 1!"
    remote_kwargs = deepcopy(DEFAULT_REMOTE_KWARGS)
    if ray_remote_kwargs is not None:
        remote_kwargs.update(ray_remote_kwargs)
    if ray_remote_kwargs is None or "max_calls" not in ray_remote_kwargs:
        remote_kwargs["max_calls"] = max(len(list_to_map) // num_workers, 1)
    remote_p = ray.remote(**remote_kwargs)(func)
    remote_p_options = {
        "num_cpus": num_cpus_per_worker,
        "num_gpus": num_gpus_per_worker,
    }

    job_refs = []
    job_refs_map = {}
    job_index = 0

    return_results = []

    if track_progress:
        from tqdm import tqdm

        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        pbar = tqdm(total=len(list_to_map), **tqdm_kwargs)

    # Setup Kwargs
    job_kwargs = {}
    if func_kwargs is not None:
        for key, value in func_kwargs.items():
            job_kwargs[key] = value
    if func_put_kwargs is not None:
        for key, value in func_put_kwargs.items():
            job_kwargs[key] = ray.put(value)

    # Start initial jobs
    for list_element in list_to_map[:num_workers]:
        result_ref = remote_p.options(**remote_p_options).remote(
            **{
                func_element_key_string: ray.put(list_element)
                if put_list_elements
                else list_element
            },
            **job_kwargs,
        )
        job_refs.append(result_ref)
        job_refs_map[result_ref] = job_index
        job_index += 1

    # Worker loop
    unfinished_list = list_to_map[num_workers:]
    unfinished = job_refs
    while unfinished:
        finished, unfinished = ray.wait(unfinished, num_returns=1)
        job_i = job_refs_map[finished[0]]
        job_res = ray.get(finished[0])

        # Handle output (e.g. logging)
        if output_handler is not None:
            output_handler(job_res)
        if track_progress:
            pbar.update(n=1)

        return_results.append((job_i, job_res))

        # Re-schedule workers
        while unfinished_list and (len(unfinished) < num_workers):
            list_element = unfinished_list[0]
            unfinished_list = unfinished_list[1:]
            result_ref = remote_p.options(**remote_p_options).remote(
                **{
                    func_element_key_string: ray.put(list_element)
                    if put_list_elements
                    else list_element
                },
                **job_kwargs,
            )
            unfinished.append(result_ref)
            job_refs_map[result_ref] = job_index
            job_index += 1

    if func_put_kwargs is not None:
        ray.internal.free(object_refs=[job_kwargs[key] for key in func_put_kwargs])
        for key in func_put_kwargs:
            del job_kwargs[key]

    if track_progress:
        pbar.close()

    return [r for _, r in sorted(return_results, key=lambda x: x[0])]


def to_batch_list(lst: list, batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]
