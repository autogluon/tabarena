# Notes


## Some thoughts for the future

* Double check reproducibility of the results (write test that we can go from running the same config twice to the same results)
* Run ProxyModel with an alternative to holdout validation? Also not sub-sample to 10k?
* Use mode steps than just 5 budgets steps per dataset?
* Investigate caching of feature selection (long shot)

## Venv information to run benchmarks

```bash
source /work/dlclarge1/purucker-fs_benchmark/venvs/fs_bench_env/bin/activate && cd /work/dlclarge1/purucker-fs_benchmark/code/fsbench/tabarena/tabflow_slurm
```


Sharing workspace with other users (this might take a while to run):

```bash
setfacl -R -m u:X:rwx /work/dlclarge1/purucker-fs_benchmark
ws_share share fs_benchmark X 
```
(https://wiki.archlinux.org/title/Access_Control_Lists#Usage)
