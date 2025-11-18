
<div align="center">

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <img src="https://avatars.githubusercontent.com/u/210855230" width="175" alt="TabArena Logo"/>
    </summary>
  </ul>
</div>

## A Living Benchmark for Machine Learning on Tabular Data 💫

---

| 🚀 [Leaderboard](https://tabarena.ai/) | 📂 [Example Scripts]( https://tabarena.ai/code-examples) | 📊 [Dataset Curation](https://tabarena.ai/data-tabular-ml-iid-study) | 📄 [Paper](https://tabarena.ai/paper-tabular-ml-iid-study) |
|:--------------------------------------:|:----------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|

---
</div>

TabArena is a living benchmarking system that makes benchmarking tabular machine learning models a reliable experience. TabArena implements best practices to ensure methods are represented at their peak potential, including cross-validated ensembles, strong hyperparameter search spaces contributed by the method authors, early stopping, model refitting, parallel bagging, memory usage estimation, and more.

TabArena currently consists of:

- 51 manually curated tabular datasets representing real-world tabular data tasks.
- 9 to 30 evaluated splits per dataset.
- 16 tabular machine learning methods, including 3 tabular foundation models.
- 25,000,000 trained models across the benchmark, with all validation and test predictions cached to enable tuning and post-hoc ensembling analysis.
- A [live TabArena leaderboard](https://huggingface.co/spaces/TabArena/leaderboard) showcasing the results.


## 🕹️ Quickstart Use Cases

We share more details on various use cases of TabArena in our [examples](examples):

* 📊 **Benchmarking Predictive Machine Learning Models**: please refer to [examples/benchmarking](examples/benchmarking).
* 🚀 **Using SOTA Tabular Models Benchmarked by TabArena**: please refer to [examples/running_tabarena_models](examples/running_tabarena_models).
* 🗃️ **Analysing Metadata and Meta-Learning**: please refer to [examples/meta](examples/meta).
* 📈 **Generating Plots and Leaderboards**: please refer to [examples/plots_and_leaderboards](examples/plots_and_leaderboards).
* 🔁 **Reproducibility**: we share instructions for reproducibility in [examples](examples).

### Datasets 
Please refer to our [dataset curation repository](https://github.com/TabArena/tabarena_dataset_curation) to learn more about or contributed data! 

### More Documentation
TabArena code is currently being polished. Detailed Documentation for TabArena will be available soon.

# 🪄 Installation

To install TabArena, ensure you are using Python 3.9-3.12. Then, run the following:

### Install UV

Ensure [UV is installed](https://docs.astral.sh/uv/getting-started/installation/) for the most stable install.

```
pip install uv  # if pip is available
```

### Install AutoGluon

In future AutoGluon installation will occur automatically, but due to changes yet to be released, we need to install AutoGluon from source.

```
git clone https://github.com/autogluon/autogluon.git
./autogluon/full_install.sh
```

### Clone the repository

```
git clone https://github.com/autogluon/tabarena.git
cd tabarena  # ensure the working directory is the project root, otherwise the below commands won't work
```

### Evaluation (Leaderboard / Metrics)

If you don't intend to fit models, this is the simplest installation.

#### UV Install (recommended)

```
uv pip install --prerelease=allow -e ./tabarena
```

#### PIP Install (not recommended)

```
pip install -e ./bencheval
pip install -e ./tabarena
```

### Benchmark (Fitting Models)

If you intend to fit models, this is required.

```
uv pip install --prerelease=allow -e ./tabarena[benchmark]

# use GIT_LFS_SKIP_SMUDGE=1 in front of the command if installing TabDPT fails due to a broken LFS/pip setup
# GIT_LFS_SKIP_SMUDGE=1 uv pip install --prerelease=allow -e ./tabarena/[benchmark]
```

### Developer Install

With this installation, you will have the latest version of AutoGluon in editable form.

```
git clone https://github.com/autogluon/autogluon.git
./autogluon/full_install.sh

git clone https://github.com/autogluon/tabarena.git
uv pip install --prerelease=allow -e ./tabarena[benchmark]
```

Recommended workflow:
Creating a custom virtual environment:
```
pip install uv
uv venv --seed --python 3.11 ~/.venvs/tabarena
source ~/.venvs/tabarena/bin/activate

git clone https://github.com/autogluon/autogluon.git
./autogluon/full_install.sh

git clone https://github.com/autogluon/tabarena.git
uv pip install -U -e tabarena/[benchmark]
```

In PyCharm, make sure to set the directory of `tabarena/` and each `src/` subdirectory of `autogluon/` as 
"Sources Root" for the IDE to find the imports.

### Example Install + Run

Creating a project:
```
pip install uv
uv init -p 3.11
uv sync
git clone https://github.com/autogluon/autogluon.git
./autogluon/full_install.sh
git clone https://github.com/autogluon/tabarena.git
cd tabarena
uv pip install --prerelease=allow -e ./tabarena[benchmark]
cd examples/benchmarking
python run_quickstart_tabarena.py 
```

# Downloading and using TabArena Artifacts

Artifacts will by default be downloaded into `~/.cache/tabarena/`. You can change this by specifying the environment variable `TABARENA_CACHE`.

The types of artifacts are:

1. Raw data -> The original results that are used to derive all other artifacts. Contains per-child test predictions from the bagged models, along with detailed metadata and system information absent from the processed results. Very large, often 100 GB per method type.
2. Processed data -> The minimal information needed for simulating HPO, portfolios, and generating the leaderboard. Often 10 GB per method type.
3. Results -> Pandas DataFrames of the results for each config and HPO setting on each task. Contains information such as test error, validation error, train time, and inference time. Generated from processed data. Used to generate leaderboards. Very small, often under 1 MB per method type.
4. Leaderboards -> Aggregated metrics comparing methods. Contains information such as ELO, win-rate, average rank, and improvability. Generated from a list of results files. Under 1 MB for all methods.
5. Figures & Plots -> Generated from results and leaderboards.

Examples of artifacts include:
* **Raw data**: [examples/meta/inspect_raw_data.py](examples/meta/inspect_raw_data.py)
* **Processed data**: [examples/meta/inspect_processed_data.py](examples/meta/inspect_processed_data.py)
* **Results**: [examples/plots/run_generate_main_leaderboard.py](examples/plots/run_generate_main_leaderboard.py)


# 📄 Publication for TabArena

If you use TabArena in a scientific publication, we would appreciate a reference to the following paper:

**TabArena: A Living Benchmark for Machine Learning on Tabular Data**, 
Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Mutalik Desai, David Salinas, Frank Hutter, Preprint., 2025

Link to publication: [arXiv](https://arxiv.org/abs/2506.16791)

Link to NeurIPS'2025: [Conference Poster and Video](https://neurips.cc/virtual/2025/loc/san-diego/poster/121499)

Bibtex entry:
```bibtex
@article{erickson2025tabarena,
  title={TabArena: A Living Benchmark for Machine Learning on Tabular Data}, 
  author={Nick Erickson and Lennart Purucker and Andrej Tschalzev and David Holzmüller and Prateek Mutalik Desai and David Salinas and Frank Hutter},
  year={2025},
  journal={arXiv preprint arXiv:2506.16791},
  url={https://arxiv.org/abs/2506.16791}, 
}
```


--- 
## Relation to TabRepo 

TabArena was built upon and now replaces [TabRepo](https://arxiv.org/pdf/2311.02971). To see details about TabRepo, the portfolio simulation repository, refer to [tabrepo.md](tabrepo.md).
