# Kumo-CellRidge-1024

This entrant uses two fine-tuned Kumo Small checkpoints and four additional row blocks. The classifier and regressor fit an affine ridge readout on at most 512 support rows from the training fold. The classifier uses temperature 0.25 and supports up to ten classes in one native pass. The support draw uses the framework seed. A fixed preprocessing seed of 104 reproduces the single-member development recipe without replaying stored per-dataset transformations.

**Benchmark exposure.** We compared 448 and 1024 on all 51 first outer test splits before choosing the 1024 submission. Earlier decisions also used labeled test examples from 48 of those datasets. Report this exposure when citing the results. The local development panel used sampled rows, whereas Lite evaluates the full outer test partition of each of the 51 datasets.

The default is a single fixed configuration. This evaluation does not tune the heads or fine-tune the weights on a benchmark fold. Run the model from the repository root with:

```sh
python packages/tabarena/src/tabarena/models/kumo_cellridge/run_lite.py
```

The entry point downloads the inference weights on first use and checks their digests. Set `KUMO_CELLRIDGE_WEIGHTS` to a directory containing both release files for offline execution. Batches on GPU are padded to 512 with repetitions of a training row. Padding predictions are discarded. This fixes the matrix shapes across query sizes to limit FP32 rounding differences. CPU uses a single query row per call. Each child stores its own support rows and category vocabulary. Reloading a child reconstructs the network from the verified weight file.

[Weights, development data and licensing](https://github.com/1587causalai/kumo-small-cellridge-1024) · [Lite results](https://github.com/1587causalai/kumo-small-cellridge-1024/blob/main/LITE.md)
