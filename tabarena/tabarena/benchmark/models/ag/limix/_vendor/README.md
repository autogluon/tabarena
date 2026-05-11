# Vendored LimiX

Upstream: https://github.com/limix-ldm-ai/LimiX (Apache-2.0)

LimiX is not published as a pip-installable package. This directory vendors
the inference-time sources required by `LimiXModel`. Files were copied
verbatim and their top-level `inference.*`, `model.*`, and `utils.*` imports
were rewritten to live under
`tabarena.benchmark.models.ag.limix._vendor`.

Not vendored: the upstream `retrieval_extension/` folder (only used by an
optional hyperparameter search loop, not by the predict path) and the
top-level `inference_classifier.py` / `inference_regression.py` example
scripts.

See `LICENSE.txt` for the upstream Apache-2.0 license.
