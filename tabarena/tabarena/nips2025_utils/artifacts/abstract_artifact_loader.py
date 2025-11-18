from __future__ import annotations


class AbstractArtifactLoader:
    def download_raw(self):
        raise NotImplementedError

    def download_processed(self):
        raise NotImplementedError

    def download_results(self):
        raise NotImplementedError
