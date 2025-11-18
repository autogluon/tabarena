from __future__ import annotations


class AbstractArtifactUploader:
    def upload_raw(self):
        raise NotImplementedError

    def upload_processed(self):
        raise NotImplementedError

    def upload_results(self):
        raise NotImplementedError
