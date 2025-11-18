from __future__ import annotations

import pandas as pd

from .method_metadata import MethodMetadata


class MethodMetadataCollection:
    def __init__(self, method_metadata_lst: list[MethodMetadata]):
        self.method_metadata_lst = method_metadata_lst

    def get_method_metadata(
        self,
        method: str,
        artifact_name: str | None = None,
        s3_bucket: str | None = None,
        s3_prefix: str | None = None,
    ) -> MethodMetadata:
        """
        Return the unique MethodMetadata that matches the provided identifiers.

        The full unique key is (method, artifact_name, s3_bucket, s3_prefix).
        This function accepts a *partial* key: it filters using only the
        provided (non-None) fields. If that partial key matches exactly one
        item, that item is returned. If it matches zero or multiple items,
        an informative exception is raised. In the multiple-match case, a
        pandas DataFrame of indistinguishable candidates is included.

        Parameters
        ----------
        method
            Method name to match (required).
        artifact_name
            Optional artifact name to further constrain the search.
        s3_bucket
            Optional S3 bucket to further constrain the search.
        s3_prefix
            Optional S3 prefix to further constrain the search.

        Returns
        -------
        MethodMetadata
            The single MethodMetadata uniquely identified by the provided fields.

        Raises
        ------
        LookupError
            If zero items match the provided filters.
        ValueError
            If multiple items match (i.e., the provided filters are insufficient
            to uniquely identify a single MethodMetadata). The error message
            includes a DataFrame of candidate rows that cannot be distinguished.
        """
        if not self.method_metadata_lst:
            raise LookupError("No MethodMetadata objects are available in the collection.")

        # 1) Fast pre-filter by method (cheap, avoids converting the whole list).
        by_method = [m for m in self.method_metadata_lst if m.method == method]

        if not by_method:
            raise LookupError(
                f"No MethodMetadata entries exist with method='{method}'."
            )

        # 2) Apply only the provided (non-None) fields.
        def ok(m: MethodMetadata) -> bool:
            if artifact_name is not None and m.artifact_name != artifact_name:
                return False
            if s3_bucket is not None and m.s3_bucket != s3_bucket:
                return False
            if s3_prefix is not None and m.s3_prefix != s3_prefix:
                return False
            return True

        candidates = [m for m in by_method if ok(m)]

        # 3) Resolve outcomes without building any DataFrame unless necessary.
        if len(candidates) == 1:
            return candidates[0]

        # Helper: build a tiny DF only for display (lazy import).
        def _candidates_df(objs: list[MethodMetadata]):
            rows = [
                {
                    "method": getattr(m, "method", None),
                    "artifact_name": getattr(m, "artifact_name", None),
                    "s3_bucket": getattr(m, "s3_bucket", None),
                    "s3_prefix": getattr(m, "s3_prefix", None),
                }
                for m in objs
            ]
            # Show unique identifier rows only
            return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)

        if len(candidates) == 0:
            # Nothing matches *with* the extra filters; show what's available for this method.
            df = _candidates_df(by_method)
            raise LookupError(
                "No MethodMetadata matches the provided filters.\n"
                f"Filters used: method={method!r}, artifact_name={artifact_name!r}, "
                f"s3_bucket={s3_bucket!r}, s3_prefix={s3_prefix!r}\n"
                "Available candidates for this method:\n"
                f"{df.to_string(index=False)}"
            )

        # More than one remains → ambiguous; show just those indistinguishable candidates.
        df = _candidates_df(candidates)
        raise ValueError(
            "Provided filters are insufficient to uniquely identify a MethodMetadata.\n"
            f"Filters used: method={method!r}, artifact_name={artifact_name!r}, "
            f"s3_bucket={s3_bucket!r}, s3_prefix={s3_prefix!r}\n"
            "Indistinguishable candidates:\n"
            f"{df.to_string(index=False)}"
        )

    def info(self) -> pd.DataFrame:
        info_lst = []
        for method_metadata in self.method_metadata_lst:
            cur_info = method_metadata.__dict__
            info_lst.append(cur_info)

        info = pd.DataFrame(info_lst)
        return info

    def upload_method_metadata_to_s3(self):
        for method_metadata in self.method_metadata_lst:
            method_uploader = method_metadata.method_uploader()
            method_uploader.upload_metadata()
