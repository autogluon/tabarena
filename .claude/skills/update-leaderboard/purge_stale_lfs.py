"""HEAD-aware purge of stale git-LFS objects on a HuggingFace Space repo.

Space repos cap git-LFS storage at 1 GB, and every leaderboard data refresh
re-uploads ~85 MB of figure zips, so production pushes start failing with
``Repository storage limit reached (Max: 1 GB)`` after roughly ten refreshes.
This purges the stored LFS objects that the *current local HEAD* does not
reference — freeing the space held by superseded refreshes while keeping every
object the revision about to be pushed still needs (e.g. the
``data_beyondarena/`` zips a TabArena-only refresh doesn't touch).

Properties:

- **Permanent**: deleted objects cannot be recovered; past Space revisions
  lose their binary files. Acceptable here because all artifacts are
  regenerable from the pipeline.
- **History-preserving**: ``rewriteHistory=false`` leaves the commit graph
  untouched, so a pending ``git push`` stays a plain fast-forward.
- **HEAD-aware**: never deletes objects referenced by the local HEAD — unlike
  the Settings-UI "select all + remove" flow, which breaks the live revision.

Run from *within the Space repo checkout*, with the local HEAD at the state
you are about to push (it defines the keep-set via ``git lfs ls-files``):

    python purge_stale_lfs.py                                        # dry run
    python purge_stale_lfs.py --delete                               # purge
    python purge_stale_lfs.py --repo-id TabArena/leaderboard-testing --delete

Requires ``huggingface_hub`` auth (``hf auth login``) with write access.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess

from huggingface_hub import HfApi
from huggingface_hub.utils import get_session, paginate


def _oid(item: dict) -> str | None:
    return item.get("fileOid") or item.get("oid") or item.get("sha")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default="TabArena/leaderboard", help="Hub repo to purge.")
    parser.add_argument("--repo-type", default="space", choices=["space", "model", "dataset"])
    parser.add_argument("--delete", action="store_true", help="Actually delete (default: dry run).")
    args = parser.parse_args()

    # LFS oids referenced by the local HEAD = the keep-set.
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git not found on PATH — run from within the Space repo checkout")
    lfs_ls = subprocess.run([git, "lfs", "ls-files", "-l"], capture_output=True, text=True, check=True).stdout  # noqa: S603
    keep = {line.split()[0] for line in lfs_ls.splitlines() if line.strip()}
    print(f"oids referenced by local HEAD: {len(keep)}")

    # Raw pagination over the lfs-files endpoint: HfApi.list_lfs_files crashes
    # on entries without a "filename" (orphaned objects), which are exactly
    # the kind of entry a purge needs to see.
    api = HfApi()
    url = f"{api.endpoint}/api/{args.repo_type}s/{args.repo_id}/lfs-files"
    headers = api._build_hf_headers()
    items = list(paginate(url, params={}, headers=headers))

    stale = [item for item in items if _oid(item) and _oid(item) not in keep]
    total_mb = sum(item.get("size", 0) for item in items) / 1e6
    stale_mb = sum(item.get("size", 0) for item in stale) / 1e6
    print(
        f"stored on server: {len(items)} objects ({total_mb:.0f} MB) | "
        f"stale: {len(stale)} ({stale_mb:.0f} MB) | keep: {len(items) - len(stale)}"
    )

    if not args.delete:
        print("dry run — pass --delete to purge the stale objects")
        return

    session = get_session()
    shas = [_oid(item) for item in stale]
    for start in range(0, len(shas), 1000):
        batch = shas[start : start + 1000]
        response = session.post(
            f"{url}/batch",
            json={"deletions": {"sha": batch, "rewriteHistory": False}},
            headers=headers,
        )
        response.raise_for_status()
        print(f"deleted batch of {len(batch)}: HTTP {response.status_code}")
    print(f"purged {stale_mb:.0f} MB")


if __name__ == "__main__":
    main()
