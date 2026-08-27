"""Fixed corpus-level policy for reproducible AMI visual preparation.

Only defects documented by the AMI corpus maintainers belong here. Detector
failures, short clips, and local download corruption must never be allowlisted.
"""
from __future__ import annotations


AMI_DATA_PROBLEMS_URL = "https://groups.inf.ed.ac.uk/ami/corpus/dataproblems.shtml"

# AMI explicitly documents that these camera streams do not exist. Keeping the
# list in code makes the AV-valid subset deterministic rather than dependent on
# whichever zero-byte files happen to be present locally.
OFFICIAL_MISSING_CLOSEUPS = frozenset(
    {
        ("TS3003d", "Closeup1"),
        ("TS3003d", "Closeup2"),
        ("TS3003d", "Closeup3"),
    }
)


def is_official_missing_closeup(meeting_id: str, closeup: str) -> bool:
    return (str(meeting_id), str(closeup)) in OFFICIAL_MISSING_CLOSEUPS
