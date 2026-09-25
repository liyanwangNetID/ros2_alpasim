from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Mapping
from .contract import canonical_json

def append_jsonl(path: Path, row: Mapping[str,Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as handle:
        handle.write(canonical_json(row)+"\n"); handle.flush(); os.fsync(handle.fileno())

def load_checkpoint(path: Path) -> dict[str,dict[str,Any]]:
    rows={}
    if not path.is_file(): return rows
    with path.open("r",encoding="utf-8") as handle:
        for line_number,line in enumerate(handle,1):
            if not line.strip(): continue
            row=json.loads(line); anchor=row.get("anchor_id")
            if not isinstance(anchor,str) or not anchor: raise ValueError(f"checkpoint line {line_number}: invalid anchor_id")
            rows[anchor]=row
    return rows
