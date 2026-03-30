#!/usr/bin/env python3
"""Export sparse reconstruction metrics from an existing COLMAP workspace."""
from __future__ import annotations

from pathlib import Path

from common import collect_sparse_summary, load_cfg, load_request_json, save_summary_json


def main() -> None:
    request = load_request_json()
    workspace = Path(request['workspace']).expanduser()
    output_json = Path(request['output_json']).expanduser()
    cfg = load_cfg(request.get('config', {}))

    save_summary_json(output_json, collect_sparse_summary(workspace, cfg))


if __name__ == '__main__':
    main()
