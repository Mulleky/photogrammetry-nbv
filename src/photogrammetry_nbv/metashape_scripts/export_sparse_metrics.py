from __future__ import annotations

from pathlib import Path

from common import collect_sparse_summary, ensure_metashape, get_or_create_chunk, load_cfg, load_request_json, save_summary_json

ensure_metashape()
import Metashape  # type: ignore


def main() -> None:
    request = load_request_json()
    project_path = Path(request['project_path']).expanduser()
    output_json = Path(request['output_json']).expanduser()
    cfg = load_cfg(request.get('config', {}))

    doc = Metashape.Document(); doc.open(str(project_path))
    chunk = get_or_create_chunk(doc, cfg.get('chunk_label', 'sparse_nbv_chunk'))
    save_summary_json(output_json, collect_sparse_summary(chunk, cfg))


if __name__ == '__main__':
    main()
