from __future__ import annotations

from pathlib import Path

from common import collect_sparse_summary, ensure_metashape, get_or_create_chunk, load_cfg, load_request_json, save_summary_json

ensure_metashape()
import Metashape  # type: ignore


def main() -> None:
    request = load_request_json()
    project_path = Path(request['project_path']).expanduser()
    new_image_paths = [str(Path(p).expanduser()) for p in request['new_image_paths']]
    output_json = Path(request['output_json']).expanduser()
    cfg = load_cfg(request.get('config', {}))

    doc = Metashape.Document(); doc.open(str(project_path))
    chunk = get_or_create_chunk(doc, cfg.get('chunk_label', 'sparse_nbv_chunk'))
    existing = {camera.photo.path for camera in chunk.cameras if getattr(camera, 'photo', None)}
    fresh = [p for p in new_image_paths if p not in existing]
    if fresh:
        chunk.addPhotos(fresh)
    align_cfg = cfg.get('align', {})
    chunk.matchPhotos(
        downscale=int(align_cfg.get('downscale', 1)),
        generic_preselection=bool(align_cfg.get('generic_preselection', True)),
        reference_preselection=bool(align_cfg.get('reference_preselection', False)),
        keypoint_limit=int(align_cfg.get('keypoint_limit', 40000)),
        tiepoint_limit=int(align_cfg.get('tiepoint_limit', 8000)),
    )
    chunk.alignCameras(reset_alignment=bool(align_cfg.get('reset_alignment', False)))
    doc.save(str(project_path))
    save_summary_json(output_json, collect_sparse_summary(chunk, cfg))


if __name__ == '__main__':
    main()
