from __future__ import annotations

from pathlib import Path

from common import ensure_metashape, get_or_create_chunk, load_cfg, load_request_json

ensure_metashape()
import Metashape  # type: ignore


def main() -> None:
    request = load_request_json()
    project_path = Path(request['project_path']).expanduser()
    output_dir = Path(request['output_dir']).expanduser()
    cfg = load_cfg(request.get('config', {}))
    dense_cfg = cfg.get('dense', {})
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = Metashape.Document(); doc.open(str(project_path))
    chunk = get_or_create_chunk(doc, cfg.get('chunk_label', 'sparse_nbv_chunk'))

    quality_map = {'ultra': Metashape.UltraQuality, 'high': Metashape.HighQuality, 'medium': Metashape.MediumQuality, 'low': Metashape.LowQuality}
    filter_map = {'aggressive': Metashape.AggressiveFiltering, 'moderate': Metashape.ModerateFiltering, 'mild': Metashape.MildFiltering, 'none': Metashape.NoFiltering}
    quality = quality_map.get(str(dense_cfg.get('quality', 'medium')).lower(), Metashape.MediumQuality)
    filter_mode = filter_map.get(str(dense_cfg.get('filter_mode', 'mild')).lower(), Metashape.MildFiltering)

    chunk.buildDepthMaps(downscale=quality, filter_mode=filter_mode)
    chunk.buildPointCloud(point_confidence=bool(dense_cfg.get('build_point_confidence', True)))
    doc.save(str(project_path))
    export_path = output_dir / 'dense_cloud.las'
    chunk.exportPointCloud(str(export_path), source_data=Metashape.PointCloudData)


if __name__ == '__main__':
    main()
