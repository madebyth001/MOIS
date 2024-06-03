import wget
from typing import Literal
from pathlib import Path

from ultralytics import YOLO

from cw.dirs import DIRS


class PretrainedModelProvider:
    _DEFAULT_VERSION = '8.2.0'
    _DEFAULT_MODEL = 's'

    def get_pretrained(
            self,
            type_: Literal['n', 's', 'm', 'l', 'x'] = _DEFAULT_MODEL,
            version: str = _DEFAULT_VERSION,
    ):
        path = self._get_pretrained_path(type_, version)
        return YOLO(path)

    def _get_pretrained_path(
            self,
            type_: Literal['n', 's', 'm', 'l', 'x'] = _DEFAULT_MODEL,
            version: str = _DEFAULT_VERSION,
    ):
        link = self._create_pretrained_model_link(type_, version)
        destination = Path(DIRS.user_data_dir)
        filename = link.split('/')[-1]
        download_path = destination / filename
        if not download_path.exists():
            wget.download(link, str(download_path))
        return download_path

    @staticmethod
    def _create_pretrained_model_link(
            type_: Literal['n', 's', 'm', 'l', 'x'] = _DEFAULT_MODEL,
            version: str = _DEFAULT_VERSION,
    ):
        return f'https://github.com/ultralytics/assets/releases/download/v{version}/yolov8{type_}.pt'


if __name__ == '__main__':
    model = PretrainedModelProvider().get_pretrained()
    results = model(['22tr.jpeg'])
    for result in results:
        boxes = result.boxes
        masks = result.masks
        keypoints = result.keypoints
        probs = result.probs
        obb = result.obb
        result.show()
        result.save(filename="result.jpg")
