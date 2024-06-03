from pathlib import Path

from ultralytics import YOLO

from cw.pretrained_model_provider import PretrainedModelProvider

_TUNED_MODEL_PATH = Path('tuned_model.pt')


def train(n):
    data_path = '/Users/madebyth/PycharmProjects/cw2-master/dataset/dataset.yml'

    model = PretrainedModelProvider().get_pretrained()
    model.train(data=data_path, epochs=n, batch=4)


def val(img: Path | str):
    model = YOLO('/Users/madebyth/PycharmProjects/cw2-master/cw/runs/detect/train2/weights/best.pt')
    results = model.predict(img)
    for result in results:
        result.show()


if __name__ == '__main__':
    train(10)
