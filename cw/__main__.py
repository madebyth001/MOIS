from pathlib import Path

from ultralytics import YOLO

from cw.pretrained_model_provider import PretrainedModelProvider

_TUNED_MODEL_PATH = Path('tuned_model.pt')


def train():
    data_path = '/Users/madebyth/PycharmProjects/cw2-master/dataset/dataset.yml'

    model = PretrainedModelProvider().get_pretrained()
    model.train(data=data_path, epochs=10, batch=4)


def val():
    model = YOLO('/Users/madebyth/PycharmProjects/cw2-master/cw/runs/detect/train2/weights/best.pt')
    img = Path('/Users/madebyth/Desktop/test1.jpeg')
    results = model.predict(img)
    for result in results:

        print(result.path)
        d = result.plot()
        result.show()


if __name__ == '__main__':
    val()
