from dotenv import dotenv_values
import torch
from torchvision.transforms import transforms
import cv2
import numpy as np
import tensorflow as tf

config = dotenv_values(".env")

# TODO add load model to gpu if available


class Rmn:
    def __init__(self):
        self.download_id = config["RMN_ID"]
        self.filename = config["RMN"]
        self.lib = "pytorch"
        self.image_size = (224, 224)
        self._model = None
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        self._model = torch.load(f"{config['DOWNLOAD_PATH']}/{self.filename}")
        self._model.eval()

    def prepare_image_for_prediction(self, image):
        # Get gray image from bgr image.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize image to suit model input size.
        image = cv2.resize(image, (48, 48))
        image = cv2.resize(image, self.image_size)

        # Convert 1 channel image to 3 channel image.
        image = np.dstack([image] * 3)

        # Transform image.
        image = self.transform(image)

        # Add '1' as batch size.
        image = torch.unsqueeze(image, dim=0)

        return image

    def predict_emotion(self, image):
        with torch.no_grad():
            prediction = torch.squeeze(self._model(image), 0)
            prediction = torch.softmax(prediction, 0)
            classification_confidence, current_prediction = torch.max(prediction, dim=0)
            return current_prediction, classification_confidence


class Lhc:
    def __init__(self):
        self.download_id = config["LHC_NET_ID"]
        self.filename = config["LHC_NET"]
        self.lib = "tensorflow"
        self.image_size = (224, 224)
        self._model = None
        self.transform = None
        self._device = None  # Tensorflow by default uses GPU if available.

    def load_model(self):
        self._model = tf.keras.models.load_model(
            f"{config['DOWNLOAD_PATH']}/{self.filename}"
        )

    def prepare_image_for_prediction(self, image):
        # Get gray image from bgr image.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize image to suit model input size.
        image = cv2.resize(image, (48, 48))
        image = cv2.resize(image, self.image_size)

        # Convert 1 channel image to 3 channel image.
        image = tf.convert_to_tensor(np.dstack([image] * 3))

        # Add '1' as batch size.
        image = tf.expand_dims(image, 0)

        return image

    def predict_emotion(self, image):
        prediction = self._model.predict_on_batch(image)
        classification_confidence = 100 * np.max(prediction)
        current_prediction = np.argmax(prediction, axis=1)[0]
        return current_prediction, classification_confidence


class ResNet18:
    def __init__(self):
        self.download_id = config["RESNET18_ID"]
        self.filename = config["RESNET18"]
        self.lib = "pytorch"
        self.image_size = (224, 224)
        self._model = None
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0,), std=(255,)),
            ]
        )
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        self._model = torch.load(
            f"{config['DOWNLOAD_PATH']}/{self.filename}",
            map_location=torch.device("cpu"),
        )
        self._model.to(self._device)
        print(self._model)

    def prepare_image_for_prediction(self, image):
        # Get gray image from bgr image.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize image to suit model input size.
        image = cv2.resize(image, (48, 48))
        image = cv2.resize(image, self.image_size)

        # Convert 1 channel image to 3 channel image.
        image = np.dstack([image] * 3)

        # Transform image.
        image = self.transform(image)

        # Add '1' as batch size.
        image = torch.unsqueeze(image, dim=0)

        return image

    def predict_emotion(self, image):
        image = image.to(self._device)
        prediction = self._model(image)
        prediction = torch.softmax(prediction, 0)
        classification_confidence, current_prediction = torch.max(prediction, dim=0)
        return current_prediction, classification_confidence
