# Import kivy dependencies 
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from layers import L1_distance
import os
os.chdir("D:/Facial Recognition/app/")
import numpy as np

class CamApp (App) :

    def build (self) :

        self.web_cam = Image(size_hint = (1, 0.8))
        self.button = Button(text = "Verify", on_press = self.verify, size_hint = (1, 0.1))
        self.verification_label = Label(text = "Verification Uninitiated", size_hint = (1, 0.1))

        layout = BoxLayout (orientation = "vertical")
        layout.add_widget (self.web_cam)
        layout.add_widget (self.button)
        layout.add_widget (self.verification_label)

        self.model = tf.keras.models.load_model(
            'siamese_neural_network.keras', 
            custom_objects = {"L1_distance" : L1_distance}
            )

        self.capture = cv2.VideoCapture(0) # 0 is my webcame, for you it might be something else (trial and error pls <3)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return layout

    def update (self, dt) :
        ret, frame = self.capture.read()
        if not ret:
            Logger.warning("Camera: Could not read a frame")
            return

        frame = frame [120 : 120 + 250, 200 : 200 + 250, :]
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]),
            colorfmt="bgr"
        )
        img_texture.blit_buffer(buf, colorfmt = 'bgr', bufferfmt = "ubyte")
        self.web_cam.texture = img_texture

    def preprocess (self, file_path) :
        byte_img = tf.io.read_file (file_path)
        img = tf.io.decode_image(byte_img, channels=3)
        img = tf.image.resize(img, (100, 100))
        img = img / 255
        return img

    def verify (self, *args) :
        detection_threshold = 0.99
        verification_threshold = 0.5
        application_data = os.path.join(os.path.dirname(__file__), "application_data")
        input_images = os.path.join(application_data, "input_images")
        verification_images = os.path.join(application_data, "verification_images")
        os.makedirs(input_images, exist_ok=True)

        save_path = os.path.join(input_images, "input_image.jpg")
        ret, frame = self.capture.read()
        if not ret:
            Logger.warning("Camera: Could not capture verification frame")
            return [], False

        frame = frame [120 : 120 + 250, 200 : 200 + 250, :]
        if not cv2.imwrite(save_path, frame):
            Logger.error(f"Camera: Could not save image to {save_path}")
            return [], False

        results = []
        input_img = self.preprocess(save_path)
        for image in os.listdir(verification_images) :
            validation_img = self.preprocess(os.path.join(verification_images, image))
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis = 1)))
            results.append(result)

        detection = np.sum(np.array(results) > detection_threshold)

        verification = detection / len(os.listdir(verification_images))
        verified = verification > verification_threshold

        self.verification_label.text = "Verified!" if verified else "Unverified, cope harder"

        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        return results, verified

if __name__ == "__main__" :
    CamApp().run()
