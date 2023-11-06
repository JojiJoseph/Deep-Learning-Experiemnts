from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2

cap = cv2.VideoCapture(0)

class Cheese(BoxLayout):
    pass

class CheeseApp(App):

    def build(self):
        self.cheese = Cheese()
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return self.cheese
    
    def update(self, dt):
        ret, frame = cap.read()
        if ret:

            frame = cv2.flip(frame, 0)
            h, w, _ = frame.shape

            texture = Texture.create(size=(w, h))
            texture.blit_buffer(frame.flatten(), colorfmt='bgr', bufferfmt='ubyte')

            self.cheese.ids.cheese.texture = texture
        

    
if __name__ == "__main__":
    CheeseApp().run()
    cap.release()