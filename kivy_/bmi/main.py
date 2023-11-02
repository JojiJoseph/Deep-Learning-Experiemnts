from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import NumericProperty


class BMI(BoxLayout):
    bmi = NumericProperty(0)
    def calculate_bmi(self, height, weight):
        bmi = weight / (height ** 2)
        return bmi
    def on_text_input(self):
        if self.ids.height.text != "" and self.ids.weight.text != "":
            self.bmi = self.calculate_bmi(float(self.ids.height.text), float(self.ids.weight.text))
        return self.bmi

class BMIApp(App):
    def build(self):
        return BMI()
    
if __name__ == "__main__":
    BMIApp().run()