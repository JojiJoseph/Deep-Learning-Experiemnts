from manim import *

class CountDownScene(Scene):
    def construct(self):
        k = ValueTracker(10)
        text = DecimalNumber(k.get_value(), num_decimal_places=0)
        text.add_updater(lambda m: m.set_value(k.get_value()))
        self.play(FadeIn(text))
        self.wait(1)
        for i in range(10):
            self.play(k.animate.set_value(k.get_value() - 1))
            self.wait(1)