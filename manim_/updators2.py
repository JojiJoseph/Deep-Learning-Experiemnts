from manim import *

class CountDownScene(Scene):
    def construct(self):
        k = ValueTracker(10)
        text = DecimalNumber(k.get_value(), num_decimal_places=1)
        text.add_updater(lambda m: m.set_value(k.get_value()))
        self.play(FadeIn(text))
        self.play(k.animate.set_value(0), rate_func=linear, run_time=10)
        self.wait(1)