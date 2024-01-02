from manim import *

class Test(Scene):
    def construct(self):
        circle = Circle(radius=0.5)
        self.play(Create(circle))
        triangle = Triangle()
        self.play(Transform(circle, triangle))
        self.wait(5)