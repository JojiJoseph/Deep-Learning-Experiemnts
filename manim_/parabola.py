from manim import *

class ParabolaScene(Scene):
    def construct(self):
        # axes
        axes = Axes(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            x_length=5,
            y_length=5,
            tips=True
        )
        # Set labels
        labels = axes.get_axis_labels(x_label='x', y_label='y')
        func_title = MathTex(r"f(x) = x^2").to_edge(UP)
        # Show ticks
        axes.add_coordinates(range(-8, 9, 4), range(-8, 9, 4))

        # parabola
        parabola = axes.plot(lambda x: x**2,x_range=[-3,3], color=BLUE)
        self.play(Create(axes))
        self.play(Create(VGroup(labels, func_title)))
        self.play(Create(parabola))
        self.wait(1)