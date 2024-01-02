from manim import *

class TextScene(Scene):
    def construct(self):
        eqn = Tex("$e = mc^2$").to_corner(UL, buff=2)
        self.play(Write(eqn))
        sq = Square(side_length=5, fill_color=RED, fill_opacity=0.5).shift(LEFT*3)
        tri = Triangle().to_edge(RIGHT)
        self.play(DrawBorderThenFill(sq), run_time=2)
        self.play(Create(tri), run_time=2)
        self.wait(1)
        self.play(eqn.animate.to_edge(RIGHT), run_time=2)
        self.play(sq.animate.scale(0.5), tri.animate.to_edge(UL), run_time=2)
        self.wait(1)