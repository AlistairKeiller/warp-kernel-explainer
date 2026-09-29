"""Geometry-led, self-contained ManimGL chapters for warporacer.

Render: manimgl labs/warp_kernels/main.py Opening -w
All motion and diagrams are original. See storyboard.md for abstractions.
"""
import math
import numpy as np
from manimlib import *

INK = "#101014"
BLUE = "#58C4DD"
YELLOW = "#FFFF00"
GREEN = "#83C167"
RED = "#FC6255"
MUTED = "#9B9BA7"
WALL = "#666674"


def words(text, size=30, color=WHITE):
    return Text(text, font_size=size).set_color(color)


def math_label(text, size=36, color=WHITE):
    return Tex(text, font_size=size).set_color(color)


def car(point=ORIGIN, heading=0, color=YELLOW, scale=1):
    body = RoundedRectangle(width=.65, height=.32, corner_radius=.07)
    body.set_fill(color, 1).set_stroke(color, 1)
    window = Rectangle(width=.18, height=.23).set_fill(INK, 1).set_stroke(width=0)
    window.shift(RIGHT * .08)
    wheels = VGroup(*[
        Line([x-.065, y, 0], [x+.065, y, 0]).set_stroke(WHITE, 3)
        for x in [-.20, .20] for y in [-.19, .19]
    ])
    return VGroup(body, window, wheels).scale(scale).rotate(heading).move_to(point)


def track():
    return VGroup(*[
        Ellipse(width=w, height=h).set_stroke(c, sw, opacity=op).set_fill(opacity=0)
        for w, h, c, sw, op in [(10, 5.4, WALL, 3, 1), (7, 2.4, WALL, 3, 1),
                                   (8.5, 3.9, BLUE, 1.5, .35)]
    ])


def pose(theta, a=4.25, b=1.95):
    return np.array([a*np.cos(theta), b*np.sin(theta), 0]), math.atan2(b*np.cos(theta), -a*np.sin(theta))


def curve(points, color=BLUE, width=3):
    return VMobject().set_points_smoothly(points).set_stroke(color, width)


class FilmScene(Scene):
    def setup(self):
        super().setup()
        self.camera.background_rgba = color_to_rgba(INK)

    def heading(self, text):
        mob = words(text, 38).set_max_width(13).to_edge(UP, buff=.4)
        self.play(FadeIn(mob, shift=UP*.15), run_time=.8)
        return mob

    def caption(self, text, color=MUTED):
        mob = words(text, 25, color).to_edge(DOWN, buff=.38)
        self.play(FadeIn(mob), run_time=.6)
        return mob

    def clear(self):
        self.play(*[FadeOut(m) for m in list(self.mobjects) if m is not self.camera.frame], run_time=.7)


class Opening(FilmScene):
    def construct(self):
        question = self.heading("What can we compute independently?")
        road = track().scale(.85)
        theta = ValueTracker(-PI/2)
        racer = car(scale=.85)
        def drive(m):
            p, a = pose(theta.get_value())
            m.become(car(.85*p, a, scale=.85))
        racer.add_updater(drive)
        self.play(ShowCreation(road), FadeIn(racer), run_time=1.5)
        self.play(theta.animate.set_value(PI/2), run_time=4, rate_func=linear)
        racer.clear_updaters()
        sentence = self.caption("Each car has its own state and action.")
        self.wait(1)
        world = VGroup(road, racer)
        mini = world.copy().scale(.27).move_to([-4.7, 1.25, 0])
        self.play(Transform(world, mini), FadeOut(sentence), run_time=1.5)
        worlds = VGroup(world)
        for row in range(3):
            for col in range(4):
                if row == 0 and col == 0:
                    continue
                w = mini.copy().move_to([-4.7+3.12*col, 1.25-1.6*row, 0])
                worlds.add(w)
        self.play(LaggedStart(*[FadeIn(w, scale=.9) for w in worlds[1:]], lag_ratio=.08), run_time=2)
        indices = VGroup(*[math_label(f"i={i}", 23, BLUE).next_to(w, DOWN, buff=.07)
                           for i, w in enumerate(worlds)])
        self.play(Write(indices))
        many = self.caption("8,000 environments, each with its own car", BLUE)
        self.wait(3)
        # Identical motion in separate coordinate systems makes independence visible.
        def advance(mob, alpha):
            p, heading = pose(PI/2 + .65*alpha)
            mob[1].become(car(mob[0].get_center() + .85*.27*p,
                              heading, scale=.85*.27))
        self.play(*[UpdateFromAlphaFunc(w, advance) for w in worlds], run_time=1.5)
        chosen = worlds[5]
        self.play(*[w.animate.fade(.84) for i, w in enumerate(worlds) if i != 5],
                  *[t.animate.set_opacity(.16) for i, t in enumerate(indices) if i != 5])
        self.wait(1)
        self.play(FadeOut(worlds), FadeOut(indices), FadeOut(many), FadeOut(question))
        equation = math_label(r"\texttt{step\_kernel}:\quad i\longmapsto\text{one car's next state}", 42)
        self.play(Write(equation), run_time=2)
        self.caption("A logical work item, scheduled by the GPU.")
        self.wait(3)


class VehicleStep(FilmScene):
    """Develop motion, turning, and the acceleration limit before integration."""

    def construct(self):
        self.velocity_components()
        self.clear()
        self.steering_geometry()
        self.clear()
        self.turning_acceleration()
        self.clear()
        self.acceleration_circle()
        self.clear()
        self.turning_limit()
        self.clear()
        self.integrate()

    def velocity_components(self):
        self.heading("Where does the next position come from?")
        origin = np.array([-4.5, -1.2, 0.])
        heading = ValueTracker(.35)
        speed = ValueTracker(3.)
        racer = car(origin, .35, scale=1.3)

        def endpoint():
            a = heading.get_value()
            return origin + speed.get_value() * np.array([np.cos(a), np.sin(a), 0])

        velocity = Arrow(origin, endpoint(), buff=0, fill_color=BLUE)
        horizontal = Line(origin, [endpoint()[0], origin[1], 0]).set_stroke(BLUE, 4)
        vertical = Line([endpoint()[0], origin[1], 0], endpoint()).set_stroke(YELLOW, 4)
        baseline = DashedLine(origin + LEFT*.8, origin + RIGHT*4.5).set_stroke(WALL, 1.5)
        self.play(ShowCreation(baseline), FadeIn(racer))
        self.play(GrowArrow(velocity), run_time=1.5)
        v_label = math_label("v", 36, BLUE).next_to(velocity, UP, buff=.15)
        self.play(Write(v_label))
        self.wait(2)
        self.play(ShowCreation(horizontal), ShowCreation(vertical), run_time=2)
        x_eq = math_label(r"\dot x=v\cos\psi", 42, BLUE).move_to([3.3, .8, 0])
        y_eq = math_label(r"\dot y=v\sin\psi", 42, YELLOW).move_to([3.3, -.25, 0])
        angle = Arc(start_angle=0, angle=.35, radius=1).shift(origin).set_stroke(WHITE, 2)
        psi = math_label(r"\psi", 30).move_to(origin + [1.2, .18, 0])
        self.play(ShowCreation(angle), Write(psi), TransformFromCopy(horizontal, x_eq), run_time=1.5)
        self.play(TransformFromCopy(vertical, y_eq), run_time=1.5)
        caption = self.caption("The arrow gives the direction and speed of motion.")

        def update_geometry(_):
            end = endpoint()
            foot = np.array([end[0], origin[1], 0])
            racer.become(car(origin, heading.get_value(), scale=1.3))
            velocity.put_start_and_end_on(origin, end)
            horizontal.put_start_and_end_on(origin, foot)
            vertical.put_start_and_end_on(foot, end)
            angle.become(Arc(start_angle=0, angle=heading.get_value(), radius=1)
                         .shift(origin).set_stroke(WHITE, 2))
            psi.move_to(origin + 1.25*np.array([np.cos(heading.get_value()/2), np.sin(heading.get_value()/2), 0]))
            v_label.next_to(velocity, UP, buff=.15)

        driver = VMobject().add_updater(update_geometry)
        self.add(driver)
        self.play(heading.animate.set_value(1.05), run_time=3)
        self.wait(1.5)
        self.play(heading.animate.set_value(.5), speed.animate.set_value(3.9), run_time=3)
        self.wait(2)
        driver.clear_updaters()
        self.remove(driver)
        self.play(FadeOut(caption), FadeOut(angle), FadeOut(psi), FadeOut(v_label),
                  FadeOut(horizontal), FadeOut(vertical), FadeOut(velocity))
        direction = np.array([np.cos(.5), np.sin(.5), 0])
        displacement = Arrow(origin, origin + 2.4*direction, buff=0, fill_color=BLUE)
        ghost = racer.copy().fade(.75)
        self.add(ghost)
        self.play(ShowCreation(displacement), racer.animate.shift(2.4*direction), run_time=2.5)
        self.caption("Over a short interval: displacement ≈ velocity × time.")
        self.wait(2)

    def steering_geometry(self):
        self.heading("How does the wheel angle set the turn?")
        rear = np.array([-4.4, -1.5, 0])
        wheelbase = 2.4
        front = rear + RIGHT*wheelbase
        steering = ValueTracker(.7)
        chassis = Line(rear, front).set_stroke(WALL, 7)
        rear_wheel = Line(rear-LEFT*.35, rear+LEFT*.35).set_stroke(WHITE, 12)
        front_wheel = Line(front-LEFT*.35, front+LEFT*.35).set_stroke(YELLOW, 12)
        base_direction = DashedLine(front, front+RIGHT*1.15).set_stroke(WALL, 1.5)
        delta_arc = Arc(start_angle=0, angle=.7, radius=.8).shift(front).set_stroke(YELLOW, 2)
        delta_label = math_label(r"\delta", 30, YELLOW).move_to(front+[1, .35, 0])
        l_brace = Brace(chassis, DOWN, buff=.3)
        l_label = math_label("L", 32, BLUE).next_to(l_brace, DOWN, buff=.12)

        def circle_center():
            return rear + UP*(wheelbase/np.tan(steering.get_value()))

        center = circle_center()
        rear_normal = Line(rear, center).set_stroke(BLUE, 2)
        front_normal = Line(front, center).set_stroke(YELLOW, 2)
        center_dot = Dot(center, radius=.065, fill_color=GREEN)
        r_label = math_label("R", 32, BLUE).move_to((rear+center)/2+LEFT*.35)
        path = Arc(start_angle=-PI/2, angle=1.15, radius=center[1]-rear[1]).shift(center).set_stroke(GREEN, 3)
        center_label = words("turn center", 23, GREEN).next_to(center_dot, UP, buff=.17)
        self.play(ShowCreation(chassis), FadeIn(rear_wheel), FadeIn(front_wheel))
        self.play(GrowFromCenter(l_brace), Write(l_label))
        self.play(Rotate(front_wheel, .7, about_point=front), ShowCreation(base_direction),
                  ShowCreation(delta_arc), Write(delta_label), run_time=2)
        self.play(ShowCreation(rear_normal), ShowCreation(front_normal), run_time=2)
        self.play(FadeIn(center_dot), Write(center_label), Write(r_label))
        self.play(ShowCreation(path), run_time=2)
        equation = math_label(r"\tan\delta={L\over R}", 44).move_to([3.1, 1, 0])
        radius_equation = math_label(r"R={L\over\tan\delta}", 44, GREEN).move_to([3.1, -.5, 0])
        self.play(Write(equation), run_time=2)
        self.play(TransformFromCopy(equation, radius_equation), run_time=2)
        caption = self.caption("The turn center lies perpendicular to both wheel directions.")
        self.wait(2)

        def update(_):
            delta = steering.get_value()
            c = circle_center()
            tangent = np.array([np.cos(delta), np.sin(delta), 0])
            front_wheel.put_start_and_end_on(front-.35*tangent, front+.35*tangent)
            rear_normal.put_start_and_end_on(rear, c)
            front_normal.put_start_and_end_on(front, c)
            center_dot.move_to(c)
            center_label.next_to(center_dot, UP, buff=.17)
            r_label.move_to((rear+c)/2+LEFT*.35)
            path.become(Arc(start_angle=-PI/2, angle=1.15, radius=c[1]-rear[1])
                        .shift(c).set_stroke(GREEN, 3))
            delta_arc.become(Arc(start_angle=0, angle=delta, radius=.8)
                             .shift(front).set_stroke(YELLOW, 2))
            delta_label.move_to(front+1.08*np.array([np.cos(delta/2), np.sin(delta/2), 0]))

        driver = VMobject().add_updater(update)
        self.add(driver)
        self.play(steering.animate.set_value(.93), run_time=3.5)
        self.wait(1.5)
        self.play(FadeOut(caption))
        self.caption("More steering gives a smaller radius.")
        self.play(steering.animate.set_value(.65), run_time=3.5)
        self.play(steering.animate.set_value(.85), run_time=3.5)
        self.wait(2)
        driver.clear_updaters()
        self.remove(driver)

    def turning_acceleration(self):
        self.heading("Steering bends the path.")
        center = np.array([-3.1, .3, 0])
        radius = 1.9
        theta = ValueTracker(-PI/2)
        path = Circle(radius=radius).move_to(center).set_stroke(WALL, 2)
        racer = car(center + DOWN*radius)
        velocity = Arrow(ORIGIN, RIGHT, buff=0, fill_color=BLUE)
        lateral = Arrow(ORIGIN, UP, buff=0, fill_color=YELLOW)
        speed = ValueTracker(1.)
        radial = DashedLine(center, center + DOWN*radius).set_stroke(WALL, 1.5)

        def geometry(_):
            angle = theta.get_value()
            normal = np.array([np.cos(angle), np.sin(angle), 0])
            tangent = np.array([-np.sin(angle), np.cos(angle), 0])
            point = center + radius*normal
            racer.become(car(point, angle+PI/2, scale=.85))
            velocity.put_start_and_end_on(point, point + .8*speed.get_value()*tangent)
            lateral.put_start_and_end_on(point, point - .35*speed.get_value()**2*normal)
            radial.put_start_and_end_on(center, point)

        driver = VMobject().add_updater(geometry)
        geometry(driver)
        self.play(ShowCreation(path), FadeIn(racer))
        self.add(driver)
        self.play(GrowArrow(velocity), ShowCreation(radial))
        r_label = math_label("R", 32, MUTED).move_to(center+[-.25, -.95, 0])
        self.play(Write(r_label))
        equation = math_label(r"R={L\over\tan\delta}", 42).move_to([3.2, 1.45, 0])
        delta_note = words("δ: steering angle     L: wheelbase", 25, MUTED).move_to([3.2, .6, 0])
        self.play(Write(equation), FadeIn(delta_note))
        self.play(theta.animate.set_value(-.1), run_time=3.5, rate_func=linear)
        self.play(GrowArrow(lateral))
        lat_equation = math_label(r"a_{\rm lateral}={v^2\over R}=v\dot\psi", 40, YELLOW).move_to([3.2, -.65, 0])
        self.play(Write(lat_equation), run_time=2)
        caption = self.caption("Changing direction takes acceleration toward the center.")
        self.play(theta.animate.set_value(1.5), run_time=3.5, rate_func=linear)
        self.wait(2)
        self.play(FadeOut(caption), FadeOut(r_label))
        ratio = VGroup(math_label(r"v\ \longrightarrow\ 2v", 34, BLUE),
                       math_label(r"a_{\rm lateral}\ \longrightarrow\ 4a_{\rm lateral}", 34, YELLOW))
        ratio.arrange(DOWN, buff=.25).move_to([3.2, -2.3, 0])
        self.play(Write(ratio), speed.animate.set_value(2), run_time=3)
        self.play(theta.animate.set_value(4.5), run_time=4, rate_func=linear)
        self.wait(2)
        driver.clear_updaters()
        self.remove(driver)

    def acceleration_circle(self):
        self.heading("How much acceleration can the tires provide?")
        origin = np.array([-3.4, -.1, 0])
        unit = 2.05
        request_x = ValueTracker(.65)
        request_y = ValueTracker(.45)
        grip = ValueTracker(1.)
        x_axis = Line(origin+LEFT*2.65, origin+RIGHT*3.15).set_stroke(WALL, 1.5)
        y_axis = Line(origin+DOWN*2.6, origin+UP*2.6).set_stroke(WALL, 1.5)
        x_name = words("accelerate", 22, BLUE).move_to(origin+[2.1, -.42, 0])
        brake = words("brake", 22, BLUE).move_to(origin+[-2, -.42, 0])
        y_name = words("turn", 24, YELLOW).move_to(origin+[0, 2.85, 0])
        self.play(ShowCreation(x_axis), ShowCreation(y_axis), Write(x_name), Write(brake), Write(y_name))
        self.caption("These axes measure acceleration, not position.")
        foot = origin+RIGHT*(unit*.65)
        point = foot+UP*(unit*.45)
        x_component = Line(origin, foot).set_stroke(BLUE, 5)
        y_component = Line(foot, point).set_stroke(YELLOW, 5)
        resultant = Arrow(origin, point, buff=0, fill_color=GREEN)
        self.play(ShowCreation(x_component), run_time=1.5)
        self.play(ShowCreation(y_component), run_time=1.5)
        self.play(GrowArrow(resultant), run_time=1.5)
        pythagoras = math_label(r"|a|^2=a_{\rm long}^2+a_{\rm lateral}^2", 36).move_to([3.3, 1.5, 0])
        self.play(Write(pythagoras), run_time=2)
        self.wait(2)
        force = math_label(r"|F|\leq\mu mg", 40).move_to([3.3, .3, 0])
        self.play(Write(force))
        limit = math_label(r"|a|\leq\mu g", 42, GREEN).move_to([3.3, -.9, 0])
        self.play(TransformFromCopy(force, limit), run_time=2)
        self.wait(2)
        circle = Circle(radius=unit).move_to(origin).set_stroke(GREEN, 2.5).set_fill(GREEN, .035)
        radius_line = Line(origin, origin+RIGHT*unit).set_stroke(GREEN, 2)
        radius_label = math_label(r"\mu g", 29, GREEN).move_to(origin+[1, -.25, 0])
        self.play(FadeOut(resultant), FadeOut(x_component), FadeOut(y_component),
                  ShowCreation(radius_line), Write(radius_label))
        self.play(ShowCreation(circle), Rotate(radius_line, TAU, about_point=origin),
                  run_time=4, rate_func=linear)
        self.play(FadeOut(radius_line), FadeOut(radius_label))
        self.wait(2)
        self.play(FadeOut(force), FadeOut(limit), FadeOut(pythagoras))
        relation = math_label(r"a_{\rm long}^2+a_{\rm lateral}^2\leq(\mu g)^2", 35).move_to([3.35, 1.65, 0])
        self.play(Write(relation))

        # Same order as deriv(): cap lateral demand, then clip longitudinal demand.
        def values():
            cap = grip.get_value()
            x, y = request_x.get_value(), request_y.get_value()
            lateral = np.clip(y, -cap, cap)
            remaining = np.sqrt(max(cap*cap-lateral*lateral, 0))
            longitudinal = np.clip(x, -remaining, remaining)
            return x, y, longitudinal, lateral, remaining

        requested = DashedLine(origin, point).set_stroke(RED, 2.5)
        request_dot = Dot(point, radius=.07, fill_color=RED)
        applied = Arrow(origin, point, buff=0, fill_color=GREEN)
        applied_dot = Dot(point, radius=.065, fill_color=GREEN)
        removed = Line(point, point+RIGHT*.001).set_stroke(RED, 4)
        allowance = Line(origin+LEFT, origin+RIGHT).set_stroke(BLUE, 4)
        horizontal_guide = DashedLine(origin, point).set_stroke(WALL, 1.5)
        legend = VGroup(words("requested", 27, RED), words("applied", 27, GREEN))
        legend.arrange(DOWN, aligned_edge=LEFT, buff=.3).move_to([3.35, .1, 0])
        explanation = words("Inside the circle, the request fits.", 26).move_to([3.3, -1.6, 0])

        def update(_):
            x, y, ax, ay, remaining = values()
            req = origin+unit*np.array([x, y, 0])
            actual = origin+unit*np.array([ax, ay, 0])
            corner = origin+unit*np.array([ax, 0, 0])
            circle.become(Circle(radius=unit*grip.get_value()).move_to(origin)
                          .set_stroke(GREEN, 2.5).set_fill(GREEN, .035))
            requested.put_start_and_end_on(origin, req)
            request_dot.move_to(req)
            applied.put_start_and_end_on(origin, actual)
            applied_dot.move_to(actual)
            # Avoid zero-length geometry at the circle's top and at exact fits.
            x_component.put_start_and_end_on(origin, corner+RIGHT*1e-6)
            y_component.put_start_and_end_on(corner, actual+UP*1e-6)
            removed.put_start_and_end_on(actual, req+RIGHT*1e-6)
            allowance.put_start_and_end_on(origin+unit*np.array([-remaining, ay, 0]),
                                          origin+unit*np.array([remaining+1e-6, ay, 0]))
            horizontal_guide.put_start_and_end_on(origin+UP*(unit*ay), actual+RIGHT*1e-6)

        driver = VMobject().add_updater(update)
        update(driver)
        self.play(FadeIn(x_component), FadeIn(y_component), FadeIn(applied), FadeIn(applied_dot), FadeIn(explanation))
        self.add(driver)
        self.play(request_x.animate.set_value(.4), request_y.animate.set_value(.65), run_time=3)
        self.wait(2)
        self.play(FadeOut(explanation), FadeIn(requested), FadeIn(request_dot), FadeIn(legend))
        outside = words("More throttle than the tires can supply.", 25, RED).move_to([3.3, -1.6, 0])
        self.play(request_x.animate.set_value(1.2), FadeIn(removed), run_time=3)
        self.play(Write(outside))
        self.wait(3)
        self.play(ShowCreation(allowance), ShowCreation(horizontal_guide))
        remaining_eq = math_label(r"|a_{\rm long}|\leq\sqrt{(\mu g)^2-a_{\rm lateral}^2}", 33, BLUE).move_to([3.3, -2.5, 0])
        self.play(Write(remaining_eq), run_time=2)
        self.wait(3)
        self.play(FadeOut(outside))
        tradeoff = words("A harder turn leaves less for acceleration.", 24).move_to([3.3, -1.55, 0])
        self.play(FadeIn(tradeoff), request_y.animate.set_value(.94), run_time=4)
        self.wait(2)
        self.play(request_y.animate.set_value(.2), run_time=4)
        self.wait(2)
        self.play(FadeOut(tradeoff))
        braking = words("Braking uses the same circle.", 27).move_to([3.3, -1.55, 0])
        self.play(FadeIn(braking), request_x.animate.set_value(-1.15), request_y.animate.set_value(.55), run_time=4)
        self.wait(3)
        self.play(FadeOut(braking), FadeOut(remaining_eq), request_x.animate.set_value(.55),
                  request_y.animate.set_value(.6), run_time=3)
        friction = words("Lower friction means a smaller circle.", 26).move_to([3.3, -1.55, 0])
        self.play(FadeIn(friction))
        old_circle = circle.copy().set_fill(opacity=0).set_stroke(WALL, 1.5, opacity=.5)
        self.add(old_circle)
        mu_label = math_label(r"\mu/\mu_0=", 30).move_to([2.9, -2.5, 0])
        mu_value = DecimalNumber(1., num_decimal_places=2, font_size=30).set_color(GREEN).next_to(mu_label, RIGHT)
        mu_value.add_updater(lambda m: m.set_value(grip.get_value()))
        self.play(FadeIn(mu_label), FadeIn(mu_value))
        self.play(grip.animate.set_value(.65), run_time=4)
        self.wait(3)
        self.play(grip.animate.set_value(.85), run_time=3)
        self.wait(2)
        self.play(FadeOut(friction), FadeOut(mu_label), FadeOut(mu_value), FadeOut(old_circle))
        cap_turn = words("At the turning limit, no forward acceleration is left.", 24).move_to([3.3, -1.55, 0])
        self.play(FadeIn(cap_turn), request_y.animate.set_value(1.15), run_time=4)
        self.wait(3)
        driver.clear_updaters()
        mu_value.clear_updaters()
        self.remove(driver)

    def turning_limit(self):
        self.heading("What happens when the requested turn needs too much grip?")
        start = np.array([-5, -1.7, 0])
        desired_radius, actual_radius = 2., 3.2
        distance = ValueTracker(0.)

        def point(radius, length):
            angle = length/radius
            return start+radius*np.array([np.sin(angle), 1-np.cos(angle), 0])

        # Equal travel distance and speed; only the capped heading rate differs.
        requested_path = curve([point(desired_radius, s) for s in np.linspace(0, 4.2, 90)], RED, 2)
        actual_path = curve([point(actual_radius, s) for s in np.linspace(0, 4.2, 90)], GREEN, 3)
        ghost = car(start, color=RED, scale=.85).fade(.4)
        racer = car(start, color=GREEN, scale=.85)
        self.play(ShowCreation(requested_path), run_time=2)
        requested_label = words("requested turn", 24, RED).next_to(requested_path, UP, buff=.15)
        self.play(Write(requested_label))
        self.play(FadeIn(ghost), FadeIn(racer))
        limit = math_label(r"a_{\rm lateral}\leq\mu g", 40, YELLOW).move_to([3,1.4,0])
        radius_limit = math_label(r"R\geq{v^2\over\mu g}", 43, GREEN).move_to([3,0,0])
        self.play(Write(limit), run_time=1.5)
        self.play(TransformFromCopy(limit, radius_limit), run_time=2)
        self.caption("The simulator limits the turn rate. The resulting path is wider.")

        def update(_):
            s = distance.get_value()
            ghost.become(car(point(desired_radius, s), s/desired_radius, RED, .85).fade(.4))
            racer.become(car(point(actual_radius, s), s/actual_radius, GREEN, .85))

        driver = VMobject().add_updater(update)
        self.add(driver)
        self.play(distance.animate.set_value(4.2), ShowCreation(actual_path), run_time=6, rate_func=linear)
        actual_label = words("limited by grip", 24, GREEN).move_to([-1.2,-1.45,0])
        self.play(Write(actual_label))
        self.wait(3)
        slower = words("Reducing speed lowers the required turning acceleration.", 24).move_to([2.8,-1.65,0])
        slower.set_max_width(6)
        self.play(FadeIn(slower))
        self.wait(3)
        driver.clear_updaters()
        self.remove(driver)

    def integrate(self):
        self.heading("Now advance the state by 1/60 second.")
        interval = Line([-5.5, 1.65, 0], [5.5, 1.65, 0]).set_stroke(WHITE, 2)
        ticks = VGroup(*[Line([x, 1.5, 0], [x, 1.8, 0]) for x in np.linspace(-5.5, 5.5, 7)])
        brace = Brace(interval, UP, buff=.15)
        dt = math_label(r"\Delta t=1/60\ \text{second}", 30).next_to(brace, UP, buff=.1)
        self.play(ShowCreation(interval), ShowCreation(ticks), GrowFromCenter(brace), Write(dt))
        sublabels = VGroup(*[math_label(str(i+1), 25, MUTED).move_to([x, 1.2, 0])
                               for i, x in enumerate(np.linspace(-4.583, 4.583, 6))])
        self.play(Write(sublabels))
        # Actual RK4 positions for six 1/360 s substeps; displacement magnified.
        state = np.array([0., 0., .15, 4.])
        h, delta, steer_rate, accel, wheelbase = 1/360, .24, .4, 1., .3302
        states = [state.copy()]
        def f(s, d):
            limit = 1.0489 * 9.81
            yaw_cap = limit / max(abs(s[3]), .5)
            yaw = np.clip(s[3]*np.tan(d)/wheelbase, -yaw_cap, yaw_cap)
            remaining = np.sqrt(max(limit*limit-(s[3]*yaw)**2, 0.))
            applied_accel = np.clip(accel, -remaining, remaining)
            return np.array([s[3]*np.cos(s[2]), s[3]*np.sin(s[2]), yaw, applied_accel])
        for _ in range(6):
            k1 = f(state, delta)
            k2 = f(state+h*k1/2, delta+steer_rate*h/2)
            k3 = f(state+h*k2/2, delta+steer_rate*h/2)
            k4 = f(state+h*k3, delta+steer_rate*h)
            state = state+h*(k1+2*k2+2*k3+k4)/6
            delta += steer_rate*h
            states.append(state.copy())
        points = [np.array([-5+145*s[0], -.7+145*s[1], 0]) for s in states]
        path = curve(points, BLUE, 2).set_opacity(.3)
        racer = car(points[0], states[0][2], scale=.9)
        self.play(ShowCreation(path), FadeIn(racer))
        formula = math_label(r"s_{n+1}=s_n+{h\over6}(k_1+2k_2+2k_3+k_4)", 36).move_to([0,-1.75,0])
        step_note = words("One RK4 update: four slope estimates, one new state", 25, BLUE).next_to(formula, DOWN, buff=.3)
        self.play(Write(formula), FadeIn(step_note))
        self.caption("Displacement magnified ×145; six real RK4 substeps.")
        for i in range(6):
            dot = Dot(points[i], radius=.05, fill_color=BLUE)
            segment = Line(ticks[i].get_center(), ticks[i+1].get_center()).set_stroke(YELLOW, 5)
            self.add(dot)
            self.play(ShowCreation(segment), Transform(racer, car(points[i+1], states[i+1][2], scale=.9)),
                      sublabels[i].animate.set_color(YELLOW), run_time=1.25)
        self.wait(3)


class RewardAndRespawn(FilmScene):
    def construct(self):
        self.heading("The map answers two different questions.")
        top, bottom = 1.6, -1.6
        walls = VGroup(Line([-6,top,0],[6,top,0]), Line([-6,bottom,0],[6,bottom,0])).set_stroke(WALL, 4)
        centerline = DashedLine([-6,0,0],[6,0,0]).set_stroke(BLUE, 2)
        waypoints = VGroup(*[Dot([x,0,0], radius=.05, fill_color=BLUE) for x in np.arange(-5,6)])
        racer = car([-3,.55,0])
        self.play(ShowCreation(walls), ShowCreation(centerline), FadeIn(waypoints), FadeIn(racer))
        self.caption("A local stretch of track; lookup maps are prepared once.")
        distance = Line([-3,.55,0],[-3,top,0]).set_stroke(YELLOW, 4)
        ring = Circle(radius=top-.55).move_to(racer).set_stroke(YELLOW, 2)
        edt = words("How close is the wall?", 29, YELLOW).move_to([-3.4,2.55,0])
        self.play(ShowCreation(ring), ShowCreation(distance), Write(edt))
        lookup = words("EDT: distance to the nearest wall", 24, YELLOW).move_to([2.7,2.55,0])
        self.play(FadeIn(lookup))
        self.wait(1.5)
        self.play(FadeOut(ring), FadeOut(distance), FadeOut(edt), FadeOut(lookup))
        nearest = Line(racer.get_center(),[-3,0,0]).set_stroke(GREEN, 4)
        lut = words("Where am I along the track?", 29, BLUE).move_to([0,2.55,0])
        self.play(Write(lut), ShowCreation(nearest), FlashAround(waypoints[2]))
        travel = Arrow([-3,-.55,0],[1,-.55,0],buff=0,fill_color=GREEN)
        progress = words("signed waypoint progress", 25, GREEN).next_to(travel, DOWN, buff=.15)
        self.play(Transform(racer,car([1,.55,0])), Transform(nearest,Line([1,.55,0],[1,0,0]).set_stroke(GREEN,4)),
                  ShowCreation(travel), run_time=2)
        self.play(Write(progress), FlashAround(waypoints[6]))
        self.wait(1)
        self.play(FadeOut(lut), FadeOut(travel), FadeOut(progress))
        reward = math_label(r"r=\text{progress}-\text{wall penalty}-\text{offset}^2", 36).move_to([0,2.55,0])
        self.play(Write(reward))
        self.play(Indicate(nearest, color=RED), run_time=1)
        self.wait(1)
        self.play(FadeOut(nearest), FadeOut(reward))
        footprint = Circle(radius=float(np.hypot(.65, .38)/2)).move_to(racer).set_stroke(RED, 2)
        clearance = math_label(r"\text{clearance}=d_{\rm wall}-{1\over2}\text{car diagonal}", 34).move_to([0,2.55,0])
        self.play(Write(clearance), ShowCreation(footprint))
        self.play(racer.animate.shift(UP*.85), footprint.animate.shift(UP*.85), run_time=1.6)
        hit = words("clearance < 0", 30, RED).move_to([3, .5, 0])
        self.play(Write(hit), FlashAround(footprint, color=RED))
        self.wait(1)
        self.play(FadeOut(hit), FadeOut(footprint), FadeOut(clearance),
                  Transform(racer, car([-4,0,0])))
        reset = words("done = 1    •    reward = −25    •    respawn", 30, RED).move_to([0,2.55,0])
        self.play(Write(reset))
        detail = words("Respawn at rest, with new friction and wheelbase values.",25).move_to([0,-2.5,0])
        self.play(FadeIn(detail))
        self.wait(3)


class WarpLidar(FilmScene):
    def construct(self):
        self.heading("How far can this ray travel?")
        # Analytic corridor: every circle really touches its nearest wall.
        # This continuous illustration explains the discrete EDT lookup in sim.py.
        walls = VGroup(Line([-6,2,0],[5.5,2,0]), Line([-6,-2,0],[5.5,-2,0]),
                       Line([5.5,-2,0],[5.5,2,0])).set_stroke(WALL, 4)
        start = np.array([-5.,-1.,0.])
        direction = np.array([.96,.28,0.])
        hit_distance = min((2-start[1])/direction[1], (5.5-start[0])/direction[0])
        end = start+hit_distance*direction
        racer = car(start- .3*direction, math.atan2(.28,.96), scale=.75)
        beam = DashedLine(start,end).set_stroke(BLUE,2,opacity=.45)
        self.play(ShowCreation(walls), FadeIn(racer), ShowCreation(beam))
        cap = self.caption("The nearest wall may be beside the ray.")
        cursor = Dot(start,radius=.065,fill_color=YELLOW)
        self.add(cursor)
        p = start.copy()
        rings = VGroup()
        jumps = VGroup()
        formula = math_label(r"p_{n+1}=p_n+d(p_n)\,\hat u", 34).move_to([0,2.7,0])
        for i in range(24):
            radius = min(2-p[1], p[1]+2, 5.5-p[0])
            if radius < .015:
                break
            ring = Circle(radius=radius).move_to(p).set_stroke(YELLOW, 2).set_fill(YELLOW,.035)
            radius_line = Line(p,p+direction*radius).set_stroke(YELLOW,3)
            self.play(ShowCreation(ring), ShowCreation(radius_line),run_time=.7 if i<3 else .3)
            if i == 0:
                label = math_label(r"d(p)",30,YELLOW).next_to(radius_line,DOWN,buff=.2)
                self.play(Write(label))
                self.wait(.8)
                self.play(FadeOut(label),Write(formula))
            q = p+direction*radius
            self.play(cursor.animate.move_to(q),ring.animate.set_stroke(opacity=.15).set_fill(opacity=0),
                      radius_line.animate.set_color(BLUE),run_time=.7 if i<3 else .3)
            rings.add(ring)
            jumps.add(radius_line)
            p=q
        self.play(cursor.animate.move_to(end), FlashAround(cursor, color=YELLOW), run_time=.5)
        self.wait(1)
        self.play(FadeOut(cap),FadeOut(rings),FadeOut(formula))
        range_line = Line(start,end).set_stroke(BLUE,4)
        ray_angle = math.atan2(direction[1], direction[0])
        normal = np.array([direction[1], -direction[0], 0])
        distance_brace = Brace(range_line, normal, buff=.22)
        result = math_label(r"r_j=\sum_n d(p_n)",34,BLUE)
        result.rotate(ray_angle).move_to(distance_brace.get_tip() + .4*normal)
        self.play(ShowCreation(range_line),GrowFromCenter(distance_brace),Write(result))
        self.caption("In the kernel: pixel EDT samples; stop at zero or the 20 m cap.")
        self.wait(3)
        self.clear()

        self.heading("Now ask in 108 directions.")
        origin = np.array([-3.3,0,0])
        racer = car(origin)
        sensor = origin+RIGHT*.27
        # Sample of a 270-degree fan, clipped to a rectangular local room.
        beams = VGroup()
        for a in np.linspace(-3*PI/4,3*PI/4,19):
            d=np.array([np.cos(a),np.sin(a),0])
            candidates=[]
            for axis, low, high in [(0,-5.9,-.7),(1,-2,2)]:
                if abs(d[axis])>1e-8:
                    candidates.append(((high if d[axis]>0 else low)-sensor[axis])/d[axis])
            beams.add(Line(sensor,sensor+min(candidates)*d).set_stroke(BLUE,1.6))
        room=Rectangle(width=5.2,height=4).move_to(origin).set_stroke(WALL,3)
        self.play(ShowCreation(room),FadeIn(racer))
        self.play(LaggedStartMap(ShowCreation,beams,lag_ratio=.04),run_time=2)
        sample=words("19 rays drawn / 108 computed",22,MUTED).move_to([-3.3,-2.5,0])
        self.play(FadeIn(sample))
        index=math_label(r"(i,j)\longmapsto\text{one ray}",38,BLUE).move_to([3,1.5,0])
        self.play(Write(index))
        cells=VGroup(*[Square(side_length=.55).set_stroke(BLUE,1.5) for _ in range(8)]).arrange(RIGHT,buff=.04).move_to([3,0,0])
        labels=VGroup(*[math_label(t,22).move_to(c) for t,c in zip([r"\delta","v","r_0","r_1",r"\cdots","r_j",r"\cdots","r_{107}"],cells)])
        self.play(ShowCreation(cells),Write(labels))
        selected=beams[10]
        self.play(selected.animate.set_stroke(YELLOW,4),cells[5].animate.set_stroke(YELLOW,3))
        self.play(TransformFromCopy(selected,labels[5]),run_time=1.2)
        write=math_label(r"\texttt{obs}[i,\,2+j]=r_j",32,YELLOW).move_to([3,-1.25,0])
        self.play(Write(write))
        self.caption("Each ray has its own loop. Different rays are independent.")
        self.wait(3)


class TheHandoff(FilmScene):
    def construct(self):
        self.heading("Two ways to divide the work.")
        rows=VGroup()
        cars=VGroup()
        for i in range(4):
            y=1.55-i*.9
            cars.add(car([-5.2,y,0],scale=.65,color=YELLOW))
            dots=VGroup(*[Dot([-3.9+j*.52,y,0],radius=.06,fill_color=YELLOW) for j in range(6)])
            links=VGroup(*[Line(dots[j].get_center(),dots[j+1].get_center()).set_stroke(YELLOW,1.5) for j in range(5)])
            rows.add(VGroup(links,dots))
        car_title=words("one item per car",29,YELLOW).move_to([-3.6,2.55,0])
        beam_title=words("one item per (car, beam)",29,BLUE).move_to([3.1,2.55,0])
        self.play(FadeIn(cars),Write(car_title))
        for j in range(6):
            self.play(*[FadeIn(row[1][j]) for row in rows],
                      *([ShowCreation(row[0][j-1]) for row in rows] if j else []),run_time=.35)
        sequential=words("6 dependent substeps",24,MUTED).move_to([-3.4,-2.1,0])
        self.play(Write(sequential))
        grid=VGroup(*[Dot([1.2+j*.35,1.55-i*.9,0],radius=.055,fill_color=BLUE) for i in range(4) for j in range(12)])
        arrow=Arrow([-.7,.2,0],[.7,.2,0],buff=.05,color=WHITE)
        self.play(ShowCreation(arrow),Write(beam_title))
        self.play(LaggedStartMap(FadeIn,grid,lag_ratio=.02),run_time=1.5)
        ray_label=words("108 independent rays per car",24,MUTED).move_to([3.2,-2.1,0])
        self.play(Write(ray_label))
        self.caption("Physics finishes first. Lidar reads the updated or respawned pose.")
        self.wait(3)
        self.clear()
        self.heading("One call to Env.step()")
        stages=VGroup(*[words(t,29,c) for t,c in [
            ("actions",BLUE),("physics",YELLOW),("lidar",BLUE),("RNG tick",MUTED)]])
        stages.arrange(RIGHT,buff=1.25).move_to([0,.75,0])
        self.play(LaggedStartMap(FadeIn,stages,lag_ratio=.3),run_time=2)
        arrows=VGroup(*[Arrow(a.get_right(),b.get_left(),buff=.15,stroke_width=2) for a,b in zip(stages,stages[1:])])
        self.play(LaggedStartMap(ShowCreation,arrows,lag_ratio=.25))
        outputs=math_label(r"\text{observation}\qquad\text{reward}\qquad\text{done}",38,GREEN).move_to([0,-.6,0])
        self.play(Write(outputs))
        brace=Brace(outputs,DOWN,buff=.25)
        shared=words("Warp buffers = Torch tensor storage",29,GREEN).next_to(brace,DOWN,buff=.2)
        self.play(GrowFromCenter(brace),Write(shared))
        self.caption("On the same CUDA device: shared output storage, Torch’s current stream.")
        self.wait(3)
        self.clear()
        final=VGroup(words("Six physics updates per car, in order.",38,YELLOW),
                     words("Independent work across cars and rays.",38,BLUE)).arrange(DOWN,buff=.55)
        self.play(Write(final),run_time=2.2)
        self.wait(3)
