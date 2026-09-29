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


def physics_derivative(state, delta, accel=1., wheelbase=.3302, mu=1.0489):
    """NumPy version of sim.py's bicycle derivative, including grip caps."""
    limit = mu * 9.81
    yaw_cap = limit / max(abs(state[3]), .5)
    yaw = np.clip(state[3]*np.tan(delta)/wheelbase, -yaw_cap, yaw_cap)
    remaining = np.sqrt(max(limit*limit-(state[3]*yaw)**2, 0.))
    return np.array([state[3]*np.cos(state[2]), state[3]*np.sin(state[2]),
                     yaw, np.clip(accel, -remaining, remaining)])


def rk4_step(state, delta, h, steer_rate, accel=1.):
    """Return trial states, derivative samples, and the weighted next state."""
    k1 = physics_derivative(state, delta, accel)
    s2 = state+h*k1/2
    k2 = physics_derivative(s2, delta+steer_rate*h/2, accel)
    s3 = state+h*k2/2
    k3 = physics_derivative(s3, delta+steer_rate*h/2, accel)
    s4 = state+h*k3
    k4 = physics_derivative(s4, delta+steer_rate*h, accel)
    result = state+h*(k1+2*k2+2*k3+k4)/6
    return [state.copy(), s2, s3, s4], [k1, k2, k3, k4], result


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
        # Different local motion makes the independent environments visible.
        def advance(index):
            def update(mob, alpha):
                p, heading = pose(PI/2 + .65*(1+.08*index)*alpha)
                mob[1].become(car(mob[0].get_center() + .85*.27*p,
                                  heading, scale=.85*.27))
            return update
        self.play(*[UpdateFromAlphaFunc(w, advance(i)) for i, w in enumerate(worlds)], run_time=3)
        chosen = worlds[5]
        self.play(*[w.animate.fade(.84) for i, w in enumerate(worlds) if i != 5],
                  *[t.animate.set_opacity(.16) for i, t in enumerate(indices) if i != 5])
        self.wait(1)
        self.play(*[FadeOut(w) for i, w in enumerate(worlds) if i != 5],
                  FadeOut(indices), FadeOut(many), run_time=1)
        self.play(chosen.animate.scale(2.8).move_to([-2.5, 0, 0]), run_time=2)
        item = math_label(r"\texttt{step\_kernel}(i=5)", 33, YELLOW).move_to([3.55, 1.25, 0])
        state = math_label(r"s_5\longmapsto s'_5", 43).move_to([3.55, 0, 0])
        action = math_label(r"a_5", 36, BLUE).move_to([3.55, -1.5, 0])
        action_arrow = Arrow([3.55, -1.1, 0], [3.55, -.4, 0], buff=0, fill_color=BLUE)
        self.play(Write(item), run_time=1.5)
        self.play(Write(state), Write(action), GrowArrow(action_arrow), run_time=2)
        self.caption("Its own action updates its own state.")
        previous_pose = chosen[1].copy().fade(.75)
        self.add(previous_pose)
        def update_selected(mob, alpha):
            p, angle = pose(PI/2 + .65*1.4 + .45*alpha)
            factor = .85*.27*2.8
            mob[1].become(car(mob[0].get_center()+factor*p, angle, scale=factor))
        self.play(FlashAround(action, color=BLUE), run_time=.8)
        self.play(UpdateFromAlphaFunc(chosen, update_selected), run_time=2)
        self.wait(2)


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
        self.rk4_estimates()
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
        self.heading_rate()
        self.clear()
        self.velocity_change()
        self.clear()
        self.turning_summary()

    def heading_rate(self):
        self.heading("How quickly does the heading turn?")
        center = np.array([-3.4, .35, 0])
        radius = 1.9
        angle = ValueTracker(.02)
        path = Circle(radius=radius).move_to(center).set_stroke(WALL, 2)
        start = center + DOWN*radius
        original_radius = Line(center, start).set_stroke(WALL, 2)
        radial = Line(center, start).set_stroke(BLUE, 2)
        traveled = Arc(start_angle=-PI/2, angle=.02, radius=radius).shift(center).set_stroke(BLUE, 5)
        angle_arc = Arc(start_angle=-PI/2, angle=.02, radius=.55).shift(center).set_stroke(YELLOW, 3)
        racer = car(start, scale=.85)
        velocity = Arrow(start, start+RIGHT, buff=0, fill_color=BLUE)
        ghost = racer.copy().fade(.7)
        delta_label = math_label(r"\Delta\psi", 29, YELLOW)
        distance_label = math_label(r"\Delta s", 29, BLUE)
        radius_label = math_label("R", 29, MUTED).move_to(center+[-.3, -.95, 0])
        self.play(ShowCreation(path), FadeIn(racer), ShowCreation(original_radius), Write(radius_label))
        self.add(ghost)
        definition = math_label(r"\psi=\text{heading angle}", 35).move_to([3.2, 2.1, 0])
        self.play(Write(definition))

        def update(_):
            a = angle.get_value()
            p = center+radius*np.array([np.sin(a), -np.cos(a), 0])
            tangent = np.array([np.cos(a), np.sin(a), 0])
            racer.become(car(p, a, scale=.85))
            velocity.put_start_and_end_on(p, p+tangent)
            radial.put_start_and_end_on(center, p)
            traveled.become(Arc(start_angle=-PI/2, angle=a, radius=radius).shift(center).set_stroke(BLUE, 5))
            angle_arc.become(Arc(start_angle=-PI/2, angle=a, radius=.55).shift(center).set_stroke(YELLOW, 3))
            delta_label.move_to(center+.95*np.array([np.sin(a/2), -np.cos(a/2), 0]))
            distance_label.move_to(center+(radius+.35)*np.array([np.sin(a/2), -np.cos(a/2), 0]))

        driver = VMobject().add_updater(update)
        self.add(driver, radial, traveled, angle_arc, velocity)
        self.play(angle.animate.set_value(.9), run_time=3)
        self.play(Write(delta_label), Write(distance_label))
        caption = self.caption("The heading and radius turn through the same angle.")
        distance_eq = math_label(r"\Delta s=R\,\Delta\psi", 40).move_to([3.2, .85, 0])
        self.play(Write(distance_eq), run_time=2)
        self.wait(2)
        time_eq = math_label(r"{\Delta s\over\Delta t}=R\,{\Delta\psi\over\Delta t}", 40).move_to(distance_eq)
        self.play(TransformMatchingTex(distance_eq, time_eq), run_time=1.5)
        distance_eq = time_eq
        self.play(FadeOut(caption))
        caption = self.caption("Divide distance and heading change by the same elapsed time.")
        self.wait(2)
        rate_eq = math_label(r"v=R\dot\psi", 43, BLUE).move_to(distance_eq)
        self.play(TransformMatchingTex(distance_eq, rate_eq), run_time=1.5)
        distance_eq = rate_eq
        rate_definition = math_label(r"\dot\psi={d\psi\over dt}", 38, YELLOW).move_to([3.2, -.65, 0])
        meaning = words("heading change per second", 25, YELLOW).move_to([3.2, -1.5, 0])
        self.play(Write(rate_definition), Write(meaning))
        self.wait(3)
        solved = math_label(r"\dot\psi={v\over R}", 44, YELLOW).move_to(distance_eq)
        self.play(TransformMatchingTex(distance_eq, solved), run_time=1.5)
        self.play(angle.animate.set_value(1.45), run_time=2.5)
        self.wait(2)
        driver.clear_updaters()
        self.remove(driver)

    def velocity_change(self):
        self.heading("Turning changes the velocity, even at constant speed.")
        center = np.array([-3.6, .3, 0])
        radius = 1.8
        angle = ValueTracker(.8)
        start = center + DOWN*radius
        path = Circle(radius=radius).move_to(center).set_stroke(WALL, 2)
        initial = car(start, scale=.75).fade(.65)
        racer = car(start, scale=.75)
        old_velocity = Arrow(start, start+RIGHT*1.15, buff=0, fill_color=BLUE).fade(.45)
        new_velocity = Arrow(start, start+RIGHT*1.15, buff=0, fill_color=BLUE)
        self.play(ShowCreation(path), FadeIn(initial), FadeIn(racer), GrowArrow(old_velocity))
        final_position = center+radius*np.array([np.sin(.8), -np.cos(.8), 0])
        def move_around_circle(mob, alpha):
            a = .8*alpha
            point = center+radius*np.array([np.sin(a), -np.cos(a), 0])
            mob.become(car(point, a, scale=.75))
        self.play(UpdateFromAlphaFunc(racer, move_around_circle), run_time=2)
        new_velocity.put_start_and_end_on(final_position, final_position+1.15*np.array([np.cos(.8),np.sin(.8),0]))
        self.play(GrowArrow(new_velocity))
        origin = np.array([1.55, -1.1, 0])
        length = 2.8
        first = Arrow(origin, origin+RIGHT*length, buff=0, fill_color=BLUE).fade(.4)
        second = Arrow(origin, origin+length*np.array([np.cos(.8),np.sin(.8),0]), buff=0, fill_color=BLUE)
        self.play(TransformFromCopy(old_velocity, first), TransformFromCopy(new_velocity, second), run_time=2)
        caption = self.caption("Put the two velocity arrows at the same starting point.")
        difference = Arrow(first.get_end(), second.get_end(), buff=0, fill_color=YELLOW)
        arc = Arc(start_angle=0, angle=.8, radius=length).shift(origin).set_stroke(YELLOW, 2, opacity=.45)
        small_arc = Arc(start_angle=0, angle=.8, radius=.65).shift(origin).set_stroke(YELLOW, 2)
        delta_label = math_label(r"\Delta\psi", 27, YELLOW).move_to(origin+[.85,.35,0])
        change_label = math_label(r"\Delta\mathbf v", 29, YELLOW).next_to(difference, RIGHT, buff=.15)
        self.play(GrowArrow(difference), ShowCreation(arc), ShowCreation(small_arc), Write(delta_label), Write(change_label))
        self.wait(2)
        formula = math_label(r"|\Delta\mathbf v|\approx v\,\Delta\psi", 36, YELLOW).move_to([3.1, 2.2, 0])
        self.play(Write(formula))
        self.play(FadeOut(caption))
        caption = self.caption("For a small turn, the tip moves about v × Δψ.")

        def update(_):
            a = angle.get_value()
            direction = np.array([np.cos(a), np.sin(a), 0])
            p = center+radius*np.array([np.sin(a), -np.cos(a), 0])
            racer.become(car(p, a, scale=.75))
            new_velocity.put_start_and_end_on(p, p+1.15*direction)
            second.put_start_and_end_on(origin, origin+length*direction)
            difference.put_start_and_end_on(origin+RIGHT*length, origin+length*direction)
            arc.become(Arc(start_angle=0, angle=a, radius=length).shift(origin).set_stroke(YELLOW, 2, opacity=.45))
            small_arc.become(Arc(start_angle=0, angle=a, radius=.65).shift(origin).set_stroke(YELLOW, 2))
            delta_label.move_to(origin+.98*np.array([np.cos(a/2),np.sin(a/2),0])+UP*.25)
            change_label.next_to(difference, RIGHT, buff=.2)

        driver = VMobject().add_updater(update)
        self.add(driver)
        self.play(angle.animate.set_value(.18), run_time=4)
        self.wait(2)
        divided = math_label(r"{|\Delta\mathbf v|\over\Delta t}\approx v\,{\Delta\psi\over\Delta t}", 35, YELLOW).move_to(formula)
        self.play(TransformMatchingTex(formula, divided), FadeOut(caption), run_time=2)
        formula = divided
        caption = self.caption("Acceleration is the change in velocity per second.")
        self.wait(2)
        limit = math_label(r"a_{\rm lateral}=v\dot\psi", 40, YELLOW).move_to(formula)
        self.play(FadeOut(caption))
        self.caption("Shrink the time interval to get the instantaneous acceleration.")
        self.play(TransformMatchingTex(formula, limit), angle.animate.set_value(.08), run_time=2)
        self.wait(2)
        driver.clear_updaters()
        self.remove(driver)
        self.clear()
        self.heading("The two expressions describe the same acceleration.")
        yaw = math_label(r"\dot\psi={v\over R}", 47, BLUE).move_to([0,1.4,0])
        acceleration = math_label(r"a_{\rm lateral}=v\dot\psi", 47, YELLOW).move_to([0,-.1,0])
        self.play(Write(yaw), Write(acceleration), run_time=2)
        substituted = math_label(r"a_{\rm lateral}=v\left({v\over R}\right)", 47, YELLOW).move_to(acceleration)
        self.play(TransformMatchingTex(acceleration, substituted), run_time=2)
        self.wait(2)
        acceleration = substituted
        result = math_label(r"a_{\rm lateral}={v^2\over R}", 50, YELLOW).move_to(acceleration)
        self.play(TransformMatchingTex(acceleration, result), run_time=1.5)
        self.caption("Speed × heading change per second = inward acceleration.")
        self.wait(3)

    def turning_summary(self):
        self.heading("Now double the speed on the same circle.")
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

    def rk4_estimates(self):
        self.heading("Sample the motion before committing to the step.")
        # Use an explicitly enlarged interval so the trial states are distinguishable.
        # The derivative and stage construction are the same as the six real substeps.
        state = np.array([0., 0., 0., 2.])
        h, delta, steer_rate = .45, .25, .15
        trials, slopes, result = rk4_step(state, delta, h, steer_rate)
        origin = np.array([-5.1, -1.6, 0])
        scale = 5.
        project = lambda s: origin + scale*np.array([s[0], s[1], 0])
        colors = [BLUE, YELLOW, "#C39BFF", GREEN]
        start = Dot(origin, radius=.075, fill_color=WHITE)
        initial = car(origin, scale=.85)
        self.play(FadeIn(initial), FadeIn(start))
        caption = self.caption("Enlarged interval: 0.45 s, so the trial states can be seen.")
        equations = VGroup(*[
            math_label(text, 31, color) for text, color in zip([
                r"k_1=f(s_n,\delta)",
                r"k_2=f(s_n+\tfrac h2 k_1,\delta_{\rm mid})",
                r"k_3=f(s_n+\tfrac h2 k_2,\delta_{\rm mid})",
                r"k_4=f(s_n+h k_3,\delta_{\rm end})",
            ], colors)
        ]).arrange(DOWN, aligned_edge=LEFT, buff=.5).move_to([3.4, .65, 0])
        arrows, dots, labels, predictions = VGroup(), VGroup(), VGroup(), VGroup()
        for i, (trial, slope, color) in enumerate(zip(trials, slopes, colors)):
            point = project(trial)
            dot = Dot(point, radius=.065, fill_color=color)
            direction = .62*np.array([slope[0], slope[1], 0])
            arrow = Arrow(point, point+direction, buff=0, fill_color=color)
            label = math_label(f"k_{i+1}", 27, color).next_to(arrow, UP, buff=.12)
            if i:
                prediction = DashedLine(origin, point).set_stroke(color, 1.5, opacity=.45)
                predictions.add(prediction)
                self.play(ShowCreation(prediction), TransformFromCopy(dots[-1], dot), run_time=1.4)
            else:
                self.play(FadeIn(dot))
            self.play(GrowArrow(arrow), Write(label), Write(equations[i]), run_time=1.6)
            arrows.add(arrow)
            dots.add(dot)
            labels.add(label)
            self.wait(1.5)
        self.wait(2)
        self.play(FadeOut(predictions), FadeOut(dots), FadeOut(labels), FadeOut(equations), FadeOut(caption))
        average = math_label(r"\Delta s={h\over6}(k_1+2k_2+2k_3+k_4)", 35).move_to([2.7, 2.55, 0])
        self.play(Write(average), run_time=2)
        self.caption("Combine the four estimates with weights 1, 2, 2, 1.")
        # Draw the position components of the weighted state increments head to tail.
        weighted = VGroup()
        tip = origin.copy()
        for i, (slope, weight, color) in enumerate(zip(slopes, [1, 2, 2, 1], colors)):
            increment = scale*h*weight/6*np.array([slope[0], slope[1], 0])
            arrow = Arrow(tip, tip+increment, buff=0, fill_color=color)
            weighted.add(arrow)
            self.play(TransformFromCopy(arrows[i], arrow), run_time=1.3)
            tip += increment
        self.play(FadeOut(arrows))
        endpoint = Dot(project(result), radius=.08, fill_color=WHITE)
        self.play(FadeIn(endpoint), Transform(initial, car(project(result), result[2], scale=.85)), run_time=2)
        updated = math_label(r"s_{n+1}=s_n+\Delta s", 35).move_to([2.7, .55, 0])
        self.play(Write(updated))
        self.wait(3)

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
        for _ in range(6):
            _, _, state = rk4_step(state, delta, h, steer_rate, accel)
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
        saved = [m for m in self.mobjects if m is not self.camera.frame]
        for mob in saved:
            mob.save_state()
        self.clear()
        self.offset_penalty()
        self.clear()
        for mob in saved:
            mob.restore()
        self.play(*[FadeIn(mob) for mob in saved])
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


    def offset_penalty(self):
        self.heading("Why square the distance from the centerline?")
        x, top = -4., 2.
        walls = VGroup(Line([-6, top, 0], [-1, top, 0]),
                       Line([-6, -top, 0], [-1, -top, 0])).set_stroke(WALL, 3)
        centerline = DashedLine([-6, 0, 0], [-1, 0, 0]).set_stroke(BLUE, 2)
        offset = ValueTracker(.6)
        racer = car([x, .6, 0])
        area = Square(side_length=.6).move_to([x+.3, .3, 0]).set_stroke(BLUE, 2).set_fill(BLUE, .25)
        distance = Line([x, 0, 0], [x, .6, 0]).set_stroke(YELLOW, 4)
        self.play(ShowCreation(walls), ShowCreation(centerline), FadeIn(racer))
        d_label = math_label("d", 31, YELLOW).move_to([x-.4, .3, 0])
        self.play(ShowCreation(distance), Write(d_label))
        self.play(GrowFromPoint(area, [x, 0, 0]), run_time=1.5)
        graph_origin = np.array([3.25, -1.25, 0])
        def gp(d):
            return graph_origin + np.array([1.55*d, 1.15*d*d, 0])
        axes = VGroup(Line(graph_origin+LEFT*2.5, graph_origin+RIGHT*2.5),
                      Line(graph_origin, graph_origin+UP*2.65)).set_stroke(WALL, 1.5)
        ticks = VGroup(*[words(str(d), 20, MUTED).move_to(graph_origin+[1.55*d, -.3, 0]) for d in [-1, 0, 1]])
        graph = curve([gp(d) for d in np.linspace(-1.45, 1.45, 120)], BLUE, 3)
        marker = Dot(gp(.6), radius=.07, fill_color=YELLOW)
        guide = DashedLine(graph_origin+RIGHT*(1.55*.6), gp(.6)).set_stroke(YELLOW, 1.5)
        formula = math_label(r"\text{penalty}=d^2", 38, BLUE).move_to([3.25, 2.25, 0])
        axis_label = words("offset", 22, MUTED).next_to(axes[0], RIGHT, buff=.15)
        self.play(ShowCreation(axes), Write(ticks), Write(axis_label))
        self.play(TransformFromCopy(area, formula), ShowCreation(graph), run_time=2)
        self.play(FadeIn(marker), ShowCreation(guide))
        label = words("offset", 24, YELLOW).move_to([2.25, -2.25, 0])
        number = DecimalNumber(.6, num_decimal_places=2, font_size=29).set_color(YELLOW).move_to([4.05, -2.25, 0])
        penalty_label = words("penalty", 24, BLUE).move_to([2.25, -2.8, 0])
        penalty_number = DecimalNumber(.36, num_decimal_places=2, font_size=29).set_color(BLUE).move_to([4.05, -2.8, 0])
        self.play(FadeIn(label), FadeIn(number), FadeIn(penalty_label), FadeIn(penalty_number))
        caption = self.caption("The square's area is the centerline penalty.")

        def update(_):
            d = offset.get_value()
            size = max(abs(d), 1e-5)
            racer.move_to([x, d, 0])
            area.become(Square(side_length=size).move_to([x+size/2, d/2, 0])
                        .set_stroke(BLUE, 2).set_fill(BLUE, .25))
            distance.put_start_and_end_on(np.array([x, 0, 0]), np.array([x, d+1e-6, 0]))
            d_label.move_to([x-.4, d/2, 0])
            marker.move_to(gp(d))
            guide.put_start_and_end_on(graph_origin+RIGHT*(1.55*d), gp(d)+UP*1e-6)
            number.set_value(d)
            penalty_number.set_value(d*d)

        driver = VMobject().add_updater(update)
        self.add(driver)
        self.wait(2)
        self.play(offset.animate.set_value(1.2), run_time=4)
        tiles = VGroup(Line([x+.6, 0, 0], [x+.6, 1.2, 0]),
                       Line([x, .6, 0], [x+1.2, .6, 0])).set_stroke(WHITE, 2)
        self.play(ShowCreation(tiles), FadeOut(caption))
        self.caption("Twice the offset gives four times the penalty.")
        self.wait(3)
        self.play(FadeOut(tiles))
        self.play(offset.animate.set_value(0), run_time=3)
        self.wait(1)
        self.play(offset.animate.set_value(-1.2), run_time=4)
        self.wait(3)
        driver.clear_updaters()
        self.remove(driver)


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
        fan_caption = self.caption("Each ray has its own loop. Different rays are independent.")
        self.wait(3)
        self.play(FadeOut(index), FadeOut(cells), FadeOut(labels), FadeOut(write), FadeOut(fan_caption))
        beams.set_stroke(BLUE, 1.6)
        plot_base = -1.45
        bar_x = np.linspace(.85, 5.55, len(beams))
        def profile(rays):
            return VGroup(*[
                Line([x, plot_base, 0], [x, plot_base+.85*ray.get_length(), 0]).set_stroke(BLUE, 5)
                for x, ray in zip(bar_x, rays)
            ])
        bars = profile(beams)
        baseline = Line([.65, plot_base, 0], [5.75, plot_base, 0]).set_stroke(WALL, 2)
        angle_labels = VGroup(*[
            math_label(text, 23, MUTED).move_to([x, plot_base-.35, 0])
            for x, text in [(bar_x[0],r"-135^\circ"),(bar_x[9],r"0^\circ"),(bar_x[-1],r"135^\circ")]
        ])
        range_label = words("distance along each ray", 26, BLUE).move_to([3.2, 2.3, 0])
        self.play(ShowCreation(baseline), Write(angle_labels), Write(range_label))
        self.play(LaggedStart(*[TransformFromCopy(ray, bar) for ray, bar in zip(beams, bars)],
                              lag_ratio=.08), run_time=4)
        self.caption("Each bar is one range measurement, ordered by beam angle.")
        self.wait(2)
        heading = ValueTracker(0.)
        def scan(angle):
            sensor_position = origin+.27*np.array([np.cos(angle), np.sin(angle), 0])
            rays = VGroup()
            for offset in np.linspace(-3*PI/4, 3*PI/4, 19):
                direction = np.array([np.cos(angle+offset), np.sin(angle+offset), 0])
                candidates = []
                for axis, low, high in [(0,-5.9,-.7),(1,-2,2)]:
                    if abs(direction[axis]) > 1e-8:
                        wall = high if direction[axis] > 0 else low
                        candidates.append((wall-sensor_position[axis])/direction[axis])
                rays.add(Line(sensor_position, sensor_position+min(candidates)*direction).set_stroke(BLUE, 1.6))
            return rays
        def update(_):
            rays = scan(heading.get_value())
            beams.become(rays)
            bars.become(profile(rays))
            racer.become(car(origin, heading.get_value()))
        driver = VMobject().add_updater(update)
        self.add(driver)
        self.play(heading.animate.set_value(.65), run_time=4)
        self.wait(1)
        self.play(heading.animate.set_value(-.35), run_time=5)
        self.wait(2)
        driver.clear_updaters()
        self.remove(driver)


class TheHandoff(FilmScene):
    def construct(self):
        self.work_grid()
        self.clear()
        self.shared_outputs()

    def work_grid(self):
        self.heading("Follow the work from cars to rays.")
        ys = [1.55, .6, -.35, -1.3]
        cars = VGroup(*[car([-5.7, y, 0], scale=.7) for y in ys])
        rows = VGroup(*[
            VGroup(*[Dot([-4.6+.5*j, y, 0], radius=.06, fill_color=WALL) for j in range(6)])
            for y in ys
        ])
        links = VGroup(*[
            Line(row[0].get_center(), row[-1].get_center()).set_stroke(WALL, 1.5)
            for row in rows
        ])
        pulses = VGroup(*[Dot(c.get_center(), radius=.09, fill_color=YELLOW) for c in cars])
        physics_label = words("six updates per car", 28, YELLOW).move_to([-3.7, 2.55, 0])
        ray_label = words("108 rays per car", 28, BLUE).move_to([3.1, 2.55, 0])
        self.play(FadeIn(cars), ShowCreation(links), FadeIn(rows), Write(physics_label))
        self.add(pulses)
        caption = self.caption("Each update uses the state produced by the previous one.")
        for j in range(6):
            self.play(*[pulse.animate.move_to(row[j]) for pulse, row in zip(pulses, rows)],
                      *[row[j].animate.set_color(YELLOW) for row in rows], run_time=.75)
        self.wait(2)
        barrier = DashedLine([-.75, -1.8, 0], [-.75, 2., 0]).set_stroke(WHITE, 1.5)
        self.play(ShowCreation(barrier), FadeOut(caption))
        caption = self.caption("Lidar starts after physics has written the new poses.")
        self.play(Write(ray_label))
        grid = VGroup(*[
            VGroup(*[Dot([.9+.43*j, y, 0], radius=.055, fill_color=BLUE) for j in range(10)])
            for y in ys
        ])
        # Each completed car state supplies all of that car's independent ray items.
        for j in range(10):
            self.play(*[TransformFromCopy(pulse, row[j]) for pulse, row in zip(pulses, grid)], run_time=.22)
        sample = words("10 columns drawn / 108 computed", 22, MUTED).move_to([2.9, -2., 0])
        self.play(FadeIn(sample))
        self.wait(2)
        self.play(FadeOut(caption))
        selected = grid[2][6]
        horizontal = DashedLine([.3, ys[2], 0], selected.get_center()).set_stroke(YELLOW, 2)
        vertical = DashedLine([selected.get_x(), 2., 0], selected.get_center()).set_stroke(YELLOW, 2)
        i_label = math_label("i=2", 25, YELLOW).move_to([.2, ys[2]-.3, 0])
        j_label = math_label("j=6", 25, YELLOW).move_to([selected.get_x(), 2.15, 0])
        self.play(ShowCreation(horizontal), ShowCreation(vertical), Write(i_label), Write(j_label),
                  selected.animate.set_color(YELLOW))
        self.play(FlashAround(selected, color=YELLOW))
        destination = math_label(r"\texttt{obs}[2,\,8]=r_6", 35, YELLOW).move_to([2.9, -2.75, 0])
        self.play(TransformFromCopy(selected, destination), run_time=1.5)
        self.caption("The first two observation entries are steering and speed.")
        self.wait(3)

    def shared_outputs(self):
        self.heading("Where do the results go?")
        xs = [-2.9, -2.32, -1.74, -1.16, -.58, 0.]
        ys = [1.05, .4, -.25, -.9]
        obs = VGroup(*[
            VGroup(*[Rectangle(width=.53, height=.53).move_to([x, y, 0]).set_stroke(WALL, 1.4)
                     for x in xs]) for y in ys
        ])
        reward = VGroup(*[Rectangle(width=.8, height=.53).move_to([1.25, y, 0]).set_stroke(WALL, 1.4) for y in ys])
        done = VGroup(*[Rectangle(width=.65, height=.53).move_to([2.6, y, 0]).set_stroke(WALL, 1.4) for y in ys])
        frame = Rectangle(width=7, height=3.25).move_to([-.15, .3, 0]).set_stroke(GREEN, 2)
        headings = VGroup(words("observation", 26, GREEN).move_to([-1.45, 2.3, 0]),
                          words("reward", 26, GREEN).move_to([1.25, 2.3, 0]),
                          words("done", 26, GREEN).move_to([2.6, 2.3, 0]))
        columns = VGroup(*[math_label(t, 22, MUTED).move_to([x, 1.65, 0])
                          for x, t in zip(xs, [r"\delta", "v", "r_0", "r_1", r"\cdots", "r_{107}"])])
        self.play(ShowCreation(frame), FadeIn(obs), FadeIn(reward), FadeIn(done), Write(headings), Write(columns))
        caption = self.caption("Four environments shown; observation columns abbreviated.")
        warp = words("Warp", 32, YELLOW).move_to([-5.2, .8, 0])
        writes = words("writes", 24, MUTED).next_to(warp, DOWN, buff=.15)
        warp_pointer = Arrow([-4.5, .45, 0], [-3.68, .45, 0], buff=0, fill_color=YELLOW)
        self.play(Write(warp), FadeIn(writes), GrowArrow(warp_pointer))
        # Illustrative buffer contents. The terminal row contains a respawned pose.
        values = [
            ["0.1", "2.0", "3.1", "2.4", r"\cdots", "4.2"],
            ["-0.2", "1.5", "2.8", "3.0", r"\cdots", "1.6"],
            ["0", "0", "2.5", "1.9", r"\cdots", "3.2"],
            ["0.2", "3.0", "1.7", "2.1", r"\cdots", "2.9"],
        ]
        entries = VGroup(*[VGroup(*[math_label(value, 20, GREEN).move_to(cell)
                                    for value, cell in zip(row, cells)]) for row, cells in zip(values, obs)])
        reward_values = VGroup(*[math_label(value, 23, GREEN if value != "-25" else RED).move_to(cell)
                                 for value, cell in zip(["0.3", "0.1", "-25", "0.4"], reward)])
        done_values = VGroup(*[math_label(str(value), 23, RED if value else GREEN).move_to(cell)
                               for value, cell in zip([0, 0, 1, 0], done)])
        stages = VGroup(words("physics", 27, YELLOW), words("lidar", 27, BLUE), words("RNG tick", 27, MUTED))
        stages.arrange(RIGHT, buff=1.2).move_to([0, -2.15, 0])
        self.play(FadeIn(stages))
        physics_cells = VGroup(*[cell for row in obs for cell in row[:2]], *reward, *done)
        physics_values = VGroup(*[value for row in entries for value in row[:2]], *reward_values, *done_values)
        self.play(physics_cells.animate.set_stroke(YELLOW, 2.5), FlashAround(stages[0], color=YELLOW))
        self.play(LaggedStartMap(FadeIn, physics_values, lag_ratio=.06), run_time=2)
        self.wait(2)
        lidar_cells = VGroup(*[cell for row in obs for cell in row[2:]])
        lidar_values = VGroup(*[value for row in entries for value in row[2:]])
        self.play(lidar_cells.animate.set_stroke(BLUE, 2.5), FlashAround(stages[1], color=BLUE))
        self.play(LaggedStartMap(FadeIn, lidar_values, lag_ratio=.05), run_time=2)
        self.wait(2)
        self.play(FlashAround(stages[2], color=WHITE))
        self.play(FadeOut(caption))
        caption = self.caption("The launches run in order on Torch's current CUDA stream.")
        torch = words("Torch", 32, GREEN).move_to([5.05, .8, 0])
        views = words("views", 24, MUTED).next_to(torch, DOWN, buff=.15)
        torch_pointer = Arrow([4.35, .45, 0], [3.4, .45, 0], buff=0, fill_color=GREEN)
        self.play(Write(torch), FadeIn(views), GrowArrow(torch_pointer), run_time=1.5)
        self.play(Indicate(frame, color=GREEN), run_time=1.5)
        self.wait(2)
        self.play(FadeOut(stages), FadeOut(caption))
        shared = words("Two views of the same output storage.", 30, GREEN).move_to([0, -2.15, 0])
        self.play(Write(shared))
        self.caption("When Warp and Torch use the same CUDA device.")
        self.wait(4)
