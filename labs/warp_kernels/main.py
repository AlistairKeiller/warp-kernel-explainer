"""One simulation step of warporacer, told as five silent ManimGL chapters.

Render one chapter:    manimgl labs/warp_kernels/main.py VehicleStep -w
Render the whole film:  python labs/warp_kernels/render.py
Build the click-through deck:  python labs/warp_kernels/render.py --slides

Every drawing is original. storyboard.md records what each picture simplifies
and where the matching code lives in warporacer/sim.py.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import textwrap

from manimlib import *

# Palette: yellow is the car and local physics, blue is motion and measurement,
# green is what the simulator applies, red is a request it cannot honor.
INK = "#0B0B10"
BLUE = "#58C4DD"
YELLOW = "#FFFF00"
GREEN = "#83C167"
RED = "#FC6255"
PURPLE = "#C39BFF"
MUTED = "#B4B4C0"
WALL = "#797985"

# Layout: diagrams live on the left, equations in a column on the right.
PHASES = ["Move", "Score", "Sense", "Return"]
GUIDE_Y = 3.72
HEADING_Y = 3.08
CAPTION_Y = -3.47
RIGHT_COLUMN = 3.2 * RIGHT

# A silent film needs time to read each caption before the picture changes.
# In slideshow mode the presenter sets the pace, so pauses collapse and every
# caption starts a new slide.
READ_LEAD = .7
WORDS_PER_SECOND = 3.1
SLIDESHOW = os.environ.get("WARP_SLIDES") == "1"
SLIDE_PAUSE = .35

# Simulator constants mirrored from warporacer/sim.py.
WHEELBASE = .3302
MU = 1.0489
G = 9.81
SUBSTEPS = 6
DT_SUB = 1 / 60 / SUBSTEPS


def words(text, size=30, color=WHITE):
    return Text(text, font="Arial", font_size=size).set_color(color)


def formula(text, size=36, color=WHITE):
    return Tex(text, font_size=size).set_color(color)


def get_curve(points, color=BLUE, width=3):
    return VMobject().set_points_smoothly(points).set_stroke(color, width)


def get_track():
    """Two walls and a faint centerline, drawn as nested ellipses."""
    return VGroup(
        Ellipse(width=10, height=5.4).set_stroke(WALL, 3),
        Ellipse(width=7, height=2.4).set_stroke(WALL, 3),
        Ellipse(width=8.5, height=3.9).set_stroke(BLUE, 1.5, opacity=.35),
    )


def track_pose(theta, road):
    """A point on the track's centerline and the counterclockwise heading there."""
    a, b = road[2].get_width() / 2, road[2].get_height() / 2
    point = road.get_center() + np.array([a * np.cos(theta), b * np.sin(theta), 0])
    return point, np.arctan2(b * np.cos(theta), -a * np.sin(theta))


class Car(VGroup):
    """A top-down car whose pose can be updated in place."""

    def __init__(self, point=ORIGIN, heading=0., color=YELLOW, scale=1.):
        body = RoundedRectangle(width=.65, height=.32, corner_radius=.07)
        body.set_fill(color, 1).set_stroke(color, 1)
        window = Rectangle(width=.18, height=.23).set_fill(INK, 1).set_stroke(width=0).shift(.08 * RIGHT)
        wheels = VGroup(
            Line([x - .065, y, 0], [x + .065, y, 0]).set_stroke(WHITE, 3)
            for x in (-.2, .2) for y in (-.19, .19)
        )
        super().__init__(body, window, wheels)
        self.heading = 0.
        self.scale(scale)
        self.pose(point, heading)

    def pose(self, point, heading):
        self.rotate(heading - self.heading).move_to(point)
        self.heading = heading
        return self


def derivative(state, delta, accel=1., wheelbase=WHEELBASE, mu=MU):
    """sim.py's bicycle derivative for state (x, y, psi, v), with the same grip caps."""
    limit = mu * G
    cap = limit / max(abs(state[3]), .5)
    yaw = np.clip(state[3] * np.tan(delta) / wheelbase, -cap, cap)
    remaining = np.sqrt(max(limit * limit - (state[3] * yaw) ** 2, 0.))
    return np.array([
        state[3] * np.cos(state[2]), state[3] * np.sin(state[2]),
        yaw, np.clip(accel, -remaining, remaining),
    ])


def rk4_step(state, delta, h, steer_rate, accel=1.):
    """One RK4 update as sim.py performs it: trial states, their slopes, and the result."""
    k1 = derivative(state, delta, accel)
    s2 = state + h * k1 / 2
    k2 = derivative(s2, delta + steer_rate * h / 2, accel)
    s3 = state + h * k2 / 2
    k3 = derivative(s3, delta + steer_rate * h / 2, accel)
    s4 = state + h * k3
    k4 = derivative(s4, delta + steer_rate * h, accel)
    return [state.copy(), s2, s3, s4], [k1, k2, k3, k4], state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


class FilmScene(Scene):
    """Shared stagecraft for the chapters.

    A guide strip names the four phases of a step. `heading` asks one question
    at a time, `caption` explains the current picture and is held until it can
    be read, and `bridge` carries a section's result into the next question.
    """

    phase = 0

    def setup(self):
        super().setup()
        self.camera.background_rgba = color_to_rgba(INK)
        self.heading_mob = None
        self.caption_mob = None
        self.caption_due = 0.
        self.pending_wipe = False
        self.beats = []
        self.captions = []
        self.slides = [0.]
        self.phases = VGroup(
            words(name, 20, YELLOW if i == self.phase else WALL)
            for i, name in enumerate(PHASES)
        ).arrange(RIGHT, buff=.65).move_to([3.6, GUIDE_Y, 0])
        self.guide = VGroup(words("ONE SIMULATION STEP", 18, MUTED).move_to([-4.85, GUIDE_Y, 0]), self.phases)
        self.add(self.guide)

    @contextmanager
    def live(self, update):
        """Run `update` every frame. It also runs once up front, so nothing jumps when motion starts."""
        driver = VMobject().add_updater(lambda mob: update())
        self.add(driver)
        try:
            yield
        finally:
            self.remove(driver)

    def get_stage(self, *keep):
        """Everything on screen except the guide, the heading, and `keep`."""
        persistent = {self.camera.frame, self.guide, self.heading_mob, *keep}
        return [mob for mob in self.mobjects if mob not in persistent]

    def wait(self, duration=None, **kwargs):
        if SLIDESHOW and duration is not None:
            duration = min(duration, SLIDE_PAUSE)
        super().wait(duration, **kwargs)

    def let_read(self):
        """Hold the current caption until it has had time to be read."""
        remaining = self.caption_due - self.time
        if remaining > 0:
            self.wait(remaining)

    def heading(self, text, *animations, wipe=False):
        """Pose the next question. With `wipe`, the stage fades out in the same motion."""
        mob = words(text, 36).set_max_width(12.7).move_to(HEADING_Y * UP)
        self.beats.append({"title": text, "time": round(self.time, 3)})
        animations = list(animations)
        if wipe or self.pending_wipe:
            self.let_read()
            animations += [FadeOut(mob) for mob in self.get_stage()]
            self.caption_mob = None
            self.pending_wipe = False
        if self.heading_mob in self.mobjects:
            animations.append(FadeTransform(self.heading_mob, mob))
        else:
            animations.append(FadeIn(mob, shift=.15 * DOWN))
        self.play(*animations, run_time=.7)
        self.heading_mob = mob
        return mob

    def caption(self, text, color=MUTED):
        mob = words(textwrap.fill(text, 78), 28, color).set_max_width(12.6).move_to(CAPTION_Y * UP)
        self.let_read()
        self.slides.append(round(self.time, 3))
        if self.caption_mob in self.mobjects:
            self.play(LaggedStart(FadeOut(self.caption_mob), FadeIn(mob), lag_ratio=.6), run_time=.8)
        else:
            self.play(FadeIn(mob), run_time=.5)
        self.caption_mob = mob
        self.caption_due = self.time + READ_LEAD + len(text.split()) / WORDS_PER_SECOND
        self.captions.append({"text": text, "time": round(self.time, 3)})
        return mob

    def bridge(self, result, connection, question):
        """Keep a section's result on screen, say what it means, then ask the next question."""
        self.let_read()
        statement = words(connection, 32).set_max_width(12).move_to(1.8 * UP)
        clearing = [FadeOut(mob) for mob in self.get_stage(result)]
        if result is not None:
            clearing.append(result.animate.set_max_width(10).move_to(.65 * UP))
        entrance = FadeIn(statement, shift=.15 * UP)
        if clearing:
            self.play(LaggedStart(AnimationGroup(*clearing), entrance, lag_ratio=.75), run_time=1.4)
        else:
            self.play(entrance, run_time=.65)
        self.caption_mob = None
        self.caption(question, WHITE)
        self.wait(3)
        self.pending_wipe = True

    def tear_down(self):
        if self.file_writer.write_to_movie:
            self.let_read()
            path = Path(self.file_writer.get_movie_file_path()).with_suffix(".beats.json")
            path.write_text(json.dumps({
                "scene": str(self), "duration": round(self.time, 3),
                "beats": self.beats, "captions": self.captions, "slides": self.slides,
            }, indent=2) + "\n")
        super().tear_down()


class Opening(FilmScene):
    """One car, its state and action, and the batch of worlds a kernel runs over."""

    def construct(self):
        self.heading("What happens between two frames of a racing simulation?")
        road = get_track().scale(.85)
        theta = ValueTracker(-PI / 2)
        racer = Car(*track_pose(theta.get_value(), road), scale=.85)
        self.play(ShowCreation(road), FadeIn(racer), run_time=1.5)
        self.caption("Start with one car. We want to predict what happens next.")
        with self.live(lambda: racer.pose(*track_pose(theta.get_value(), road))):
            self.play(theta.animate.set_value(.1), run_time=4)
        world = VGroup(road, racer)
        self.play(world.animate.scale(.68).move_to([-3.45, .2, 0]), run_time=1.5)

        known = VGroup(
            words("What we know", 30, YELLOW),
            words("position and heading\nspeed and wheel angle", 27),
        ).arrange(DOWN, buff=.3)
        asked = VGroup(
            words("What we ask for", 30, BLUE),
            words("turn the steering wheel\naccelerate or brake", 27),
        ).arrange(DOWN, buff=.3)
        panel = VGroup(known, asked).arrange(DOWN, buff=.8).move_to(RIGHT_COLUMN)
        self.play(FadeIn(known), run_time=1.5)
        self.caption("The state describes the car now: where it is and how it is moving.")
        self.wait(3)
        self.play(FadeIn(asked), run_time=1.5)
        self.caption("An action asks the wheel to turn and the car to speed up or slow down.")
        self.wait(3)
        stages = VGroup(
            words(text, 29) for text in ["Move the car", "Score the move", "Measure the view"]
        ).arrange(DOWN, aligned_edge=LEFT, buff=.65).move_to(RIGHT_COLUMN)
        self.play(LaggedStart(
            FadeOut(panel), LaggedStartMap(FadeIn, stages, shift=.2 * RIGHT, lag_ratio=.35), lag_ratio=.7,
        ), run_time=2.5)
        self.caption("One step moves the car, scores that move, then measures what the car sees.")
        self.wait(3.5)

        self.heading("Now make thousands of separate little worlds.")
        self.caption("The cars do not interact, so every world can compute its own next step.")
        worlds = VGroup(world.copy().scale(.27 / .68) for _ in range(12))
        worlds.arrange_in_grid(3, 4, h_buff=.8, v_buff=.35).move_to(.35 * DOWN)
        self.play(FadeOut(stages), ReplacementTransform(world, worlds[0]), run_time=1.5)
        self.play(LaggedStart(*(FadeIn(w, scale=.9) for w in worlds[1:]), lag_ratio=.08), run_time=2)
        indices = VGroup(formula(f"i={i}", 23, BLUE).next_to(w, DOWN, buff=.07) for i, w in enumerate(worlds))
        self.play(Write(indices))
        self.caption("Twelve worlds stand for a batch of 8,000 independent environments.", BLUE)
        self.wait(2.5)

        def advance(group, alpha):
            # Slightly different speeds make the independence visible.
            for i, (road, racer) in enumerate(group):
                racer.pose(*track_pose(.1 + .65 * (1 + .08 * i) * alpha, road))

        self.play(UpdateFromAlphaFunc(worlds, advance), run_time=3)
        chosen = worlds[5]
        others = VGroup(*worlds[:5], *worlds[6:])
        other_indices = VGroup(*indices[:5], *indices[6:])
        self.play(others.animate.fade(.65), other_indices.animate.set_opacity(.35))
        self.wait(1)
        self.play(FadeOut(others), FadeOut(indices), run_time=1)
        self.play(chosen.animate.scale(2.8).move_to([-2.5, 0, 0]), run_time=2)

        self.heading("A kernel applies the same program to many work items.")
        kernel = VGroup(
            words("One work item = one car", 30, YELLOW),
            formula(R"\texttt{step\_kernel}(i=5)", 30, MUTED),
            words("state + action → next state", 27),
        ).arrange(DOWN, buff=.55).move_to(RIGHT_COLUMN + .5 * UP)
        self.play(FadeIn(kernel[0]), Write(kernel[1]), FadeIn(kernel[2]), run_time=2)
        self.caption("Warp spreads these work items over the hardware. A work item is a task, not a GPU core.")
        road, racer = chosen
        self.add(racer.copy().fade(.75), racer)
        self.play(UpdateFromAlphaFunc(racer, lambda m, a: m.pose(*track_pose(1.01 + .45 * a, road))), run_time=2)
        self.wait(2.5)
        self.caption("Let's follow this one car through Move, Score, Sense, and Return.", WHITE)
        self.wait(3.5)


class VehicleStep(FilmScene):
    """Move: velocity, steering, the grip budget, and RK4 integration."""

    def construct(self):
        result = self.velocity()
        self.bridge(result, "Velocity tells us how position changes.",
                    "How does steering change the direction of that velocity?")
        result = self.steering()
        self.bridge(result, "Wheel angle and speed set how fast the heading turns.",
                    "Changing direction takes acceleration. How much?")
        result = self.turning_acceleration()
        self.bridge(result, "Twice the speed needs four times the turning acceleration.",
                    "Can the tires always provide that much?")
        result = self.grip_circle()
        self.bridge(result, "Turning and speeding up share one grip budget.",
                    "What does that cap do to the path?")
        result = self.capped_turn()
        self.bridge(result, "We now know which motion the tires allow.",
                    "How do we turn these rates of change into a new state?")
        result = self.rk4()
        self.bridge(result, "Four trial estimates make one update.",
                    "Repeat that small update six times to advance one frame.")
        self.substeps()
        self.caption("Our car has a new state. Next: was that move useful?", WHITE)
        self.wait(3)

    def velocity(self):
        self.heading("Where does the next position come from?")
        origin = np.array([-4.5, -1.2, 0])
        heading, speed = ValueTracker(.35), ValueTracker(3.)
        tip = lambda: origin + speed.get_value() * rotate_vector(RIGHT, heading.get_value())
        foot = lambda: np.array([tip()[0], origin[1], 0])
        racer = Car(origin, heading.get_value(), scale=1.3)
        baseline = DashedLine(origin + .8 * LEFT, origin + 4.5 * RIGHT).set_stroke(WALL, 1.5)
        velocity = Arrow(origin, tip(), buff=0, fill_color=BLUE)
        horizontal = Line(origin, foot()).set_stroke(BLUE, 4)
        vertical = Line(foot(), tip()).set_stroke(YELLOW, 4)
        angle = Arc(0, heading.get_value(), radius=1, arc_center=origin).set_stroke(WHITE, 2)
        psi = formula(R"\psi", 30)
        v_label = formula("v", 36, BLUE)

        def update():
            racer.pose(origin, heading.get_value())
            velocity.put_start_and_end_on(origin, tip())
            horizontal.put_start_and_end_on(origin, foot())
            vertical.put_start_and_end_on(foot(), tip())
            angle.become(Arc(0, heading.get_value(), radius=1, arc_center=origin).set_stroke(WHITE, 2))
            psi.move_to(origin + 1.25 * rotate_vector(RIGHT, heading.get_value() / 2))
            v_label.next_to(velocity, UP, buff=.15)

        update()
        self.play(ShowCreation(baseline), FadeIn(racer))
        self.caption("The velocity arrow points where the car is going. Its length is the speed.")
        self.play(GrowArrow(velocity), Write(v_label), run_time=1.5)
        self.wait(1.5)
        self.caption("Split it into horizontal and vertical parts: how fast x and y change.")
        self.play(ShowCreation(horizontal), ShowCreation(vertical), run_time=2)
        equations = VGroup(
            formula(R"\dot x=v\cos\psi", 42, BLUE),
            formula(R"\dot y=v\sin\psi", 42, YELLOW),
        ).arrange(DOWN, buff=.6).move_to(RIGHT_COLUMN + .3 * UP)
        self.play(ShowCreation(angle), Write(psi), TransformFromCopy(horizontal, equations[0]), run_time=1.5)
        self.play(TransformFromCopy(vertical, equations[1]), run_time=1.5)
        self.caption("v is speed, ψ is heading, and a dot means change per second.")
        self.wait(2.5)
        with self.live(update):
            self.caption("Turn the arrow: both parts change together.")
            self.play(heading.animate.set_value(1.05), run_time=3)
            self.play(heading.animate.set_value(.5), run_time=2)
            self.caption("Keep the direction and go faster: both parts grow.")
            self.play(speed.animate.set_value(3.9), run_time=2)
            self.wait(1.5)
        self.play(FadeOut(VGroup(angle, psi, v_label, horizontal, vertical, velocity)))
        direction = rotate_vector(RIGHT, .5)
        displacement = Arrow(origin, origin + 2.4 * direction, buff=0, fill_color=BLUE)
        self.add(racer.copy().fade(.75), racer)
        self.play(ShowCreation(displacement), racer.animate.shift(2.4 * direction), run_time=2.5)
        approximation = formula(R"\text{displacement}\approx\text{velocity}\times\text{time}", 32, BLUE)
        approximation.set_max_width(6).next_to(equations, DOWN, buff=.9)
        self.play(Write(approximation))
        self.caption("Over a short time the car moves about velocity × time, as long as the direction barely changes.")
        self.wait(3)
        return approximation

    def steering(self):
        self.heading("How does the wheel angle set the turn?")
        rear = np.array([-4.4, -1.5, 0])
        wheelbase = 2.4
        front = rear + wheelbase * RIGHT
        steering = ValueTracker(.7)
        center = lambda: rear + UP * wheelbase / np.tan(steering.get_value())
        chassis = Line(rear, front).set_stroke(WALL, 7)
        rear_wheel = Line(rear + .35 * LEFT, rear + .35 * RIGHT).set_stroke(WHITE, 12)
        front_wheel = Line(front + .35 * LEFT, front + .35 * RIGHT).set_stroke(YELLOW, 12)
        base_direction = DashedLine(front, front + 1.15 * RIGHT).set_stroke(WALL, 1.5)
        delta_arc = Arc(0, steering.get_value(), radius=.8, arc_center=front).set_stroke(YELLOW, 2)
        delta_label = formula(R"\delta", 30, YELLOW).move_to(front + 1.08 * rotate_vector(RIGHT, steering.get_value() / 2))
        l_brace = Brace(chassis, DOWN, buff=.3)
        l_label = formula("L", 32, BLUE).next_to(l_brace, DOWN, buff=.12)
        rear_normal = Line(rear, center()).set_stroke(BLUE, 2)
        front_normal = Line(front, center()).set_stroke(YELLOW, 2)
        center_dot = Dot(center(), radius=.065, fill_color=GREEN)
        center_label = words("turn center", 23, GREEN).next_to(center_dot, UP, buff=.17)
        r_label = formula("R", 32, BLUE).move_to((rear + center()) / 2 + .35 * LEFT)
        path = Arc(-PI / 2, 1.15, radius=center()[1] - rear[1], arc_center=center()).set_stroke(GREEN, 3)

        def update():
            delta, c = steering.get_value(), center()
            tangent = rotate_vector(RIGHT, delta)
            front_wheel.put_start_and_end_on(front - .35 * tangent, front + .35 * tangent)
            rear_normal.put_start_and_end_on(rear, c)
            front_normal.put_start_and_end_on(front, c)
            center_dot.move_to(c)
            center_label.next_to(center_dot, UP, buff=.17)
            r_label.move_to((rear + c) / 2 + .35 * LEFT)
            path.become(Arc(-PI / 2, 1.15, radius=c[1] - rear[1], arc_center=c).set_stroke(GREEN, 3))
            delta_arc.become(Arc(0, delta, radius=.8, arc_center=front).set_stroke(YELLOW, 2))
            delta_label.move_to(front + 1.08 * rotate_vector(RIGHT, delta / 2))

        self.play(ShowCreation(chassis), FadeIn(rear_wheel), FadeIn(front_wheel))
        self.play(GrowFromCenter(l_brace), Write(l_label))
        self.play(Rotate(front_wheel, steering.get_value(), about_point=front), ShowCreation(base_direction),
                  ShowCreation(delta_arc), Write(delta_label), run_time=2)
        self.caption("Each wheel rolls straight ahead, so its turn center lies on a line perpendicular to it.")
        self.play(ShowCreation(rear_normal), ShowCreation(front_normal), run_time=2)
        self.play(FadeIn(center_dot), Write(center_label), Write(r_label))
        self.play(ShowCreation(path), run_time=2)
        equation = formula(R"\tan\delta={L\over R}", 44).move_to(RIGHT_COLUMN + 1.9 * UP)
        radius = formula(R"R={L\over\tan\delta}", 44, GREEN).next_to(equation, DOWN, buff=.45)
        glossary = words("δ = wheel angle\nL = distance between the wheels\nR = radius of the turn", 23, MUTED)
        glossary.next_to(radius, DOWN, buff=.4)
        self.play(Write(equation), FadeIn(glossary), run_time=2)
        self.play(TransformFromCopy(equation, radius), run_time=2)
        self.caption("The wheelbase and the radius form a right triangle, so R = L / tan δ.")
        self.wait(2)
        with self.live(update):
            self.caption("Turn the wheel further: the turn center moves closer and the circle tightens.")
            self.play(steering.animate.set_value(.93), run_time=4)
            self.wait(1.5)
        rate = formula(R"\dot\psi={v\over R}", 44, YELLOW).next_to(glossary, DOWN, buff=.4)
        c = center()
        rider = Car(c + (c[1] - rear[1]) * DOWN, 0, scale=.6)
        self.caption("Drive that circle at speed v and the heading turns at v/R radians per second.")
        self.play(Write(rate), FadeIn(rider), run_time=1.5)
        self.play(UpdateFromAlphaFunc(
            rider, lambda m, a: m.pose(c + (c[1] - rear[1]) * rotate_vector(DOWN, 1.15 * a), 1.15 * a),
        ), run_time=3)
        self.wait(1)
        request = formula(R"\dot\psi={v\tan\delta\over L}", 44, YELLOW).move_to(rate)
        self.caption("Substitute R. This is the heading rate the kernel requests from wheel angle and speed.")
        self.play(TransformMatchingTex(rate, request), run_time=1.5)
        self.wait(2.5)
        return request

    def turning_acceleration(self):
        self.heading("Why does turning need acceleration?")
        center = np.array([-3.3, .2, 0])
        radius = 1.9
        theta, speed = ValueTracker(-PI / 2), ValueTracker(1.)
        point = lambda: center + radius * rotate_vector(RIGHT, theta.get_value())
        path = Circle(radius=radius).move_to(center).set_stroke(WALL, 2)
        racer = Car(point(), theta.get_value() + PI / 2, scale=.85)
        radial = DashedLine(center, point()).set_stroke(WALL, 1.5)
        velocity = Arrow(point(), point() + .8 * rotate_vector(UP, theta.get_value()), buff=0, fill_color=BLUE)
        lateral = Arrow(point(), center, buff=0, fill_color=YELLOW)
        r_label = formula("R", 32, MUTED).move_to(center + [-.25, -.95, 0])

        def update():
            angle, v = theta.get_value(), speed.get_value()
            normal, tangent = rotate_vector(RIGHT, angle), rotate_vector(UP, angle)
            racer.pose(point(), angle + PI / 2)
            velocity.put_start_and_end_on(point(), point() + .8 * v * tangent)
            lateral.put_start_and_end_on(point(), point() - .35 * v * v * normal)
            radial.put_start_and_end_on(center, point())

        self.play(ShowCreation(path), FadeIn(racer), ShowCreation(radial), Write(r_label))
        self.play(GrowArrow(velocity))
        self.caption("Drive around the circle at a steady speed. The velocity arrow keeps changing direction.")
        with self.live(update):
            self.play(theta.animate.set_value(-.1), run_time=3.5)
            self.caption("Changing velocity is acceleration, even at constant speed. Here it points toward the center.")
            self.play(FadeIn(lateral))
            acceleration = formula(R"a_{\rm lateral}=v\,\dot\psi", 42, YELLOW).move_to(RIGHT_COLUMN + 1.4 * UP)
            self.play(Write(acceleration), run_time=1.5)
            self.caption("A faster heading change, or more speed, means more inward acceleration: a = v·ψ̇.")
            self.play(theta.animate.set_value(1.4), run_time=3.5)
            substituted = formula(R"a_{\rm lateral}={v^2\over R}=v\,\dot\psi", 42, YELLOW).move_to(acceleration)
            self.caption("With ψ̇ = v/R, that is also v² over R.")
            self.play(TransformMatchingTex(acceleration, substituted), run_time=1.5)
            self.wait(2)
            self.heading("Now double the speed on the same circle.")
            ratio = VGroup(
                formula(R"v\ \longrightarrow\ 2v", 34, BLUE),
                formula(R"a_{\rm lateral}\ \longrightarrow\ 4a_{\rm lateral}", 34, YELLOW),
            ).arrange(DOWN, buff=.3).next_to(substituted, DOWN, buff=1.1)
            self.play(Write(ratio), speed.animate.set_value(2), run_time=3)
            self.caption("Double the speed: the blue arrow doubles, but the yellow arrow grows fourfold.")
            self.play(theta.animate.set_value(4.5), run_time=4)
            self.wait(1.5)
        return substituted

    def grip_circle(self):
        self.heading("How much acceleration can the tires provide?")
        origin = np.array([-3.4, -.1, 0])
        unit = 2.05
        request_x, request_y, grip = ValueTracker(.65), ValueTracker(.45), ValueTracker(1.)
        axes = VGroup(
            Line(origin + 2.65 * LEFT, origin + 3.15 * RIGHT),
            Line(origin + 2.6 * DOWN, origin + 2.6 * UP),
        ).set_stroke(WALL, 1.5)
        axis_labels = VGroup(
            words("accelerate", 22, BLUE).move_to(origin + [2.1, -.42, 0]),
            words("brake", 22, BLUE).move_to(origin + [-2, -.42, 0]),
            words("turn", 24, YELLOW).move_to(origin + [0, 2.85, 0]),
        )
        self.play(ShowCreation(axes), Write(axis_labels))
        self.caption("A new picture. Up and down is turning; left and right is braking and throttle.")
        self.wait(2)
        limit = formula(R"|a|\leq\mu g", 42, GREEN).move_to(RIGHT_COLUMN + 1.7 * UP)
        self.caption("Friction caps the tires' push at μg: μ is grip, g is gravity.")
        self.play(Write(limit))
        circle = Circle(radius=unit).move_to(origin).set_stroke(GREEN, 2.5).set_fill(GREEN, .035)
        radius_line = Line(origin, origin + unit * RIGHT).set_stroke(GREEN, 2)
        radius_label = formula(R"\mu g", 29, GREEN).move_to(origin + [1, -.25, 0])
        self.play(ShowCreation(radius_line), Write(radius_label))
        self.play(ShowCreation(circle), Rotate(radius_line, TAU, about_point=origin), run_time=4, rate_func=linear)
        self.caption("Every acceleration inside this circle is possible. The circle is the grip budget.")
        self.play(FadeOut(radius_line), FadeOut(radius_label))
        self.wait(2)
        relation = formula(R"a_{\rm long}^2+a_{\rm lateral}^2\leq(\mu g)^2", 35).move_to(limit)
        self.play(FadeTransform(limit, relation))

        def clipped():
            """Same order as deriv(): cap the turn, then fit throttle into what is left."""
            cap, x, y = grip.get_value(), request_x.get_value(), request_y.get_value()
            lateral = np.clip(y, -cap, cap)
            remaining = np.sqrt(max(cap * cap - lateral * lateral, 0))
            return np.clip(x, -remaining, remaining), lateral, remaining

        requested_point = lambda: origin + unit * np.array([request_x.get_value(), request_y.get_value(), 0])
        applied_point = lambda: origin + unit * np.array([*clipped()[:2], 0])
        requested = DashedLine(origin, requested_point()).set_stroke(RED, 2.5)
        request_dot = Dot(requested_point(), radius=.07, fill_color=RED)
        applied = Arrow(origin, applied_point(), buff=0, fill_color=GREEN)
        applied_dot = Dot(applied_point(), radius=.065, fill_color=GREEN)
        x_component = Line(origin, origin).set_stroke(BLUE, 5)
        y_component = Line(origin, origin).set_stroke(YELLOW, 5)
        removed = Line(origin, origin).set_stroke(RED, 4)
        allowance = Line(origin, origin).set_stroke(BLUE, 4)
        guide = DashedLine(origin, origin + RIGHT).set_stroke(WALL, 1.5)
        legend = VGroup(words("requested", 27, RED), words("applied", 27, GREEN))
        legend.arrange(DOWN, aligned_edge=LEFT, buff=.3).move_to(RIGHT_COLUMN + .1 * UP)
        note_position = RIGHT_COLUMN + 1.6 * DOWN
        readout_position = RIGHT_COLUMN + 2.5 * DOWN

        def update():
            ax, ay, remaining = clipped()
            req, actual = requested_point(), applied_point()
            corner = origin + unit * np.array([ax, 0, 0])
            circle.become(Circle(radius=unit * grip.get_value()).move_to(origin).set_stroke(GREEN, 2.5).set_fill(GREEN, .035))
            requested.put_start_and_end_on(origin, req)
            request_dot.move_to(req)
            applied.put_start_and_end_on(origin, actual)
            applied_dot.move_to(actual)
            # A tiny offset keeps zero-length lines well defined at exact fits.
            x_component.put_start_and_end_on(origin, corner + 1e-6 * RIGHT)
            y_component.put_start_and_end_on(corner, actual + 1e-6 * UP)
            removed.put_start_and_end_on(actual, req + 1e-6 * RIGHT)
            allowance.put_start_and_end_on(origin + unit * np.array([-remaining, ay, 0]),
                                          origin + unit * np.array([remaining + 1e-6, ay, 0]))
            guide.put_start_and_end_on(origin + unit * ay * UP, actual + 1e-6 * RIGHT)

        with self.live(update):
            fits = words("Inside the circle, the request fits.", 26).move_to(note_position)
            self.play(FadeIn(x_component), FadeIn(y_component), FadeIn(applied), FadeIn(applied_dot), FadeIn(fits))
            self.caption("Blue is throttle, yellow is turning, and green is the acceleration the tires deliver.")
            self.play(request_x.animate.set_value(.4), request_y.animate.set_value(.65), run_time=3)
            self.wait(1.5)
            outside = words("More throttle than the tires can supply.", 25, RED).move_to(note_position)
            self.play(FadeIn(requested), FadeIn(request_dot), FadeIn(legend), FadeOut(fits))
            self.caption("Red is the request. Too much throttle for the grip, and the simulator clips it.")
            self.play(request_x.animate.set_value(1.2), FadeIn(removed), run_time=3)
            self.play(Write(outside))
            self.wait(2)
            remaining_eq = formula(R"|a_{\rm long}|\leq\sqrt{(\mu g)^2-a_{\rm lateral}^2}", 33, BLUE).move_to(readout_position)
            self.play(FadeIn(allowance), FadeIn(guide), Write(remaining_eq), run_time=2)
            self.caption("The blue slice is the throttle or braking left for this turn. A harder turn leaves less.")
            self.play(request_y.animate.set_value(.94), run_time=4)
            self.wait(1)
            self.play(request_y.animate.set_value(.2), run_time=3)
            self.wait(1)
            self.play(FadeOut(outside), FadeOut(remaining_eq),
                      request_x.animate.set_value(.55), request_y.animate.set_value(.6), run_time=3)
            friction = words("Lower friction means a smaller circle.", 26).move_to(note_position)
            mu_value = DecimalNumber(1., num_decimal_places=2, font_size=30).set_color(GREEN)
            readout = VGroup(formula(R"\mu/\mu_0=", 30), mu_value).arrange(RIGHT, buff=.15).move_to(readout_position)
            old_circle = circle.copy().set_fill(opacity=0).set_stroke(WALL, 1.5, opacity=.5)
            self.caption("Now lower the grip itself. The whole circle shrinks around the same request.")
            self.play(FadeIn(friction), FadeIn(readout))
            self.add(old_circle)
            with self.live(lambda: mu_value.set_value(grip.get_value())):
                self.play(grip.animate.set_value(.65), run_time=4)
                self.wait(2)
                self.play(grip.animate.set_value(.85), run_time=3)
            self.wait(1)
            cap_turn = words("At the turning limit, nothing is left for throttle.", 24).move_to(note_position)
            self.play(FadeOut(friction), FadeOut(readout), FadeOut(old_circle), FadeIn(cap_turn),
                      request_y.animate.set_value(1.15), run_time=4)
            self.caption("The kernel caps turning first, then fits throttle or braking into whatever grip remains.")
            self.wait(3)
        return relation

    def capped_turn(self):
        self.heading("What happens when the requested turn needs too much grip?")
        start = np.array([-5, -1.7, 0])
        desired_radius, actual_radius = 2., 3.2
        distance = ValueTracker(0.)
        point = lambda radius, s: start + radius * np.array([np.sin(s / radius), 1 - np.cos(s / radius), 0])
        # Equal speed and travel distance; only the capped heading rate differs.
        requested_path = get_curve([point(desired_radius, s) for s in np.linspace(0, 4.2, 90)], RED, 2)
        actual_path = get_curve([point(actual_radius, s) for s in np.linspace(0, 4.2, 90)], GREEN, 3)
        ghost = Car(start, 0, RED, .85).fade(.4)
        racer = Car(start, 0, GREEN, .85)
        self.play(ShowCreation(requested_path), run_time=2)
        requested_label = words("requested turn", 24, RED).next_to(requested_path, UP, buff=.15)
        self.play(Write(requested_label), FadeIn(ghost), FadeIn(racer))
        limit = formula(R"a_{\rm lateral}\leq\mu g", 40, YELLOW).move_to(RIGHT_COLUMN + 1.4 * UP)
        radius_limit = formula(R"R\geq{v^2\over\mu g}", 43, GREEN).next_to(limit, DOWN, buff=.7)
        self.play(Write(limit), run_time=1.5)
        self.play(TransformFromCopy(limit, radius_limit), run_time=2)
        self.caption("The simulator limits the turn rate, so the path comes out wider than requested.")

        def update():
            s = distance.get_value()
            ghost.pose(point(desired_radius, s), s / desired_radius)
            racer.pose(point(actual_radius, s), s / actual_radius)

        with self.live(update):
            self.play(distance.animate.set_value(4.2), ShowCreation(actual_path), run_time=6)
            actual_label = words("limited by grip", 24, GREEN).move_to([-1.2, -1.45, 0])
            self.play(Write(actual_label))
            self.caption("Slowing down lowers the turning acceleration a bend needs.")
            self.wait(3)
        return radius_limit

    def rk4(self):
        self.heading("A curved move needs more than one straight-line guess.")
        # An enlarged interval keeps the four trial states apart on screen; the
        # construction is otherwise identical to the six real substeps.
        state = np.array([0., 0., 0., 2.])
        h, delta, steer_rate = .45, .25, .15
        trials, slopes, result = rk4_step(state, delta, h, steer_rate)
        origin = np.array([-5.1, -1.6, 0])
        project = lambda s: origin + 5 * np.array([s[0], s[1], 0])
        colors = [BLUE, YELLOW, PURPLE, GREEN]
        racer = Car(origin, 0, scale=.85)
        start = Dot(origin, radius=.075, fill_color=WHITE)
        self.play(FadeIn(racer), FadeIn(start))
        self.caption("RK4 samples the motion four times, then combines the samples into one update.")
        self.wait(3)
        ruler = Line([-5.1, 2, 0], [-.3, 2, 0]).set_stroke(WALL, 2)
        ruler_labels = VGroup(
            words(text, 22, MUTED).move_to([x, 2.4, 0])
            for x, text in [(-5.1, "start"), (-2.7, "halfway"), (-.3, "end")]
        )
        notes = VGroup(
            words("Trial predictions • enlarged interval h = 0.45 s", 23, MUTED).set_max_width(5.8).move_to([-2.8, -2.65, 0]),
            words("f = rates of change", 24, MUTED).move_to(RIGHT_COLUMN + 2.45 * DOWN),
        )
        self.play(ShowCreation(ruler), FadeIn(ruler_labels), FadeIn(notes))
        equations = VGroup(
            formula(text, 27, color).set_max_width(6.3).move_to(RIGHT_COLUMN + 1.75 * DOWN)
            for text, color in zip([
                R"k_1=f(s_n,\delta)",
                R"k_2=f(s_n+\tfrac h2 k_1,\delta_{\rm mid})",
                R"k_3=f(s_n+\tfrac h2 k_2,\delta_{\rm mid})",
                R"k_4=f(s_n+h k_3,\delta_{\rm end})",
            ], colors)
        )
        steps = VGroup(
            words(text, 26, color).set_max_width(6.3)
            for text, color in zip([
                "1. Sample the motion now", "2. Predict halfway; sample again",
                "3. Refine halfway; sample again", "4. Predict the end; sample again",
            ], colors)
        ).arrange(DOWN, aligned_edge=LEFT, buff=.4).move_to(RIGHT_COLUMN + .45 * UP)
        explanations = [
            "Start from the current state. k₁ is its rate of change.",
            "Predict the halfway state with k₁, and sample the motion there.",
            "Improve the halfway prediction with k₂, and sample again.",
            "Predict the end of the interval with k₃. Take one last sample.",
        ]
        mark_x = [-5.1, -2.7, -2.7, -.3]
        arrows, dots, labels, cars, marks, predictions = (VGroup() for _ in range(6))
        for i, (trial, slope, color) in enumerate(zip(trials, slopes, colors)):
            point = project(trial)
            dot = Dot(point, radius=.065, fill_color=color)
            arrow = Arrow(point, point + .62 * np.array([slope[0], slope[1], 0]), buff=0, fill_color=color)
            label = formula(f"k_{i + 1}", 27, color).next_to(arrow, UP, buff=.12)
            trial_car = Car(point, trial[2], color, .65).set_opacity(.35)
            mark = Dot([mark_x[i], 2 + (.1 if i == 2 else 0), 0], radius=.07, fill_color=color)
            if i:
                prediction = DashedLine(origin, point).set_stroke(color, 1.5, opacity=.45)
                predictions.add(prediction)
                self.play(ShowCreation(prediction), TransformFromCopy(dots[-1], dot),
                          arrows[-1].animate.set_opacity(.3), cars[-1].animate.set_opacity(.12),
                          FadeOut(labels[-1]), FadeOut(equations[i - 1]), run_time=1.4)
            else:
                self.play(FadeIn(dot))
            self.play(FadeIn(trial_car), GrowArrow(arrow), Write(label), FadeIn(steps[i]), FadeIn(mark), run_time=1.6)
            self.caption(explanations[i])
            self.play(Write(equations[i]), run_time=1)
            for group, mob in zip((arrows, dots, labels, cars, marks), (arrow, dot, label, trial_car, mark)):
                group.add(mob)
            self.wait(4)
        self.heading("Now turn those four predictions into one actual move.",
                     FadeOut(VGroup(predictions, dots, labels[-1], equations[-1], steps, cars,
                                    ruler, ruler_labels, marks, notes)))
        average = formula(R"\Delta s={h\over6}(k_1+2k_2+2k_3+k_4)", 35).set_max_width(6.4).move_to(RIGHT_COLUMN + 1.6 * UP)
        self.play(Write(average), run_time=2)
        self.caption("Combine the four samples with weights 1, 2, 2, 1. The middle two count double.")
        weights = VGroup(formula(str(w), 38, c) for w, c in zip([1, 2, 2, 1], colors))
        weights.arrange(RIGHT, buff=.7).next_to(average, DOWN, buff=.6)
        self.play(FadeIn(weights))
        # Chain the weighted position increments head to tail.
        tip = origin.copy()
        for i, (slope, weight, color) in enumerate(zip(slopes, [1, 2, 2, 1], colors)):
            increment = 5 * h * weight / 6 * np.array([slope[0], slope[1], 0])
            arrow = Arrow(tip, tip + increment, buff=0, fill_color=color)
            self.play(TransformFromCopy(arrows[i].copy().set_opacity(1), arrow), Indicate(weights[i], color=color), run_time=1.3)
            tip = tip + increment
        self.play(FadeOut(arrows))
        endpoint = Dot(project(result), radius=.08, fill_color=WHITE)
        self.play(FadeIn(endpoint), Transform(racer, Car(project(result), result[2], scale=.85)), run_time=2)
        updated = formula(R"s_{n+1}=s_n+\Delta s", 35).next_to(weights, DOWN, buff=.9)
        self.play(Write(updated))
        self.caption("Only now does the car move. The same weighted sum also updates heading and speed.")
        self.wait(3)
        return updated

    def substeps(self):
        self.heading("Now advance the state by 1/60 second.")
        interval = Line(5.5 * LEFT, 5.5 * RIGHT).shift(1.65 * UP).set_stroke(WHITE, 2)
        ticks = VGroup(Line(.15 * DOWN, .15 * UP).move_to([x, 1.65, 0]) for x in np.linspace(-5.5, 5.5, 7))
        brace = Brace(interval, UP, buff=.15)
        dt = formula(R"\Delta t=1/60\ \text{second}", 28).next_to(brace, UP, buff=.1)
        numbers = VGroup(
            formula(str(i + 1), 25, MUTED).move_to([(ticks[i].get_x() + ticks[i + 1].get_x()) / 2, 1.2, 0])
            for i in range(SUBSTEPS)
        )
        self.play(ShowCreation(interval), ShowCreation(ticks), GrowFromCenter(brace), Write(dt))
        self.play(Write(numbers))
        # The simulator's six real substeps, with the tiny displacement magnified on screen.
        state = np.array([0., 0., .15, 4.])
        delta, steer_rate = .24, .4
        states = [state.copy()]
        for _ in range(SUBSTEPS):
            *_, state = rk4_step(state, delta, DT_SUB, steer_rate)
            delta += steer_rate * DT_SUB
            states.append(state.copy())
        points = [np.array([-5 + 145 * s[0], -.7 + 145 * s[1], 0]) for s in states]
        path = get_curve(points, BLUE, 2).set_opacity(.3)
        racer = Car(points[0], states[0][2], scale=.9)
        self.play(ShowCreation(path), FadeIn(racer))
        update = formula(R"s_{n+1}=s_n+{h\over6}(k_1+2k_2+2k_3+k_4)", 36).move_to(1.75 * DOWN)
        note = words("One RK4 update: four slope estimates, one new state", 25, BLUE).next_to(update, DOWN, buff=.3)
        self.play(Write(update), FadeIn(note))
        self.caption("Each yellow interval is one RK4 update. Its result becomes the next starting state.")
        self.play(FadeIn(words("Movement magnified ×145", 22, MUTED).move_to(.15 * DOWN)))
        for i in range(SUBSTEPS):
            segment = Line(ticks[i].get_center(), ticks[i + 1].get_center()).set_stroke(YELLOW, 5)
            self.play(ShowCreation(segment), FadeIn(Dot(points[i], radius=.05, fill_color=BLUE)),
                      Transform(racer, Car(points[i + 1], states[i + 1][2], scale=.9)),
                      numbers[i].animate.set_color(YELLOW), run_time=1.25)
        self.wait(2.5)


class RewardAndRespawn(FilmScene):
    """Score: the two track maps, the reward terms, and what a crash does."""

    phase = 1

    def construct(self):
        self.reward_terms()
        self.offset_penalty()
        self.crash_and_respawn()
        self.caption("Now measure what this car sees. The wall-distance map will help again.", WHITE)
        self.wait(3)

    def get_corridor(self, top=1.6):
        walls = VGroup(Line([-6, y, 0], [6, y, 0]) for y in (top, -top)).set_stroke(WALL, 4)
        centerline = DashedLine(6 * LEFT, 6 * RIGHT).set_stroke(BLUE, 2)
        waypoints = VGroup(Dot([x, 0, 0], radius=.05, fill_color=BLUE) for x in np.arange(-5, 6))
        return VGroup(walls, centerline, waypoints)

    def reward_terms(self):
        self.heading("The car has moved. Was that a useful move?")
        top = 1.6
        walls, centerline, waypoints = self.get_corridor(top)
        racer = Car([-3, .55, 0])
        self.play(ShowCreation(walls), ShowCreation(centerline), FadeIn(waypoints), FadeIn(racer))
        self.caption("Reward is a score: progress along the track is good, hugging a wall is bad.")
        self.wait(3)
        ring = Circle(radius=top - .55).move_to(racer).set_stroke(YELLOW, 2)
        distance = Line(racer.get_center(), [-3, top, 0]).set_stroke(YELLOW, 4)
        wall_question = words("How close is the wall?", 29, YELLOW).move_to([-3.4, 2.55, 0])
        wall_answer = words("A precomputed wall-distance map", 24, YELLOW).move_to([2.7, 2.55, 0])
        self.play(ShowCreation(ring), ShowCreation(distance), Write(wall_question))
        self.play(FadeIn(wall_answer))
        self.caption("Built once when the track loads: every pixel stores its distance to the nearest wall.")
        self.wait(3)
        nearest = Line(racer.get_center(), [-3, 0, 0]).set_stroke(GREEN, 4)
        track_question = words("Where am I along the track?", 29, BLUE).move_to(2.55 * UP)
        self.play(FadeOut(VGroup(ring, distance, wall_question, wall_answer)), Write(track_question),
                  ShowCreation(nearest), FlashAround(waypoints[2]))
        self.caption("A second map gives the nearest centerline waypoint: our place along the track.")
        selected = Dot([-3, 0, 0], radius=.09, fill_color=GREEN)
        travel = Arrow([-3, -.55, 0], [-2, -.55, 0], buff=0, fill_color=GREEN).set_opacity(0)
        progress = words("signed waypoint progress: +0", 25, GREEN).move_to([-1, -1.05, 0])
        self.play(FadeIn(selected), Write(progress))
        self.add(travel)
        self.wait(1)
        self.caption("The waypoint index moves in whole steps. Its change since the last step is the progress.")

        def advance(mob, alpha):
            x = -3 + 4 * alpha
            # Equal spacing is illustrative; the simulator reads an integer lookup table.
            index = int(np.clip(np.floor(x + 5.5), 0, len(waypoints) - 1))
            point = waypoints[index].get_center()
            mob.move_to([x, .55, 0])
            nearest.put_start_and_end_on(mob.get_center(), point)
            selected.move_to(point)
            if index > 2:
                travel.become(Arrow([-3, -.55, 0], [point[0], -.55, 0], buff=0, fill_color=GREEN))
            else:
                travel.set_opacity(0)
            progress.become(words(f"signed waypoint progress: {index - 2:+d}", 25, GREEN).move_to([-1, -1.05, 0]))

        self.play(UpdateFromAlphaFunc(racer, advance), run_time=5, rate_func=linear)
        self.play(FlashAround(waypoints[6]))
        self.wait(.5)
        reward = VGroup(
            formula(text, 32, color) for text, color in [
                ("r=", WHITE), (R"\text{progress}", GREEN),
                (R"-\text{wall penalty}", RED), (R"-\text{offset}^2", BLUE),
            ]
        ).arrange(RIGHT, buff=.18).move_to(2.55 * UP)
        self.play(FadeOut(VGroup(track_question, travel, selected)), FadeIn(reward[0]),
                  ReplacementTransform(progress, reward[1]), run_time=1.4)
        self.caption("Moving forward adds reward. Moving backward subtracts it.")
        self.wait(2.5)
        self.play(FadeIn(reward[2]))
        self.caption("Within a short band of a wall, subtract a penalty. Farther away it vanishes.")
        self.wait(2.5)
        self.play(TransformFromCopy(nearest, reward[3]), run_time=1.4)
        self.caption("Also subtract the square of the sideways offset from the centerline. Why the square?")
        self.play(Indicate(nearest, color=RED))
        self.wait(3)

    def offset_penalty(self):
        self.heading("Why square the distance from the centerline?", wipe=True)
        x = -4.
        walls = VGroup(Line([-6, y, 0], [-1, y, 0]) for y in (2, -2)).set_stroke(WALL, 3)
        centerline = DashedLine([-6, 0, 0], [-1, 0, 0]).set_stroke(BLUE, 2)
        offset = ValueTracker(.6)
        racer = Car([x, .6, 0])
        area = Square(side_length=.6).move_to([x + .3, .3, 0]).set_stroke(BLUE, 2).set_fill(BLUE, .25)
        distance = Line([x, 0, 0], [x, .6, 0]).set_stroke(YELLOW, 4)
        d_label = formula("d", 31, YELLOW).move_to([x - .4, .3, 0])
        self.play(ShowCreation(walls), ShowCreation(centerline), FadeIn(racer))
        self.play(ShowCreation(distance), Write(d_label))
        self.play(GrowFromPoint(area, [x, 0, 0]), run_time=1.5)
        graph_origin = np.array([3.25, -1.25, 0])
        gp = lambda d: graph_origin + np.array([1.55 * d, 1.15 * d * d, 0])
        axes = VGroup(
            Line(graph_origin + 2.5 * LEFT, graph_origin + 2.5 * RIGHT),
            Line(graph_origin, graph_origin + 2.65 * UP),
        ).set_stroke(WALL, 1.5)
        ticks = VGroup(words(str(d), 20, MUTED).move_to(graph_origin + [1.55 * d, -.3, 0]) for d in (-1, 0, 1))
        axis_label = words("offset", 22, MUTED).next_to(axes[0], RIGHT, buff=.15)
        graph = get_curve([gp(d) for d in np.linspace(-1.45, 1.45, 120)], BLUE, 3)
        marker = Dot(gp(.6), radius=.07, fill_color=YELLOW)
        guide = DashedLine(graph_origin + 1.55 * .6 * RIGHT, gp(.6)).set_stroke(YELLOW, 1.5)
        penalty = formula(R"\text{penalty}=d^2", 38, BLUE).move_to([3.25, 2.25, 0])
        self.play(ShowCreation(axes), Write(ticks), Write(axis_label))
        self.play(TransformFromCopy(area, penalty), ShowCreation(graph), run_time=2)
        self.play(FadeIn(marker), ShowCreation(guide))
        offset_value = DecimalNumber(.6, num_decimal_places=2, font_size=29).set_color(YELLOW)
        penalty_value = DecimalNumber(.36, num_decimal_places=2, font_size=29).set_color(BLUE)
        readout = VGroup(words("offset", 24, YELLOW), offset_value, words("penalty", 24, BLUE), penalty_value)
        readout.arrange_in_grid(2, 2, h_buff=1.2, v_buff=.25).move_to([3.25, -2.5, 0])
        self.play(FadeIn(readout))
        self.caption("The square's area is the centerline penalty.")

        def update():
            d = offset.get_value()
            size = max(abs(d), 1e-5)
            racer.move_to([x, d, 0])
            area.become(Square(side_length=size).move_to([x + size / 2, d / 2, 0]).set_stroke(BLUE, 2).set_fill(BLUE, .25))
            distance.put_start_and_end_on(np.array([x, 0, 0]), np.array([x, d + 1e-6, 0]))
            d_label.move_to([x - .4, d / 2, 0])
            marker.move_to(gp(d))
            guide.put_start_and_end_on(graph_origin + 1.55 * d * RIGHT, gp(d) + 1e-6 * UP)
            offset_value.set_value(d)
            penalty_value.set_value(d * d)

        with self.live(update):
            self.wait(1.5)
            self.play(offset.animate.set_value(1.2), run_time=4)
            quarters = VGroup(Line([x + .6, 0, 0], [x + .6, 1.2, 0]), Line([x, .6, 0], [x + 1.2, .6, 0])).set_stroke(WHITE, 2)
            self.play(ShowCreation(quarters))
            self.caption("Twice the offset gives four times the penalty.")
            self.wait(2.5)
            self.play(FadeOut(quarters), offset.animate.set_value(0), run_time=3)
            self.play(offset.animate.set_value(-1.2), run_time=4)
            self.caption("The same distance on either side costs the same. Squaring makes it symmetric.")
            self.wait(2.5)

    def crash_and_respawn(self):
        self.heading("Too little clearance ends the episode.", wipe=True)
        corridor = self.get_corridor()
        racer = Car([1, .55, 0])
        self.play(FadeIn(corridor), FadeIn(racer))
        footprint = Circle(radius=float(np.hypot(.65, .38) / 2)).move_to(racer).set_stroke(RED, 2)
        clearance = formula(R"\text{clearance}=d_{\rm wall}-{1\over2}\text{car diagonal}", 34).move_to(2.55 * UP)
        self.play(Write(clearance), ShowCreation(footprint))
        self.caption("A circle encloses the car. The wall has to stay outside it.")
        self.wait(3)
        self.play(racer.animate.shift(.85 * UP), footprint.animate.shift(.85 * UP), run_time=1.6)
        hit = words("clearance < 0", 30, RED).move_to([3, .5, 0])
        self.play(Write(hit), FlashAround(footprint, color=RED))
        self.caption("The wall enters the circle: clearance is negative, and this episode is over.")
        self.wait(3)
        reset = words("done = 1    •    reward = −25    •    respawn", 30, RED).move_to(2.55 * UP)
        self.play(FadeOut(VGroup(hit, footprint)), FadeTransform(clearance, reset), Transform(racer, Car([-4, 0, 0])))
        detail = words("A fresh start: a new waypoint, zero speed, straight wheels.", 25).move_to(2.5 * DOWN)
        self.play(FadeIn(detail))
        self.caption("done marks the ended episode. The car respawns immediately at a random waypoint.")
        self.wait(3)


class WarpLidar(FilmScene):
    """Sense: one ray's safe jumps, then the fan of rays as an observation row."""

    phase = 2

    def construct(self):
        result = self.one_ray()
        self.bridge(result, "One ray gives one distance.", "Now repeat that in 108 directions around the car.")
        self.fan()
        self.caption("We have moved, scored, and sensed one car. Let's put the batch back together.", WHITE)
        self.wait(3)

    def one_ray(self):
        self.heading("How far does a ray travel before it hits a wall?")
        # A smooth corridor stands in for the pixel map: every circle touches its nearest wall.
        walls = VGroup(
            Line([-6, 2, 0], [5.5, 2, 0]), Line([-6, -2, 0], [5.5, -2, 0]), Line([5.5, -2, 0], [5.5, 2, 0]),
        ).set_stroke(WALL, 4)
        start = np.array([-5., -1., 0.])
        direction = np.array([.96, .28, 0.])
        wall_distance = lambda p: min(2 - p[1], p[1] + 2, 5.5 - p[0])
        end = start + min((2 - start[1]) / direction[1], (5.5 - start[0]) / direction[0]) * direction
        ray_angle = np.arctan2(direction[1], direction[0])
        racer = Car(start - .3 * direction, ray_angle, scale=.75)
        beam = DashedLine(start, end).set_stroke(BLUE, 2, opacity=.45)
        self.play(ShowCreation(walls), FadeIn(racer), ShowCreation(beam))
        self.caption("Lidar measures distance along a ray. The wall-distance map from the last chapter helps again.")
        self.wait(3)
        cursor = Dot(start, radius=.065, fill_color=YELLOW)
        rule = formula(R"p_{n+1}=p_n+d(p_n)\,\hat u", 32).move_to(2.55 * UP)
        glossary = words("p = sample point    d(p) = wall distance    û = ray direction", 22, MUTED).move_to(2.65 * DOWN)
        p = start.copy()
        rings = VGroup()
        for i in range(24):
            radius = wall_distance(p)
            if radius < .015:
                break
            ring = Circle(radius=radius).move_to(p).set_stroke(YELLOW, 2).set_fill(YELLOW, .035)
            jump = Line(p, p + radius * direction).set_stroke(YELLOW, 3)
            pace = .7 if i < 3 else .3
            self.play(ShowCreation(ring), ShowCreation(jump), *([FadeIn(cursor)] if i == 0 else []), run_time=pace)
            if i == 0:
                nearest = Line(p, [p[0], -2, 0]).set_stroke(YELLOW, 3)
                label = formula("d(p)", 30, YELLOW).next_to(jump, DOWN, buff=.2)
                self.play(ShowCreation(nearest), Write(label))
                self.caption("No wall lies inside this circle, even though the nearest one is beside the ray. Jump one radius.")
                self.wait(3)
                self.play(FadeOut(label), FadeOut(nearest), Write(rule), FadeIn(glossary))
            elif i == 1:
                self.caption("At the new point, look up the distance again. Draw the new circle, and jump again.")
                self.wait(2.5)
            elif i == 2:
                self.caption("Repeat. The jumps shrink as the ray closes in on the wall.")
                self.wait(2)
            p = p + radius * direction
            self.play(cursor.animate.move_to(p), ring.animate.set_stroke(opacity=.15).set_fill(opacity=0),
                      jump.animate.set_color(BLUE), run_time=pace)
            rings.add(ring)
        self.play(cursor.animate.move_to(end), FlashAround(cursor, color=YELLOW), run_time=.5)
        self.wait(1)
        self.play(FadeOut(rings), FadeOut(rule), FadeOut(glossary))
        range_line = Line(start, end).set_stroke(BLUE, 4)
        normal = np.array([direction[1], -direction[0], 0])
        brace = Brace(range_line, normal, buff=.22)
        result = formula(R"r_j=\sum_n d(p_n)", 34, BLUE).rotate(ray_angle).move_to(brace.get_tip() + .4 * normal)
        self.play(ShowCreation(range_line), GrowFromCenter(brace), Write(result))
        self.caption("Add up the jumps: that is the ray's range, capped at 20 meters.")
        self.wait(3)
        self.play(result.animate.rotate(-ray_angle), run_time=.8)
        return result

    def fan(self):
        self.heading("Now ask in 108 directions.")
        origin = 3.3 * LEFT
        room = Rectangle(width=5.2, height=4).move_to(origin).set_stroke(WALL, 3)
        racer = Car(origin)
        plot_base = -1.45
        bar_x = np.linspace(.85, 5.55, 19)

        def get_rays(yaw=0.):
            """Nineteen of the 108 beams, clipped to the room. The sensor sits ahead of the car's center."""
            sensor = origin + .27 * rotate_vector(RIGHT, yaw)

            def reach(d):
                return min(
                    ((high if d[axis] > 0 else low) - sensor[axis]) / d[axis]
                    for axis, low, high in [(0, -5.9, -.7), (1, -2, 2)] if abs(d[axis]) > 1e-8
                )

            return VGroup(
                Line(sensor, sensor + reach(d) * d).set_stroke(BLUE, 1.6)
                for d in (rotate_vector(RIGHT, yaw + a) for a in np.linspace(-3 * PI / 4, 3 * PI / 4, 19))
            )

        def get_profile(rays):
            return VGroup(
                Line([x, plot_base, 0], [x, plot_base + .85 * ray.get_length(), 0]).set_stroke(BLUE, 5)
                for x, ray in zip(bar_x, rays)
            )

        beams = get_rays()
        self.play(ShowCreation(room), FadeIn(racer))
        self.play(LaggedStartMap(ShowCreation, beams, lag_ratio=.04), run_time=2)
        sample = words("19 rays drawn / 108 computed", 22, MUTED).next_to(room, DOWN, buff=.2)
        self.play(FadeIn(sample))
        index = VGroup(
            formula(R"(i,j)\longmapsto\text{one ray}", 32, BLUE),
            words("i chooses the car\nj chooses the beam", 28, MUTED),
        ).arrange(DOWN, buff=.4).move_to(RIGHT_COLUMN + .9 * UP)
        self.play(Write(index[0]), FadeIn(index[1]))
        self.caption("A lidar work item is one car and one beam. Each ray marches on its own.")
        self.play(beams[10].animate.set_stroke(YELLOW, 4))
        self.wait(2.5)

        self.heading("The fan of rays becomes a row of numbers.", FadeOut(index), beams[10].animate.set_stroke(BLUE, 1.6))
        bars = get_profile(beams)
        baseline = Line([.65, plot_base, 0], [5.75, plot_base, 0]).set_stroke(WALL, 2)
        angle_labels = VGroup(
            formula(text, 23, MUTED).move_to([x, plot_base - .35, 0])
            for x, text in [(bar_x[0], R"-135^\circ"), (bar_x[9], R"0^\circ"), (bar_x[-1], R"135^\circ")]
        )
        title = words("distance along each ray", 26, BLUE).move_to(RIGHT_COLUMN + 2.3 * UP)
        self.play(ShowCreation(baseline), Write(angle_labels), Write(title))
        self.play(LaggedStart(*(TransformFromCopy(ray, bar) for ray, bar in zip(beams, bars)), lag_ratio=.08), run_time=4)
        self.caption("Long bar, distant wall. Short bar, nearby wall. Each bar is one ray, in beam order.")
        self.wait(2)
        yaw = ValueTracker(0.)

        def update():
            rays = get_rays(yaw.get_value())
            beams.become(rays)
            bars.become(get_profile(rays))
            racer.pose(origin, yaw.get_value())

        with self.live(update):
            self.play(yaw.animate.set_value(.65), run_time=4)
            self.caption("Turn the car and every beam turns with it. The whole observation row changes.")
            self.play(yaw.animate.set_value(-.35), run_time=5)
            self.wait(1.5)


class TheHandoff(FilmScene):
    """Return: the order of the launches, the shared output buffers, and the loop closing."""

    phase = 3

    def construct(self):
        result = self.work_grid()
        self.bridge(result, "Sequential inside a car or a ray; independent across cars and rays.",
                    "Each work item writes its result into the matching output row.")
        result = self.shared_outputs()
        self.bridge(result, "Warp writes the results; Torch can view the same buffers.",
                    "Now return to the car we started with, and follow one complete step.")
        self.close_the_loop()

    def work_grid(self):
        self.heading("What runs in order, and what can run independently?")
        ys = [1.55, .6, -.35, -1.3]
        cars = VGroup(Car([-5.7, y, 0], scale=.7) for y in ys)
        rows = VGroup(VGroup(Dot([-4.6 + .5 * j, y, 0], radius=.06, fill_color=WALL) for j in range(6)) for y in ys)
        links = VGroup(Line(row[0].get_center(), row[-1].get_center()).set_stroke(WALL, 1.5) for row in rows)
        pulses = VGroup(Dot(car.get_center(), radius=.09, fill_color=YELLOW) for car in cars)
        row_labels = VGroup(formula(f"i={i}", 22, MUTED).next_to(car, LEFT, buff=.15) for i, car in enumerate(cars))
        physics_label = words("six updates per car", 28, YELLOW).move_to([-3.7, 2.55, 0])
        ray_label = words("108 rays per car", 28, BLUE).move_to([3.1, 2.55, 0])
        self.play(FadeIn(cars), ShowCreation(links), FadeIn(rows), FadeIn(row_labels), FadeIn(pulses), Write(physics_label))
        self.caption("Down the page, cars advance independently. Across a row, the six updates run in order.")
        for j in range(SUBSTEPS):
            self.play(*(pulse.animate.move_to(row[j]) for pulse, row in zip(pulses, rows)),
                      *(row[j].animate.set_color(YELLOW) for row in rows), run_time=.75)
        self.wait(1.5)
        barrier = DashedLine([-.75, -1.8, 0], [-.75, 2, 0]).set_stroke(WHITE, 1.5)
        then = words("then", 22, MUTED).next_to(barrier, DOWN, buff=.1)
        self.play(ShowCreation(barrier), FadeIn(then))
        self.caption("Lidar starts after physics has written the new poses.")
        self.play(Write(ray_label))
        grid = VGroup(VGroup(Dot([.9 + .43 * j, y, 0], radius=.055, fill_color=BLUE) for j in range(10)) for y in ys)
        # Each finished car state fans out into that car's independent ray items.
        for j in range(10):
            self.play(*(TransformFromCopy(pulse, row[j]) for pulse, row in zip(pulses, grid)), run_time=.22)
        sample = words("Each blue dot = one beam work item", 22, MUTED).move_to([2.9, -2, 0])
        self.play(FadeIn(sample))
        self.wait(1.5)
        selected = grid[2][6]
        guides = VGroup(
            DashedLine([.3, ys[2], 0], selected.get_center()),
            DashedLine([selected.get_x(), 2, 0], selected.get_center()),
        ).set_stroke(YELLOW, 2)
        i_label = formula("i=2", 25, YELLOW).move_to([.2, ys[2] - .3, 0])
        j_label = formula("j=6", 25, YELLOW).move_to([selected.get_x(), 2.15, 0])
        self.play(ShowCreation(guides), Write(i_label), Write(j_label), selected.animate.set_color(YELLOW))
        self.play(FlashAround(selected, color=YELLOW))
        destination = formula(R"\texttt{obs}[2,\,8]=r_6", 35, YELLOW).move_to([2.9, -2.75, 0])
        self.play(TransformFromCopy(selected, destination), run_time=1.5)
        self.caption("Car 2, beam 6 writes entry 8: the first two entries hold steering and speed.")
        self.wait(3)
        return destination

    def shared_outputs(self):
        self.heading("Where do the results go?")
        xs = [-2.9, -2.32, -1.74, -1.16, -.58, 0.]
        ys = [1.05, .4, -.25, -.9]
        cell = lambda width, x, y: Rectangle(width=width, height=.53).move_to([x, y, 0]).set_stroke(WALL, 1.4)
        obs = VGroup(VGroup(cell(.53, x, y) for x in xs) for y in ys)
        reward = VGroup(cell(.8, 1.25, y) for y in ys)
        done = VGroup(cell(.65, 2.6, y) for y in ys)
        frame = Rectangle(width=7, height=3.25).move_to([-.15, .3, 0]).set_stroke(GREEN, 2)
        titles = VGroup(
            words("observation", 26, GREEN).move_to([-1.45, 2.3, 0]),
            words("reward", 26, GREEN).move_to([1.25, 2.3, 0]),
            words("done", 26, GREEN).move_to([2.6, 2.3, 0]),
        )
        meanings = VGroup(
            words(text, 19, MUTED).move_to([x, 2.65, 0])
            for text, x in [("wheel, speed, ranges", -1.45), ("score", 1.25), ("episode ended?", 2.6)]
        )
        columns = VGroup(
            formula(text, 22, MUTED).move_to([x, 1.65, 0])
            for x, text in zip(xs, [R"\delta", "v", "r_0", "r_1", R"\cdots", "r_{107}"])
        )
        self.play(ShowCreation(frame), FadeIn(obs), FadeIn(reward), FadeIn(done),
                  Write(titles), Write(columns), FadeIn(meanings))
        self.caption("Four environments shown; the observation columns are abbreviated.")
        warp = VGroup(words("Warp", 32, YELLOW), words("writes", 24, MUTED)).arrange(DOWN, buff=.15).move_to([-5.2, .6, 0])
        warp_arrow = Arrow([-4.5, .45, 0], [-3.68, .45, 0], buff=0, fill_color=YELLOW)
        self.play(Write(warp[0]), FadeIn(warp[1]), GrowArrow(warp_arrow))
        # Illustrative values. The third row has just crashed and respawned.
        values = [
            ["0.1", "2.0", "3.1", "2.4", R"\cdots", "4.2"],
            ["-0.2", "1.5", "2.8", "3.0", R"\cdots", "1.6"],
            ["0", "0", "2.5", "1.9", R"\cdots", "3.2"],
            ["0.2", "3.0", "1.7", "2.1", R"\cdots", "2.9"],
        ]
        entries = VGroup(
            VGroup(formula(value, 20, GREEN).move_to(cell) for value, cell in zip(row, cells))
            for row, cells in zip(values, obs)
        )
        reward_values = VGroup(
            formula(value, 23, RED if value == "-25" else GREEN).move_to(cell)
            for value, cell in zip(["0.3", "0.1", "-25", "0.4"], reward)
        )
        done_values = VGroup(
            formula(str(value), 23, RED if value else GREEN).move_to(cell)
            for value, cell in zip([0, 0, 1, 0], done)
        )
        stages = VGroup(words("move + score", 27, YELLOW), words("sense", 27, BLUE), words("randomness tick", 27, MUTED))
        stages.arrange(RIGHT, buff=1.2).move_to(2.15 * DOWN)
        self.play(FadeIn(stages))
        physics_cells = VGroup(*(cell for row in obs for cell in row[:2]), *reward, *done)
        physics_values = VGroup(*(value for row in entries for value in row[:2]), *reward_values, *done_values)
        self.play(physics_cells.animate.set_stroke(YELLOW, 2.5), FlashAround(stages[0], color=YELLOW))
        self.play(LaggedStartMap(FadeIn, physics_values, lag_ratio=.06), run_time=2)
        self.caption("Physics writes steering, speed, reward, and done. Each row belongs to one car.")
        self.wait(2)
        lidar_cells = VGroup(*(cell for row in obs for cell in row[2:]))
        lidar_values = VGroup(*(value for row in entries for value in row[2:]))
        self.play(lidar_cells.animate.set_stroke(BLUE, 2.5), FlashAround(stages[1], color=BLUE))
        self.play(LaggedStartMap(FadeIn, lidar_values, lag_ratio=.05), run_time=2)
        self.caption("Then lidar fills the range columns from each car's updated, or respawned, position.")
        self.wait(2)
        tick = formula(R"\texttt{tick}[0]=42", 30, YELLOW).move_to([4.95, -.65, 0])
        next_tick = formula(R"\texttt{tick}[0]=43", 30, YELLOW).move_to(tick)
        tick_note = words("one counter on the\nsimulation device", 21, MUTED).next_to(tick, UP, buff=.2)
        tick_arrow = Arrow(stages[2].get_top(), tick.get_bottom(), buff=.15, fill_color=MUTED)
        bump = words("bump_kernel: tick[0] += 1", 24, YELLOW).move_to(2.8 * DOWN)
        self.play(FlashAround(stages[2], color=WHITE), Write(tick), FadeIn(tick_note), GrowArrow(tick_arrow), Write(bump))
        self.caption("Last, one device counter ticks. Respawns mix it with the seed and car index.")
        self.play(TransformMatchingTex(tick, next_tick))
        self.wait(2.5)
        self.play(FadeOut(VGroup(next_tick, tick_note, tick_arrow, bump)))
        torch = VGroup(words("Torch", 32, GREEN), words("views", 24, MUTED)).arrange(DOWN, buff=.15).move_to([5.05, .6, 0])
        torch_arrow = Arrow([4.35, .45, 0], [3.4, .45, 0], buff=0, fill_color=GREEN)
        self.caption("Torch, the learning side, views these same buffers when it shares the CUDA device.")
        self.play(Write(torch[0]), FadeIn(torch[1]), GrowArrow(torch_arrow), run_time=1.5)
        self.play(Indicate(frame, color=GREEN), run_time=1.5)
        self.wait(1.5)
        shared = words("Two views of the same output storage.", 30, GREEN).move_to(2.15 * DOWN)
        self.play(FadeTransform(stages, shared))
        self.caption("No copies on a shared CUDA device. Elsewhere the outputs are copied once per step.")
        self.wait(3)
        return shared

    def close_the_loop(self):
        self.heading("One small world, repeated thousands of times.")
        road = get_track().scale(.48).move_to([-3.75, .2, 0])
        theta = ValueTracker(-PI / 2)
        racer = Car(*track_pose(theta.get_value(), road), scale=.6)
        self.play(ShowCreation(road), FadeIn(racer))
        summary = VGroup(
            words(text, 28).set_max_width(5.8) for text in [
                "Move: respect the available grip", "Score: reward useful progress",
                "Sense: measure the new view", "Return: hand the results to Torch",
            ]
        ).arrange(DOWN, aligned_edge=LEFT, buff=.55).move_to(RIGHT_COLUMN + .5 * UP)
        with self.live(lambda: racer.pose(*track_pose(theta.get_value(), road))):
            for line, phase in zip(summary, self.phases):
                self.play(FadeIn(line, shift=.2 * RIGHT), phase.animate.set_color(YELLOW),
                          theta.animate.increment_value(.35), run_time=1.5)
                self.wait(.6)
            self.caption("The program is the same. Each car supplies its own state, action, and track.")
            self.play(theta.animate.increment_value(.9), run_time=4)
        self.caption("That independence is what gives Warp so much work to do in parallel.", WHITE)
        self.wait(4)
