# Companion script

The film is silent and teaches through its pictures and captions. This script
tells the same story in prose, with a little more detail where the film keeps
things short. It is not a synchronized subtitle file; the captions and their
reading time live in `main.py`.

## Opening

What happens between two frames of a racing simulation? Start with one car.
We know its position and heading, its speed, and its wheel angle. Together
those numbers are its state: a description of the car right now.

An action asks the steering wheel to turn and the car to accelerate or brake.
The simulator takes the state and the action, moves the car, scores that move,
and measures what the car can see from its new position. The learning code
receives those results and chooses the next action. One complete step is Move,
Score, Sense, and Return.

Now imagine thousands of copies of this little world. They are separate
environments even when they use the same track, so each can take its next step
on its own. Eight thousand is a representative batch, not a count of GPU cores.

A kernel is one program applied to many work items. Here one physics work item
owns one car. Warp hands those items to the hardware. Pick environment five and
bring it closer; we follow its update before putting the batch back together.

## VehicleStep

### Velocity and position

The velocity arrow points where the car is going, and its length is the speed.
Split it into horizontal and vertical parts: those are the rates at which x and
y change. The letter v is speed, psi is heading, and a dot means change per
second. Turn the arrow and both parts change together; lengthen it and both
grow. Over a short time the car moves about velocity times time. That works
while the direction barely changes, but a turning car changes direction.

### Steering

Draw only the front and rear wheels. Each wheel rolls straight ahead, so its
turn center lies on a line perpendicular to it. The two perpendiculars meet at
the turn center. The wheelbase and the radius form a right triangle, so the
radius is the wheelbase divided by the tangent of the wheel angle. Turn the
wheel further and the center moves closer: the circle tightens.

Drive that circle at speed v. Each lap covers two pi R and turns the heading by
two pi, so the heading turns at v over R radians per second. Substitute the
radius and the heading rate becomes v tan delta over L. That is exactly the
request the kernel computes from wheel angle and speed. With straight wheels
the tangent is zero and the heading holds. A zero steering-rate action keeps
the current wheel angle; it does not straighten turned wheels.

### Turning needs acceleration

Drive around a circle at a steady speed. The velocity arrow keeps changing
direction, and a changing velocity is an acceleration even when the speed is
constant. For a small turn the change in velocity is about the speed times the
change in heading, measured in radians; the exact chord is twice the speed
times the sine of half the angle. Divide by the elapsed time and shrink the
interval: the inward acceleration is speed times the heading rate, which with
the heading rate v over R is v squared over R.

Double the speed on the same circle and the required turning acceleration
becomes four times larger. A bend that is easy at low speed can ask more of the
tires than they have when approached faster.

### The grip circle

Put those demands in a new picture. Up and down is turning; left and right is
braking and throttle. Friction limits how hard the tires can push: mu times the
car's weight, which divided by mass is an acceleration of mu g. Every
acceleration inside a circle of that radius is possible. The circle is the grip
budget.

Inside the circle a request is applied as asked. Ask for more throttle than the
grip allows and the simulator clips it: it keeps the turning component and
fits throttle or braking into the chord that remains, whose half-width is the
square root of mu g squared minus the turning acceleration squared. A harder
turn leaves less of that slice; braking draws from the same slice.

Lower the friction and the whole circle shrinks around the same request. A
request that used to fit may no longer fit without any change to the controls.
Exceeding the circle does not change mu; the model caps accelerations rather
than simulating a slipping tire. Finally, ask for more turning than the top of
the circle allows. The kernel caps turning first, and at that limit nothing is
left for throttle.

### The effect on the path

The red curve is the requested turn. The green car travels at the same speed
but cannot change heading fast enough, so its path is wider. The minimum radius
grows with speed squared and shrinks with grip. Slowing down lowers the turning
acceleration a bend needs.

### Four trial states

Before committing to the next state, sample how the motion changes across the
interval. The interval is enlarged here so the predictions stay distinguishable.
Start from the current state; that gives k one. Use it to predict a state
halfway through the interval and sample the motion there for k two. Refine the
halfway prediction with k two and sample again for k three. Predict the end of
the interval with k three and take one last sample, k four. Each evaluation
uses the steering angle at its own time and applies the grip limits.

These dots are trial states, not four successive moves. Combine the four
samples with weights one, two, two, one. The position components chain into a
path from the start to the new position, and the same weighted sum updates
heading and speed. Only now does the car move.

### Six small updates

One simulation step lasts a sixtieth of a second. The kernel divides it into
six intervals and applies one RK4 update to each, with every evaluation under
the grip limits and the steering angle advancing through the interval. Each
result becomes the next starting state, so the six updates run in order within
one car's work item. The movement on screen is magnified so each update is
visible.

## RewardAndRespawn

After the move, two precomputed maps answer two questions. How close is the
nearest wall? A distance transform, built once when the track loads, stores
that answer at every pixel. Where are we along the track? A second lookup table
gives the nearest centerline waypoint. Its index changes in whole steps; the
signed change since the previous step, wrapped around the loop, is the
progress. Forward counts positive and backward negative.

The reward adds progress, subtracts a penalty inside a short band near a wall,
and subtracts the squared sideways offset from the centerline. Progress and the
wall penalty also carry speed factors; the film shows the structure.

Why square the offset? Draw the offset as one side of a square. Its area is the
penalty, and the dot on the parabola shows the same relationship. Move twice as
far from the centerline and four copies of the original square fit inside the
new one. The same distance on either side costs the same.

The car has size. Subtract half its diagonal from the wall distance to get the
clearance. If clearance goes negative the episode is done, the reward is minus
twenty-five, and the car respawns immediately at a random waypoint with zero
speed, straight wheels, and fresh friction and wheelbase scales within fifteen
percent of nominal. A ten-thousand-step timeout also respawns but carries no
crash penalty.

## WarpLidar

Take one ray. How far does it travel before it hits a wall? At its starting
point, look up the wall distance. The nearest wall may be beside the ray, but
no wall lies inside a circle of that radius, so the ray can safely jump one
radius. At the new point look the distance up again, draw the new circle, and
jump again. The jumps shrink as the ray closes in on the wall. Add them up and
the sum is the ray's range, capped at twenty meters.

The drawing uses smooth walls. The implementation reads pixel distances from the
raster map, stops on a zero sample or the range cap, and converts back to
meters. On CUDA the map is read through a texture; on CPU from an array.

Now spread rays over a 270-degree fan. The film draws nineteen; the simulator
computes 108. Each ray marches on its own, so a lidar work item has two indices:
car i and beam j. Straighten the rays into bars in beam order and you have the
scan as a list of distances. Turn the car and every beam turns with it; the
whole observation row changes. That changing list is the lidar part of what the
policy sees.

## TheHandoff

Follow one row: each yellow point is a physics update using the state from the
one before, so the six updates run in order. The rows are different cars and
advance independently. After physics has written the new poses, lidar can
start, and each car supplies 108 independent beam items. Car two, beam six
writes observation entry eight, because the first two entries hold steering
and speed.

Now the output arrays: observations, rewards, and done flags, one row per car.
Physics writes steering, speed, reward, and done; the row that just crashed has
already respawned, so its steering and speed are zero. Then lidar fills the
range entries from those updated poses. Last, one device counter ticks once per
step. On a later respawn the physics kernel mixes that tick with the seed and
car index to draw a new waypoint, friction scale, and wheelbase scale.

Torch is the learning code's tensor library. When Warp and Torch share a CUDA
device, Torch's output tensors are views of these same buffers, and the kernels
launch on Torch's stream so physics finishes before lidar reads its results.
With Torch on another device the outputs are copied once per step.

Return to the car from the opening. Move it within the available grip, score
its progress, measure its new view, and hand the results to Torch. The program
is the same for every car; each supplies its own state, action, and track. That
independence is what gives Warp thousands of useful tasks to run in parallel.
