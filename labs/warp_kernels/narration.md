# Voice-over script

The renders are silent. Each paragraph follows a visible beat; hold on the
picture after its claim. Keep the tone curious and leave room for the motion.

## Opening

Watch just one car for a moment. Its next position depends on its own state,
its own controls, and its track. It doesn't need to know where another car is.

So imagine making thousands of copies of this little world. These are separate
environments, even if they use the same track. Each can take its next step
independently. Eight thousand is a representative batch, not a physical GPU
core count.

That is our first division of work. In `step_kernel`, index i means: compute
one car's next state. What happens inside that item?

## VehicleStep

### Velocity and position

Start with the velocity arrow. The car moves in the direction it points, and
its length tells us the speed. Split it into horizontal and vertical parts.
Those are the rates at which x and y change.

Turn the arrow, and the two components change together. Make it longer, and
both grow. Over a short enough interval, multiplying velocity by time gives
approximately how far the car moves. But if the car is turning during that
interval, its velocity is changing too.

### Steering and turning acceleration

Draw just the front and rear wheels. A wheel points along its local motion,
so the turn center must lie on a line perpendicular to that wheel. Draw the
two perpendiculars and find where they meet.

The wheelbase and turning radius make a right triangle. Its geometry gives
radius equals wheelbase divided by the tangent of the steering angle. Turn
the front wheel further, and the intersection moves closer. The turn tightens.

The blue velocity arrow stays tangent to the path. It keeps changing direction,
so there must be acceleration toward the center. That's the yellow arrow.
Its length is proportional to speed squared divided by the turning radius.

Now double the speed, keeping the same turn. The required turning acceleration
becomes four times as large. This is why a bend that is easy at low speed can
ask too much of the tires when approached faster.

### Constructing the acceleration circle

Let's put those demands in a different picture. These axes measure acceleration.
Horizontal means speeding up or braking; vertical means turning left or right.

The blue and yellow components form a right triangle. The green diagonal is
the combined acceleration. Its length follows directly from Pythagoras.

In this model, the tire force is limited by friction: mu times the normal
force, which we take to be the car's weight. Divide by mass, and the acceleration
limit is mu g. Draw every possible direction with that same maximum length.
The endpoints trace a circle.

A point inside the circle is an acceleration the tires can supply. Move it
around and both components change, but their combined length stays within the
limit.

### Asking for more than the tires can supply

Now keep the turning demand fixed and ask for more throttle. The red point is
what we requested. Once it leaves the circle, the green point can no longer
follow it. The requested acceleration exceeds the available grip.

Here is what this simulator does: it keeps the turning component, then clips
the forward component at the circle. The red gap shows the acceleration that
was requested but cannot be applied. Asking for more doesn't increase the
force the tires can provide.

Cut the circle at this turning acceleration. The blue chord shows how much
forward acceleration or braking is still possible. Pythagoras gives its
half-width: the square root of mu g squared minus turning acceleration squared.

Increase the turning demand. That chord shrinks. Ease the turn, and it grows
again. Move to the left side and the same constraint applies to braking.
Braking and cornering share the available tire force too.

### Lower friction

Now leave the request alone and reduce friction. The circle itself shrinks.
A request that previously fit can become impossible without changing the
controls at all. The green point moves inward to the new boundary.

Exceeding the circle doesn't change mu in this simulator. It means the demand
is larger than the available grip. Lower mu is a separate change to that grip.
The model represents the limit by capping accelerations; it doesn't simulate
the detailed motion of a slipping tire.

Finally, ask for more turning than even the top of the circle allows. The
kernel caps the turning component first. At that limit, no forward acceleration
remains.

### The effect on the path

What does that cap do to the car? The red curve shows the requested turn.
The green car travels at the same speed, but it cannot change heading quickly
enough to follow that curve. Its path is wider.

The minimum turning radius grows with speed squared and shrinks as available
grip increases. Reducing speed reduces the acceleration needed to follow a
particular bend.

### Six small updates

Now we can compute the next state. One simulation step lasts a sixtieth of a
second. The kernel divides that into six smaller intervals.

Each interval uses RK4: four estimates of the rates of change, with the middle
two carrying twice the weight. Every estimate applies the grip limits we just
constructed. Steering advances through the interval as well.

The displacement here is magnified so we can see each update. One produces
the starting state for the next, so these six updates happen in order within
each car's work item.

## RewardAndRespawn

After the move, the map answers two different questions. First: how close is
the nearest wall? The distance transform stores that answer at every pixel.
The yellow circle makes the meaning of that number visible.

Second: where are we along the track? A separate lookup gives the nearest
centerline waypoint. Moving forward through those waypoints earns progress.
The reward also subtracts a near-wall penalty and the squared sideways offset
from the centerline. Progress and wall penalties include speed-dependent
weights; the displayed equation shows their structure.

The car has size. Subtract half its diagonal from the wall distance to get
clearance. If that clearance becomes negative, the kernel marks the episode
done and replaces the reward with minus twenty-five.

It also respawns the car immediately: a new waypoint, zero speed and steering,
and fresh friction and wheelbase scales within fifteen percent of nominal.
A ten-thousand-step timeout also resets, but doesn't by itself apply the crash
penalty. The observations will describe the respawned car.

## WarpLidar

Now take one ray. How far can it travel before it hits a wall?

At its starting point, ask the distance field for the nearest wall. Notice
that the nearest wall can be beside the ray. Still, a circle of that radius
contains no wall, so we can advance by one radius along the ray.

At the new point, ask again. Draw the new circle, advance by its radius, and
repeat. Every blue segment was a yellow radius. Add those lengths to get the
range measurement.

This drawing uses smooth walls to show the geometry. The implementation reads
pixel distances from the precomputed EDT, stops on a zero sample or at its
range limit, and converts the total back into meters. The range is capped at
twenty meters.

Now spread the rays over a two-hundred-seventy-degree fan. We draw nineteen;
the simulator computes one hundred and eight. Each ray can do its own march.
So this time, a work item has two indices: car i and beam j.

That ray writes one entry in the observation row, after steering and speed.
On CUDA the distance field is read through a texture; on CPU it is read from
an array. Both use this same marching procedure.

## TheHandoff

Here are the two divisions of work next to each other. On the left, one item
per car, with six dependent physics updates. On the right, one item per car
and beam, each with its own ray-march loop.

Physics must finish before lidar: the rays need the updated pose, or the
respawned pose if the episode just ended.

Put that into one environment call. Copy in the actions, run physics, run
lidar, then advance the random clock used by future respawns. Return the
observations, rewards, and done flags.

When Warp and Torch use the same CUDA device, those output tensors view the
same storage. The launches use Torch's current stream, preserving the order.

Each car runs its six physics updates in order. Cars and lidar rays supply
the independent work.
