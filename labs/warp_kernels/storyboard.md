# Scene notes

The film develops one visible object at a time. Color follows meaning:
yellow is the selected car, local physics, or a clearance query; blue is
motion, centerline information, or completed lidar measurements; red marks a
request beyond the grip limit or a collision; green marks applied acceleration. Labels sit beside their geometry. The frame clears when the explanation moves to a different subject.

## Reference study

The local `3b1b-videos` sources informed the presentation:

- `_2024/manim_demo/lorenz.py`: calculate trajectories, then animate objects
  along them; equations describe the motion the viewer is seeing.
- `_2024/transformers/network_flow.py`: introduce a concrete instance, expand
  it into a repeated structure, and select part of that structure to explain.

These chapters use original geometry and narration, with no imported reference
assets or repository-specific helpers. They use ManimGL `Tex` for mathematics,
restrained color, staged reveals, and transformations that express relationships.

## Chapters

| Scene | Visual development | Main takeaway |
| --- | --- | --- |
| `Opening` | A car travels around a large track; that world shrinks into twelve separate worlds; the worlds advance at different rates; one environment enlarges beside its state and action. | One physics work item owns one independent car. |
| `VehicleStep` | Resolve velocity into components; construct the turn center from wheel normals; derive heading rate from arc length; compare translated velocity arrows and take the small-interval limit; substitute heading rate to obtain inward acceleration; double speed; construct the acceleration circle; move a request beyond it; animate the remaining-acceleration chord, braking, reduced friction, and turn saturation; compare requested and capped paths; construct four RK4 trial states and add their weighted displacement vectors; integrate six real substeps. | The combined acceleration has a limit. The kernel clips the request to what the tires can supply. |
| `RewardAndRespawn` | A wall-distance circle is replaced by a waypoint projection; progress and lateral offset become reward terms; a growing square and live parabola explain the offset penalty; the car footprint crosses the wall and respawns. | Wall distance and waypoint lookup answer different questions. |
| `WarpLidar` | A ray advances exactly one clearance radius at a time; its segments become a range with a brace parallel to the ray; a fan expands the work into `(car, beam)` indices and observation slots; the rays straighten into a range profile that changes as the car turns. | Marching is sequential within a ray, independent across rays. |
| `TheHandoff` | Pulses traverse six physics updates, then spread into a ray grid; row and column guides select one output slot; an abbreviated output table fills first from physics and then lidar; Warp and Torch point to the same data. | Physics precedes sensing; the output tensors can view Warp's existing buffers. |

## Deliberate abstractions

- Twelve worlds stand for a representative batch of 8,000. They are separate
  environments, not cars interacting on one shared track. Work indices do not
  represent physical GPU cores.
- The opening track is an ellipse. The reward scene is a straight local stretch.
  Neither is a screenshot of a simulator map.
- The integration example computes the simulator's RK4 scheme at `h=1/360 s`,
  starting at 4 m/s with 0.24 rad steering, 0.4 rad/s steering rate, 1 m/s²
  requested acceleration, 0.3302 m wheelbase, and nominal mu = 1.0489.
  Every derivative evaluation caps yaw and longitudinal acceleration exactly
  as `deriv()` does, including the 0.5 m/s denominator floor. These initial
  conditions activate the grip limit. Screen displacement is magnified 145 times.
- The four-stage RK4 construction uses an enlarged 0.45 s interval, initial
  speed 2 m/s, steering 0.25 rad, steering rate 0.15 rad/s, and requested
  acceleration 1 m/s². It uses the same derivative and RK4 helper as the
  actual six-substep example. The diagram draws position components of the
  derivative vectors; the full state also contains heading and speed.
  Trial positions are not successive committed car positions. The arrows in
  the final sum have lengths proportional to `h * weight * derivative / 6`.
- The steering construction uses enlarged wheel angles so the triangle is
  readable; these are not the simulator's 0.4189 rad steering limit.
- The heading-rate derivation uses arc length `Δs = R Δψ`, with angles in
  radians, then divides by `Δt` to obtain `v = R ψ̇`. The velocity-change
  construction translates equal-length tangent vectors to a shared origin.
  Its chord is approximately `v Δψ` for a small angle; shrinking the interval
  gives `a_lateral = v ψ̇`, then substitution gives `v²/R`. The demonstration
  is a constant-speed left turn. For a general turn the inward acceleration
  magnitude is `v |ψ̇|`; signed components depend on the turn direction.
- The turning diagram uses schematic lengths and playback timing. Its inward
  acceleration arrow grows fourfold when the velocity arrow doubles.
- The grip plot uses acceleration units normalized by the initial mu g. It
  first caps the lateral request, then clips the longitudinal request to the
  remaining chord. It is not a radial projection of the requested vector.
  The full circle describes the envelope above the kernel's low-speed floor;
  it is not a tire slip or load-transfer simulation.
- A request outside the circle exceeds available grip. It does not lower mu.
  The separate friction sweep changes mu from its starting value to 0.65
  times that value, then to 0.85. This is a teaching comparison, not a claim
  about the simulator's ±15% respawn randomization.
- The requested and capped turn paths use illustrative radii 2 and 3.2 at
  equal speed and travel distance. The requested lateral acceleration is
  1.6 times the cap. The wider curve illustrates limited yaw rate, not a
  separately simulated skid.
- Reward terms show structure, not all coefficients: signed wrapped waypoint
  progress and wall proximity also have speed factors. The collision example
  sets reward to −25. Timeout alone does not imply that penalty.
- The centerline demonstration holds the other reward terms aside. It plots
  signed lateral offset against its square; the implementation computes the
  absolute offset before squaring, with the same result. The displayed
  coefficient is the source's `CENTER_COEF = 1.0`.
- Clearance uses a bounding circle with radius half the car diagonal. The
  footprint diagram is schematic. A crash or 10,000 steps immediately resets;
  friction and wheelbase scales vary by ±15%.
- The sphere-march drawing uses the exact continuous distance to three corridor
  walls. Every displayed jump equals that distance, and each circle touches a
  nearest wall. The source instead samples a raster EDT in pixel coordinates,
  stops at a zero sample (or map boundary/range cap), and converts to meters.
  The smooth illustration stops once the residual is under 0.015 drawing units.
- Nineteen fan rays and a small output strip stand for 108 beams over 270° and
  an observation row of 110 values. The sensor is ahead of the car center.
- The range profile uses the same nineteen illustrative rays as the fan.
  Their lengths share one fixed display scale. During rotation, ray origins
  follow the sensor mount and every wall intersection is recomputed each frame.
  Beam angles are relative to the car, not fixed world headings.
- The ending shows four environments and ten beam columns as a readable sample.
  The dotted separator marks launch order, not a barrier inside a kernel.
  Selecting `(i=2, j=6)` gives `obs[2, 8]` because the first two entries hold
  steering and speed.
- The output table contains illustrative values, not a captured simulator batch.
  Its terminal row has reward −25, done = 1, and zero steering/speed to show
  immediate respawn. Range values appear in the later lidar pass. The green
  frame groups three output arrays; it does not claim they share one contiguous
  allocation. Warp and Torch view the same respective output buffers.
- CUDA uses a nearest-sampled texture; CPU uses direct EDT array reads.
  The final shared-storage claim applies when Warp and Torch share a CUDA device.
  A different Torch device uses explicit output copies.

## Source map

Paths are relative to the sibling `warporacer` repository.

| Implementation | Subject |
| --- | --- |
| `warporacer/sim.py:30` | time step and six substeps |
| `warporacer/sim.py:54` | bicycle derivative and friction limit |
| `warporacer/sim.py:63` | four RK4 slopes and steering midpoints |
| `warporacer/sim.py:75` | array and texture ray marching |
| `warporacer/sim.py:112` | one-car work item |
| `warporacer/sim.py:152` | EDT, waypoint progress, reward, collision and reset |
| `warporacer/sim.py:202` | `(i, j)` lidar work item and observation writes |
| `warporacer/sim.py:275` | persistent buffers and Torch views |
| `warporacer/sim.py:325` | padded map stacks and recorded launches |
| `warporacer/sim.py:412` | action copy, ordered launches, returned outputs |
| `warporacer/track.py:19` | precomputed track maps |

## Render review

Render all five scenes, then inspect representative frames from every visual
beat. Check velocity components, the fourfold turning-acceleration change, the
requested/applied vector separation, horizontal clipping at fixed lateral
acceleration, the shrinking chord, negative longitudinal demand when braking,
the friction sweep, the wider capped turn, equation margins, the six highlighted
intervals, the ray-aligned range brace, wall tangencies,
waypoint projections, the fourfold square area, signed-offset symmetry, reset
behavior, ray-to-bar correspondence, and the rotating observation profile. Rendered
chapters and the concatenated tour are silent; narration is maintained separately.
