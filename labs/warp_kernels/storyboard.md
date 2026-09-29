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
| `Opening` | A car travels around a large track; that world shrinks into twelve separate worlds; a single environment is selected. | One physics work item owns one independent car. |
| `VehicleStep` | Resolve velocity into components; construct the turn center from wheel normals; follow the tangent and inward acceleration around a turn; double speed; construct the acceleration circle; move a request beyond it; animate the remaining-acceleration chord, braking, reduced friction, and turn saturation; compare requested and capped paths; integrate six RK4 substeps. | The combined acceleration has a limit. The kernel clips the request to what the tires can supply. |
| `RewardAndRespawn` | A wall-distance circle is replaced by a waypoint projection; progress and lateral offset become reward terms; the car footprint crosses the wall and respawns. | Wall distance and waypoint lookup answer different questions. |
| `WarpLidar` | A ray advances exactly one clearance radius at a time; its segments become a range with a brace parallel to the ray; a fan expands the work into `(car, beam)` indices and observation slots. | Marching is sequential within a ray, independent across rays. |
| `TheHandoff` | Rows of six dependent steps become a grid of beam items; launch order resolves into shared output storage. | Physics precedes sensing; Warp and Torch can share the outputs. |

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
- The steering construction uses enlarged wheel angles so the triangle is
  readable; these are not the simulator's 0.4189 rad steering limit.
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
waypoint projections, reset behavior, and the final observation strip. Rendered
chapters and the concatenated tour are silent; narration is maintained separately.
