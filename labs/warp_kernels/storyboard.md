# Scene notes

The film follows one car through Move, Score, Sense, and Return. The opening
names the inputs in everyday language, each chapter answers a few questions in
turn, and the ending returns to the same track to recap the four phases. A guide
strip at the top of every frame shows which phase the viewer is in.

Each picture develops one idea at a time. Color follows meaning: yellow is the
selected car and local physics, blue is motion and measurement, green is what
the simulator applies, and red is a request it cannot honor or a collision.
Labels sit beside their geometry; equations collect in a column on the right.
Between sections, a bridge isolates the result just found, states what it means,
and asks the next question before the next picture appears.

The film is silent. Captions explain what to notice, wrap in a reserved area at
the bottom, and are held for `0.7 + words / 3.1` seconds before anything replaces
them. Motion is introduced one quantity at a time: turn the velocity, then
lengthen it; turn the wheel further once; keep the request fixed while grip
changes.

## Reference study

The scene code follows the working conventions of the public
[3Blue1Brown repository](https://github.com/3b1b/videos): scenes are classes
whose `construct` method calls one section at a time, groups are built
declaratively and laid out with `arrange` and `next_to`, LaTeX uses raw
strings, and dynamic geometry is driven by updaters. Specific scenes informed
specific decisions:

- `LorenzAttractor` (`_2024/manim_demo/lorenz.py`): compute trajectories first,
  then animate objects along them. The RK4 and six-substep pictures are drawn
  from states computed with the simulator's own scheme.
- `HighLevelNetworkFlow` (`_2024/transformers/network_flow.py`): introduce one
  concrete instance, expand it into a repeated structure, select one part to
  explain. This is car → twelve worlds → environment five.
- `AttentionPatterns` (`_2024/transformers/attention.py`): carry a familiar
  object into a more abstract representation. Ray → range → observation bar.
- `WaysToCombine` (`_2022/convolutions/discrete.py`): color quantities in
  equations to match their visual counterparts.
- `DifferentConceptions` (`_2016/eola/chapter1.py`): give vectors a concrete
  meaning before operating on them symbolically. The velocity arrow precedes its
  components, and the four RK4 sample arrows precede their weighted sum.

All drawings and text are original; nothing is copied from that repository.

## Chapters

| Scene | Sections | Main takeaway |
| --- | --- | --- |
| `Opening` | One car on a track; its state and action in plain words; a preview of move, score, and sense; twelve separate worlds; environment five brought forward with the kernel named. | A kernel repeats one program across independent work items. |
| `VehicleStep` | Velocity components; the bicycle turn center and `R = L / tan δ`, then `ψ̇ = v tan δ / L`; inward acceleration `v²/R` and the fourfold cost of doubling speed; the grip circle with a request clipped into it, a shrinking circle, and the turning cap; the wider capped path; four RK4 samples combined into one update; six substeps over 1/60 s. | Turning and throttle share one grip budget, and the kernel clips requests to it before integrating. |
| `RewardAndRespawn` | The wall-distance map and the nearest-waypoint map; signed waypoint progress; the three reward terms; a square and a parabola for the offset penalty; the clearance circle crossing a wall, then respawn. | Two precomputed maps answer two different questions about the car's position. |
| `WarpLidar` | A ray advancing one safe radius at a time, summed into a range; a fan of rays indexed by `(i, j)`; the fan straightened into a bar profile that changes as the car turns. | Marching is sequential within a ray and independent across rays. |
| `TheHandoff` | Pulses crossing six updates per car, then fanning into beam items; one item's output slot; the observation, reward, and done buffers filled by physics then lidar; the RNG tick; Warp and Torch pointing at the same storage. | Physics precedes sensing, and Torch views Warp's buffers without copies on a shared CUDA device. |

## Deliberate abstractions

- Twelve worlds stand for a representative batch of 8,000. They are separate
  environments, not cars sharing one track. Work indices are tasks, not GPU
  cores.
- The opening track is an ellipse. The reward scene is a straight corridor.
  Neither is a screenshot of a simulator map.
- The steering drawing uses enlarged wheel angles so the triangle is readable;
  the simulator's steering limit is 0.4189 rad. `R` is a geometric aid; the
  kernel computes the heading rate `v tan δ / L` directly. The film states
  `ψ̇ = v / R` as the rate of going around a circle rather than deriving it
  from arc length.
- The inward acceleration is presented as `v ψ̇ = v² / R` for a constant-speed
  turn, with the arrow lengths schematic. The chord-and-arc derivation of `Δv`
  stays in the companion script.
- The grip plot normalizes acceleration by the initial `μ g`. Like `deriv()`, it
  caps the lateral request first and then clips the longitudinal request to the
  remaining chord; it is not a radial projection. The circle describes the
  envelope above the kernel's 0.5 m/s speed floor and does not model tire slip.
- A request outside the circle exceeds the available grip; it does not change
  `μ`. The separate friction sweep scales `μ` to 0.65 and then 0.85 of its
  starting value as a teaching comparison, not the simulator's ±15% jitter.
- The requested and capped paths use radii 2 and 3.2 at equal speed and
  travel; the wider curve illustrates a limited yaw rate, not a skid.
- The four-stage RK4 construction uses an enlarged interval `h = 0.45 s`
  starting at 2 m/s with 0.25 rad steering, 0.15 rad/s steering rate, and
  1 m/s² requested acceleration, computed with the same derivative and stage
  structure as the kernel. The diagram shows position components; the state
  also carries heading and speed. Trial positions are not committed car moves.
- The six-substep picture integrates the simulator's scheme at `h = 1/360 s`
  from 4 m/s, 0.24 rad steering, 0.4 rad/s steering rate, 1 m/s² requested
  acceleration, the nominal wheelbase, and nominal `μ`, including the grip caps
  in every evaluation. Screen displacement is magnified 145 times.
- Reward terms show structure, not coefficients: progress and wall proximity
  also carry speed factors. The crash example sets reward to −25; a timeout
  alone ends the episode without that penalty.
- The car moves smoothly while the selected waypoint, progress arrow, and count
  snap at nearest-waypoint boundaries. The count is relative to the starting
  waypoint; the kernel uses the wrapped integer index change since the previous
  step. Equal spacing illustrates the lookup table.
- The centerline demonstration plots signed offset against its square; the
  implementation squares the absolute offset, with the same result.
  `CENTER_COEF` is 1.0.
- Clearance uses a bounding circle of radius half the car diagonal. A crash or
  10,000 steps respawns immediately with friction and wheelbase scales within
  ±15% of nominal.
- The sphere-march drawing uses exact distances to three straight walls, so
  every circle touches its nearest wall. The kernel samples a raster distance
  field in pixels, stops on a zero sample or the range cap, and converts to
  meters. The drawing stops once the residual is under 0.015 units.
- Nineteen fan rays stand for 108 beams over 270°. The sensor sits ahead of
  the car center, beam angles are relative to the car, and every wall
  intersection is recomputed each frame during the turn.
- The handoff shows four environments and ten beam columns. The dotted
  separator marks launch order, not a barrier inside a kernel. Selecting
  `(i = 2, j = 6)` gives `obs[2, 8]` because the first two entries hold steering
  and speed.
- The output table holds illustrative values; its third row has just crashed
  and respawned, so steering and speed read zero. The green frame groups three
  arrays and does not claim one contiguous allocation. CUDA samples the
  distance field through a texture; CPU reads the array directly.
- The RNG tick is a one-element int32 array on the simulation device. After
  lidar, `bump_kernel` increments it once per `Env.step()`, and respawn seeds
  its randomness from the seed and `tick[0] * num_envs + i`. The values 42 and
  43 are illustrative.
- Zero-copy output views apply when Warp and Torch share a CUDA device. With
  Torch on a different device the outputs are copied once per step.

## Source map

Paths are relative to the sibling `warporacer` repository.

| Implementation | Subject |
| --- | --- |
| `warporacer/sim.py:30` | time step and six substeps |
| `warporacer/sim.py:55` | bicycle derivative and friction limit |
| `warporacer/sim.py:65` | four RK4 slopes and steering midpoints |
| `warporacer/sim.py:75` | array and texture ray marching |
| `warporacer/sim.py:107` | RNG counter increment |
| `warporacer/sim.py:112` | one-car work item |
| `warporacer/sim.py:152` | wall distance, waypoint progress, reward, collision, reset |
| `warporacer/sim.py:202` | `(i, j)` lidar work item and observation writes |
| `warporacer/sim.py:280` | persistent buffers and Torch views |
| `warporacer/sim.py:325` | padded map stacks and recorded launches |
| `warporacer/sim.py:417` | ordered launches and the output copy fallback |
| `warporacer/sim.py:428` | action copy and returned outputs |
| `warporacer/track.py:19` | precomputed track maps |

## Slideshow cut

`render.py --slides` renders with `WARP_SLIDES=1`, which caps every pause at
0.35 s because the presenter sets the pace. Each caption starts a new slide, so
a click plays the caption's crossfade and the motion that belongs to it, then
holds on the finished picture. Headings and the guide strip act as slide
titles. The deck is one clip per slide plus `player.html`.

## Render review

`render.py --review` tiles one frame per caption into contact sheets under
`videos/review/`. Check: the velocity components, the right triangle at the turn
center, the inward arrow growing fourfold, the clipped request and the chord,
the shrinking circle, the turning cap, the wider capped path, the four trial
states and the chained increments, the six highlighted intervals, the waypoint
snapping, the fourfold square, the symmetric parabola, the clearance circle and
reset, the ray-aligned range brace, wall tangencies, the ray-to-bar
correspondence, and the rotating profile. Captions must not overlap diagrams.
