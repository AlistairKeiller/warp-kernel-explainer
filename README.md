# How Warp moves the cars

A short, scene-based ManimGL explainer for the Warp kernels in the neighboring
[`warporacer`](../warporacer) project.

The physics chapter follows velocity components, steering geometry, turning
acceleration, and the tire grip limit. Arc length explains heading change per
second; comparing velocity arrows explains the small-angle approximation before
deriving the turning-acceleration formula. Requested and applied acceleration are
shown separately. The circle contracts when friction drops, and a pair of
trajectories shows how a capped turn rate widens the car's path. Six RK4 updates
then advance the state using the simulator's grip limits. An enlarged RK4
interval shows where its four derivative samples come from and how their
weighted vectors combine.

The lidar chapter follows a ray through clearance circles that touch the nearest
wall. The range brace and its equation run parallel to the ray. The ray fan
then becomes a range profile that responds to the car's heading. The reward
chapter shows progress snapping between waypoint indices and links sideways
offset to a square's area and a live penalty graph.
The ending follows work items into an output table, showing which values physics
and lidar write, where the device RNG counter lives and when it advances,
and how Torch views the same buffers. Narration and scene notes
live beside the source.

[Watch the rendered tour](videos/WarpKernelTour.mp4) (9:33, 1080p, silent).

## Render a scene

This project uses the ManimGL fork used by `UCI_F1tenth_slides`.

```bash
uv sync
uv run manimgl labs/warp_kernels/main.py \
  Opening VehicleStep RewardAndRespawn WarpLidar TheHandoff \
  -w -r 1920x1080 --fps 30 --video_dir videos
ffmpeg -y -f concat -safe 0 -i videos/chapters.ffconcat -c copy videos/WarpKernelTour.mp4
```

Install the usual ManimGL requirements, including LaTeX and FFmpeg. The explicit
`--video_dir` keeps the chapter files in this project's `videos/` directory.
For an interactive preview, omit `-w`; `-p` enables presenter pauses.
The last command joins the five silent chapter renders in order. The voice-over
in `labs/warp_kernels/narration.md` is an editorial draft; record it and retime
holds before producing a narrated cut.

## Project layout

```text
labs/
└── warp_kernels/
    ├── main.py         # Renderable Manim scenes and small geometry helpers
    ├── narration.md    # Voice-over draft, organized by scene and beat
    └── storyboard.md  # Visual intent, timing, and source-code map
```

The `labs/<topic>/` directory follows the self-contained lab layout in
`UCI_F1tenth_slides/labs/`: scene source and the material needed to present it
stay together. `main.py` follows the 3Blue1Brown videos repository's scene-first
working format: named scenes are authored as animation code, with one visual
idea developed at a time. These are original drawings and narration; no scenes,
assets, or helper code were copied from `3b1b-videos`.

## Technical scope

The film follows `Env.step()` in `warporacer/warporacer/sim.py`:

1. A Torch action batch is copied into the persistent Warp action buffer.
2. `step_kernel` runs once per environment. Each logical item integrates one
   car for six RK4 substeps, then writes physics state, reward, and `done`.
3. `lidar_kernel` (CPU) or `lidar_tex_kernel` (CUDA) runs once per
   car-and-beam. Each ray sphere-marches the precomputed wall-distance field.
4. `bump_kernel` advances the device RNG clock used on respawn.
5. The environment returns observation, reward, and `done` tensors. On the
   same CUDA device, these are Torch views of the Warp buffers and the kernels
   launch on Torch's current stream.

The presentation distinguishes the two kinds of parallel work: a car's RK4
substeps and an individual beam's ray-march loop are sequential within their
logical work item; cars and beams are the dimensions spread across Warp threads.
These are logical work items, not a claim that 8,000 physical GPU cores execute
at once. CUDA schedules them in blocks (128 threads per block here). On CPU,
`Env` splits environment ranges across a thread pool and uses direct array
lookups for lidar.
It also points out that the EDT and nearest-waypoint lookup table are prepared
when tracks load, so the hot step only reads those maps.

The source map and the script cite exact locations in the sibling simulator.
If you move this explainer away from `warporacer`, update those relative paths.
