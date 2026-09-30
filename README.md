# How Warp moves the cars

A silent ManimGL explainer for the Warp kernels in the neighboring
[`warporacer`](../warporacer) project.

Follow one car through **Move → Score → Sense → Return**, then see how that
same work scales to thousands of independent environments. The opening explains
state, action, and kernel before introducing code. A small guide stays visible
throughout the film. Each diagram answers one question, and transitions keep
the previous result visible while explaining why the next question follows.

The physics chapter builds steering geometry and the shared grip budget from
visible motion. A short clarification connects the geometric radius to the
kernel's heading-rate request and shows the straight-wheel case. RK4 uses a
time ruler, faint trial cars, and four samples before committing one actual
update. Reward terms appear individually. Lidar reuses the familiar wall-distance
circle, then turns rays into an observation profile. The ending assembles the
output buffers and returns to the original car.

[Watch the rendered tour](videos/WarpKernelTour.mp4) (14:04, 1080p, 30 fps, silent).
[Chapter timestamps](videos/chapters.md) are also embedded in the MP4.
Explanations appear in the frame; captions wrap and stay visible long enough
to read before being replaced. No audio is needed to follow the sequence.

## Render

This project uses the ManimGL fork used by `UCI_F1tenth_slides`.
Install the usual ManimGL requirements, including LaTeX and FFmpeg.

```bash
uv sync
uv run python labs/warp_kernels/render.py --review

# Faster 720p review, stored separately in videos/preview/
uv run python labs/warp_kernels/render.py --preview --review

# Render an individual chapter
uv run manimgl labs/warp_kernels/main.py VehicleStep -w -r 1920x1080 --fps 30 --video_dir videos

# Reassemble existing chapter files with navigation markers
uv run python labs/warp_kernels/render.py --assemble-only
```

`render.py` renders all five chapters and joins them into the tour with MP4
chapter markers. `--review` extracts a contact sheet for every caption beat into
`videos/review/`. An existing environment can be used with
`--manimgl /absolute/path/to/manimgl`. For interactive preview, omit `-w` from
the individual scene command; `-p` enables presenter pauses.

## Project layout

```text
labs/
└── warp_kernels/
    ├── main.py         # Manim scenes, geometry, captions, and reading time
    ├── render.py       # Render, assemble, add chapters, extract review frames
    ├── narration.md    # Companion explanation, organized by scene and beat
    └── storyboard.md   # Visual intent, reference study, and source-code map
```

The `labs/<topic>/` directory follows the self-contained lab layout in
`UCI_F1tenth_slides/labs/`. `main.py` follows the 3Blue1Brown repository's
scene-first working format, with one visual idea developed at a time.
The [reference study](labs/warp_kernels/storyboard.md#reference-study) links the
specific local sources and the presentation decisions they informed.
These are original drawings and text; no scenes, assets, or helper code were
copied from `3b1b-videos`. The companion script explains the same story in prose;
the silent film's captions are authored alongside its animations.

## Technical scope

The film follows the bicycle simulator at warporacer commit
`e16e4473cb86e6ae2e1fa1b078e5a1e04b20d7f2`, specifically `Env.step()` in
`warporacer/warporacer/sim.py`:

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
