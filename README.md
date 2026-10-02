# How Warp moves the cars

A silent ManimGL explainer for the Warp kernels in the neighboring
[`warporacer`](../warporacer) project.

Follow one car through **Move → Score → Sense → Return**, then see how the same
work scales to thousands of independent environments. The opening explains
state, action, and kernel before any code appears, and a small guide strip
stays visible throughout. Each picture answers one question; between sections,
the result just found stays on screen while the next question is asked.

The physics chapter builds steering geometry and the shared grip budget from
visible motion, then integrates with RK4: four samples, one update, six
updates per frame. Reward terms appear one at a time. Lidar reuses the
wall-distance idea, then turns rays into an observation profile. The ending
assembles the output buffers and returns to the original car.

[Watch the rendered tour](videos/WarpKernelTour.mp4) (11:23, 1080p, 30 fps, silent).
[Chapter timestamps](videos/chapters.md) are also embedded in the MP4.
Captions wrap in a reserved area and stay on screen long enough to read.

## Render

The project depends on upstream [ManimGL](https://github.com/3b1b/manim),
which needs LaTeX and FFmpeg installed.

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

`render.py` renders the five chapters and joins them into the tour with MP4
chapter markers. `--review` tiles one frame per caption into contact sheets
under `videos/review/`. Pass `--manimgl /path/to/manimgl` to use another
installation. For interactive work, omit `-w` from the single-chapter command;
`-p` adds presenter pauses.

## Project layout

```text
labs/
└── warp_kernels/
    ├── main.py         # Scenes: stagecraft, geometry, captions, reading time
    ├── render.py       # Render, assemble, add chapters, extract review frames
    ├── narration.md    # Companion explanation in prose, by scene
    └── storyboard.md   # Visual intent, abstractions, and source-code map
```

`main.py` follows the working conventions of the public
[3Blue1Brown repository](https://github.com/3b1b/videos): one scene per
chapter, one section per method, declarative groups laid out with `arrange`
and `next_to`, raw-string LaTeX, and updaters for dynamic geometry. The
[reference study](labs/warp_kernels/storyboard.md#reference-study) names the
scenes that informed specific presentation choices. All drawings and text are
original.

## Technical scope

The film follows the bicycle simulator at warporacer commit
`e16e4473cb86e6ae2e1fa1b078e5a1e04b20d7f2`, specifically `Env.step()` in
`warporacer/warporacer/sim.py`:

1. A Torch action batch is copied into the persistent Warp action buffer.
2. `step_kernel` runs once per environment. Each work item integrates one car
   for six RK4 substeps under the friction-circle cap, then writes physics
   state, reward, and `done`, respawning on a crash or timeout.
3. `lidar_kernel` (CPU) or `lidar_tex_kernel` (CUDA) runs once per car and
   beam. Each ray sphere-marches the precomputed wall-distance field.
4. `bump_kernel` advances the device RNG counter used by later respawns.
5. The environment returns observation, reward, and `done` tensors. On a
   shared CUDA device these are Torch views of the Warp buffers, and the
   kernels launch on Torch's current stream.

The film distinguishes the two kinds of work: a car's six substeps and a
beam's march loop are sequential inside their work item; cars and beams are
the dimensions spread across Warp threads. Work items are tasks, not physical
GPU cores. On CPU, `Env` splits environment ranges across a thread pool and
reads the distance field directly. The wall-distance field and the
nearest-waypoint table are prepared when tracks load, so the hot step only
reads them.

The storyboard's source map cites exact lines in the sibling simulator. If you
move this explainer away from `warporacer`, update those relative paths.
