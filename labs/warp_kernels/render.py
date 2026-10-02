"""Render the chapters, join them into the film, and extract review frames.

uv run python labs/warp_kernels/render.py --preview --review   # 720p, into videos/preview/
uv run python labs/warp_kernels/render.py                      # 1080p, into videos/
uv run python labs/warp_kernels/render.py --slides             # click-through deck, into videos/slides/
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
SOURCE = "labs/warp_kernels/main.py"
FILM = "WarpKernelTour.mp4"
CHAPTERS = [
    ("Opening", "One car, many worlds"),
    ("VehicleStep", "Move: motion, grip, and small updates"),
    ("RewardAndRespawn", "Score: progress, penalties, and a fresh start"),
    ("WarpLidar", "Sense: safe jumps along a ray"),
    ("TheHandoff", "Return: parallel work and shared outputs"),
]


def run(args):
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True)


def duration(path):
    return float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], text=True))


def timestamp(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02}:{seconds:02}"


def render(directory, preview, manimgl, slideshow=False):
    command = [manimgl] if manimgl else ["uv", "run", "manimgl"]
    env = os.environ | ({"WARP_SLIDES": "1"} if slideshow else {})
    subprocess.run(command + [
        SOURCE, *(scene for scene, _ in CHAPTERS), "-w", "-q",
        "-r", "1280x720" if preview else "1920x1080", "--fps", "15" if preview else "30",
        "--video_dir", str(directory),
    ], cwd=ROOT, check=True, env=env)


def assemble(directory):
    """Concatenate the chapters and embed chapter markers."""
    concat = directory / "chapters.ffconcat"
    concat.write_text("ffconcat version 1.0\n" + "".join(f"file '{scene}.mp4'\n" for scene, _ in CHAPTERS))
    metadata = [";FFMETADATA1", "title=One step in warporacer"]
    table = ["# Film chapters", "", "The film is silent; explanations appear in the frame.", "",
             "| Start | Chapter |", "| --- | --- |"]
    offset = 0.
    for scene, title in CHAPTERS:
        length = duration(directory / f"{scene}.mp4")
        metadata += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={round(1000 * offset)}",
                     f"END={round(1000 * (offset + length))}", f"title={title}"]
        table.append(f"| {timestamp(offset)} | {title} |")
        offset += length
    table += ["", f"Total duration: {timestamp(offset)}.", ""]
    (directory / "chapters.md").write_text("\n".join(table))
    with tempfile.TemporaryDirectory() as temp:
        meta_path = Path(temp) / "chapters.ffmetadata"
        meta_path.write_text("\n".join(metadata) + "\n")
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
             "-i", concat, "-i", meta_path, "-map_metadata", "1", "-map_chapters", "1",
             "-c", "copy", "-movflags", "+faststart", directory / FILM])
    print(f"Film ready: {directory / FILM} ({timestamp(offset)})", flush=True)


def review(directory, per_sheet=12):
    """Grab a frame at the end of every caption beat and tile them into contact sheets."""
    target = directory / "review"
    target.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        frame = Path(temp) / "frame.jpg"
        for scene, _ in CHAPTERS:
            data = json.loads((directory / f"{scene}.beats.json").read_text())
            captions = data["captions"]
            # Sample just before the next caption starts crossfading in.
            ends = [c["time"] - 1 for c in captions[1:]] + [data["duration"]]
            for page in range(math.ceil(len(captions) / per_sheet)):
                slots = list(range(page * per_sheet, min((page + 1) * per_sheet, len(captions))))
                sheet = Image.new("RGB", (1920, 390 * math.ceil(len(slots) / 3)), "#0B0B10")
                draw = ImageDraw.Draw(sheet)
                for slot, index in enumerate(slots):
                    time = max(captions[index]["time"], min(data["duration"] - .15, ends[index]))
                    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", time,
                         "-i", directory / f"{scene}.mp4", "-frames:v", "1", "-vf", "scale=640:360", frame])
                    x, y = (slot % 3) * 640, (slot // 3) * 390
                    with Image.open(frame) as image:
                        sheet.paste(image, (x, y))
                    draw.text((x + 12, y + 365), f"{scene}  {timestamp(time)}  ·  beat {index + 1}",
                              fill="#B4B4C0", font_size=18)
                sheet.save(target / f"{scene}-{page + 1:02}.jpg", quality=90)
    print(f"Review sheets: {target}", flush=True)


def cut_slides(directory):
    """Split every chapter at its slide boundaries into one clip per slide."""
    clips = directory / "clips"
    clips.mkdir(exist_ok=True)
    manifest = []
    for scene, title in CHAPTERS:
        data = json.loads((directory / f"{scene}.beats.json").read_text())
        bounds = data["slides"] + [data["duration"]]
        for k, (start, end) in enumerate(zip(bounds, bounds[1:])):
            clip = clips / f"{scene}-{k:02}.mp4"
            run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", start, "-to", end,
                 "-i", directory / f"{scene}.mp4", "-an", "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", clip])
            manifest.append({"src": f"clips/{clip.name}", "chapter": title})
    player = (Path(__file__).parent / "player.html").read_text()
    (directory / "index.html").write_text(player.replace("__SLIDES__", json.dumps(manifest)))
    print(f"Deck ready: {directory / 'index.html'} ({len(manifest)} slides)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="720p at 15 fps, into videos/preview/")
    parser.add_argument("--slides", action="store_true", help="Build the click-through deck in videos/slides/")
    parser.add_argument("--assemble-only", action="store_true", help="Reuse the existing chapter renders")
    parser.add_argument("--review", action="store_true", help="Also extract a frame per caption beat")
    parser.add_argument("--manimgl", help="Path to a manimgl executable (default: uv run manimgl)")
    args = parser.parse_args()
    directory = ROOT / "videos" / ("slides" if args.slides else "preview" if args.preview else "")
    directory.mkdir(parents=True, exist_ok=True)
    if not args.assemble_only:
        render(directory, args.preview, args.manimgl, slideshow=args.slides)
    if args.slides:
        cut_slides(directory)
    else:
        assemble(directory)
    if args.review:
        review(directory)


if __name__ == "__main__":
    main()
