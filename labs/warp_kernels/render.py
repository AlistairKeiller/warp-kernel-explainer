"""Render, assemble, and review the silent film from the project directory.

uv run python labs/warp_kernels/render.py --preview
uv run python labs/warp_kernels/render.py
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
SCENES = ["Opening", "VehicleStep", "RewardAndRespawn", "WarpLidar", "TheHandoff"]
TITLES = ["One car, many worlds", "Move: motion, grip, and small updates",
          "Score: progress, penalties, and a fresh start", "Sense: safe jumps along a ray",
          "Return: parallel work and shared outputs"]


def run(args):
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True)


def duration(path):
    return float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], text=True))


def timestamp(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02}:{seconds:02}"


def assemble(directory):
    concat = directory / "chapters.ffconcat"
    concat.write_text("ffconcat version 1.0\n" + "".join(f"file '{scene}.mp4'\n" for scene in SCENES))
    metadata = [";FFMETADATA1", "title=One step in warporacer"]
    chapters = ["# Film chapters", "", "The film is silent; explanations appear in the frame.", "",
                "| Start | Chapter |", "| --- | --- |"]
    offset = 0.
    for scene, title in zip(SCENES, TITLES):
        length = duration(directory / f"{scene}.mp4")
        metadata.extend(["[CHAPTER]", "TIMEBASE=1/1000", f"START={round(1000*offset)}",
                         f"END={round(1000*(offset+length))}", f"title={title}"])
        chapters.append(f"| {timestamp(offset)} | {title} |")
        offset += length
    chapters.extend(["", f"Total duration: {timestamp(offset)}.", ""])
    (directory / "chapters.md").write_text("\n".join(chapters))
    with tempfile.TemporaryDirectory() as temp:
        meta_path = Path(temp) / "chapters.ffmetadata"
        meta_path.write_text("\n".join(metadata) + "\n")
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
             "-safe", "0", "-i", concat, "-i", meta_path, "-map_metadata", "1",
             "-map_chapters", "1", "-c", "copy", "-movflags", "+faststart",
             directory / "WarpKernelTour.mp4"])
    print(f"Film ready: {directory / 'WarpKernelTour.mp4'} ({timestamp(offset)})", flush=True)


def review(directory):
    """Sample every caption beat, rather than hoping fixed-interval frames cover it."""
    target = directory / "review"
    target.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        for scene in SCENES:
            data = json.loads((directory / f"{scene}.beats.json").read_text())
            captions = data["captions"]
            for page in range(math.ceil(len(captions) / 12)):
                page_count = min(12, len(captions)-page*12)
                sheet = Image.new("RGB", (1920, 390*math.ceil(page_count/3)), "#0B0B10")
                draw = ImageDraw.Draw(sheet)
                for slot, index in enumerate(range(page*12, min((page+1)*12, len(captions)))):
                    caption = captions[index]
                    end = caption.get("end", captions[index+1]["time"] if index+1 < len(captions) else data["duration"])
                    time = min(data["duration"]-.15, end-.3)
                    frame = Path(temp) / "frame.jpg"
                    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", time,
                         "-i", directory / f"{scene}.mp4", "-frames:v", "1",
                         "-vf", "scale=640:360", frame])
                    x, y = (slot % 3)*640, (slot // 3)*390
                    with Image.open(frame) as image:
                        sheet.paste(image, (x, y))
                    draw.text((x+12, y+365), f"{scene}  {timestamp(time)}  ·  beat {index+1}",
                              fill="#B4B4C0", font_size=18)
                sheet.save(target / f"{scene}-{page+1:02}.jpg", quality=90)
    print(f"Review sheets: {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="720p, 15 fps, in videos/preview")
    parser.add_argument("--assemble-only", action="store_true", help="Reuse existing chapter renders")
    parser.add_argument("--review", action="store_true", help="Also extract every caption beat")
    parser.add_argument("--manimgl", help="Path to an existing ManimGL executable")
    args = parser.parse_args()
    directory = ROOT / "videos" / ("preview" if args.preview else "")
    directory.mkdir(parents=True, exist_ok=True)
    if not args.assemble_only:
        command = [args.manimgl or shutil.which("manimgl")] if args.manimgl or shutil.which("manimgl") else ["uv", "run", "manimgl"]
        run(command + ["labs/warp_kernels/main.py", *SCENES, "-w", "-q", "-r",
                       "1280x720" if args.preview else "1920x1080", "--fps",
                       "15" if args.preview else "30", "--video_dir", str(directory)])
    assemble(directory)
    if args.review:
        review(directory)


if __name__ == "__main__":
    main()
