#!/usr/bin/env python3
"""Монтаж по файлу задания: скачивает видео с Google Drive, монтирует ffmpeg, загружает результат обратно."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

REMOTE = os.environ.get("RCLONE_REMOTE", "gdrive")


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def process(job_file):
    job = json.loads(pathlib.Path(job_file).read_text(encoding="utf-8"))
    work = pathlib.Path(tempfile.mkdtemp())

    inputs = []
    for i, src in enumerate(job["inputs"]):
        dst = work / f"in{i}{pathlib.Path(src).suffix}"
        run(["rclone", "copyto", f"{REMOTE}:{src}", dst])
        inputs.append(dst)
    n = len(inputs)

    w, h = (job.get("resolution") or "1920x1080").split("x")
    fps = int(job.get("fps", 30))
    fit = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
           f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1")

    parts = []
    vf, af = [], []

    if n > 1:
        for i in range(n):
            parts.append(f"[{i}:v]{fit},fps={fps},format=yuv420p[v{i}]")
            parts.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}]")
        parts.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[vc][ac]")
        vin, ain = "vc", "ac"
    else:
        vin, ain = "0:v", "0:a"
        if job.get("resolution"):
            vf.append(fit)

    if job.get("subtitles"):
        run(["rclone", "copyto", f"{REMOTE}:{job['subtitles']}", work / "subs.srt"])
        vf.append("subtitles=subs.srt")
    if job.get("denoise_audio"):
        af.append("afftdn")
    if job.get("normalize_audio", True):
        af.append("loudnorm=I=-16:TP=-1.5:LRA=11")

    parts.append(f"[{vin}]{','.join(vf) or 'null'}[v]")
    parts.append(f"[{ain}]{','.join(af) or 'anull'}[a]")

    out = work / "out.mp4"
    cmd = ["ffmpeg", "-y"]
    for p in inputs:
        cmd += ["-i", p]
    cmd += ["-filter_complex", ";".join(parts), "-map", "[v]", "-map", "[a]"]

    trim = job.get("trim") or {}
    if trim.get("start"):
        cmd += ["-ss", trim["start"]]
    if trim.get("end"):
        cmd += ["-to", trim["end"]]

    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", job.get("crf", 20),
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart"]
    cmd += job.get("extra_args", [])
    cmd.append(out)

    run(cmd, cwd=work)
    run(["rclone", "copyto", out, f"{REMOTE}:{job['output']}"])
    print(f"Готово: {job['output']}")


if __name__ == "__main__":
    files = sys.argv[1:]
    if not files:
        print("Нет заданий для выполнения.")
        sys.exit(0)
    for f in files:
        process(f)
