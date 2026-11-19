#!/usr/bin/env python3
"""Монтаж по заданиям: из файлов jobs/*.json в репозитории и/или из папки Jobs на Google Drive."""
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


def drive_job_names():
    r = subprocess.run(
        ["rclone", "lsf", f"{REMOTE}:Jobs", "--files-only", "--include", "*.json"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("Папка Jobs недоступна:", r.stderr.strip())
        return []
    return [n.strip() for n in r.stdout.splitlines() if n.strip()]


def poll_drive():
    names = drive_job_names()
    if not names:
        print("Новых заданий на Диске нет.")
        return 0
    failed = 0
    for name in names:
        local = pathlib.Path(tempfile.mkdtemp()) / name
        run(["rclone", "copyto", f"{REMOTE}:Jobs/{name}", local])
        # сразу убираем задание из очереди, чтобы следующий запуск не взял его повторно
        run(["rclone", "moveto", f"{REMOTE}:Jobs/{name}", f"{REMOTE}:Jobs/Processing/{name}"])
        try:
            process(local)
            dest = "Done"
        except Exception as e:
            print("ОШИБКА:", e, flush=True)
            failed += 1
            dest = "Failed"
            subprocess.run(["rclone", "rcat", f"{REMOTE}:Jobs/Failed/{name}.error.txt"],
                           input=str(e).encode("utf-8"))
        run(["rclone", "moveto", f"{REMOTE}:Jobs/Processing/{name}", f"{REMOTE}:Jobs/{dest}/{name}"])
    return failed


if __name__ == "__main__":
    args = sys.argv[1:]
    use_drive = "--drive" in args
    files = [a for a in args if a != "--drive"]
    failed = 0
    for f in files:
        try:
            process(f)
        except Exception as e:
            print("ОШИБКА:", e, flush=True)
            failed += 1
    if use_drive:
        failed += poll_drive()
    if not files and not use_drive:
        print("Нет заданий для выполнения.")
    sys.exit(1 if failed else 0)
