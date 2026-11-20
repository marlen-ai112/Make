#!/usr/bin/env python3
"""Монтаж по заданиям: из файлов jobs/*.json в репозитории и/или из папки Jobs на Google Drive.

Возможности задания (.json):
  inputs, output            - обязательно
  trim {start,end}          - обрезка по времени
  resolution "1080x1920"    - формат кадра
  cut_silence {threshold_db, min_silence, pad} - вырезать паузы
  auto_subtitles {language, model}             - распознать речь и вшить красивые субтитры
  subtitles "Input/x.srt"   - готовые субтитры с Диска
  denoise_audio, normalize_audio, preset, crf, fps, extra_args
"""
import importlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

REMOTE = os.environ.get("RCLONE_REMOTE", "gdrive")


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd))[:300], flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


# ---------- анализ видео ----------

def probe(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:stream_side_data=rotation:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    st = d["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    rot = 0
    for sd in st.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if abs(rot) % 180 == 90:
        w, h = h, w
    return w, h, float(d["format"]["duration"])


def audio_levels(path):
    """Громкость звука по окнам 50 мс (RMS, дБ)."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-ac", "1", "-af",
         "aresample=16000,asetnsamples=n=800:p=0,astats=metadata=1:reset=1,"
         "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
         "-f", "null", "-"],
        capture_output=True, text=True)
    vals = []
    for m in re.finditer(r"lavfi\.astats\.Overall\.RMS_level=(\S+)", r.stdout):
        try:
            v = float(m.group(1))
        except ValueError:
            v = -90.0
        vals.append(max(v, -90.0))
    return vals


def auto_threshold(path):
    """Порог тишины подбирается по самому видео: между фоновым шумом и громкостью речи."""
    vals = sorted(audio_levels(path))
    if len(vals) < 20:
        print("Мало данных о звуке, порог -32 дБ", flush=True)
        return -32.0
    noise = vals[int(len(vals) * 0.05)]
    speech = vals[int(len(vals) * 0.90)]
    thr = -32.0 if speech - noise < 8 else noise + 0.45 * (speech - noise)
    thr = max(-55.0, min(-22.0, thr))
    print(f"Звук: фон {noise:.1f} дБ, речь {speech:.1f} дБ, порог тишины {thr:.1f} дБ", flush=True)
    return thr


def speech_segments(path, duration, cfg):
    """Находит участки с речью. Паузы длиннее min_silence вырезаются, по краям речи остаётся pad секунд."""
    thr = cfg.get("threshold_db", "auto")
    if thr == "auto":
        thr = auto_threshold(path)
    min_sil = cfg.get("min_silence", 0.4)
    pad = cfg.get("pad", 0.1)
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn",
         "-af", f"silencedetect=noise={thr}dB:d={min_sil}", "-f", "null", "-"],
        capture_output=True, text=True)
    starts = [max(0.0, float(x)) for x in re.findall(r"silence_start: (-?[\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    if len(starts) > len(ends):
        ends.append(duration)
    print(f"Найдено пауз: {len(starts)}", flush=True)
    speech, cur = [], 0.0
    for s, e in zip(starts, ends):
        if s > cur:
            speech.append((cur, s))
        cur = max(cur, e)
    if cur < duration:
        speech.append((cur, duration))
    padded = [(max(0.0, a - pad), min(duration, b + pad)) for a, b in speech]
    merged = []
    for a, b in padded:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    merged = [(a, b) for a, b in merged if b - a >= 0.3]
    kept = sum(b - a for a, b in merged)
    print(f"Пауз вырезано: оставлено {kept:.1f} с из {duration:.1f} с, кусков: {len(merged)}", flush=True)
    return merged or [(0.0, duration)]


def cut_graph(segs, with_video):
    lines, labels = [], ""
    for i, (a, b) in enumerate(segs):
        if with_video:
            lines.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{i}]")
        lines.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[a{i}]")
        labels += (f"[v{i}]" if with_video else "") + f"[a{i}]"
    out = "[vc][ac]" if with_video else "[ac]"
    lines.append(f"{labels}concat=n={len(segs)}:v={1 if with_video else 0}:a=1{out}")
    return lines


# ---------- субтитры ----------

def ensure_whisper():
    try:
        from faster_whisper import WhisperModel
        return WhisperModel
    except ImportError:
        pass
    target = "/tmp/pydeps"
    spec = os.environ.get("WHISPER_PIP_SPEC", "faster-whisper")
    run([sys.executable, "-m", "pip", "install", "--quiet", "--target", target, spec])
    sys.path.insert(0, target)
    importlib.invalidate_caches()
    from faster_whisper import WhisperModel
    return WhisperModel


def transcribe(wav, language, model_name):
    WhisperModel = ensure_whisper()
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(wav), language=language, vad_filter=True,
                                      word_timestamps=True, beam_size=5)
    words = []
    for seg in segments:
        for w in (seg.words or []):
            t = w.word.strip()
            if t:
                words.append((w.start, w.end, t))
    print(f"Распознано слов: {len(words)}", flush=True)
    return words


def build_chunks(words, max_chars, max_words=7, max_dur=3.2, max_gap=0.6):
    chunks, cur = [], []
    for w in words:
        if cur:
            text = " ".join(x[2] for x in cur)
            too_long = len(text) + 1 + len(w[2]) > max_chars or len(cur) >= max_words
            too_slow = w[1] - cur[0][0] > max_dur or w[0] - cur[-1][1] > max_gap
            ends_sentence = text.endswith((".", "?", "!", "…"))
            if too_long or too_slow or ends_sentence:
                chunks.append(cur)
                cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    out = []
    for i, c in enumerate(chunks):
        start, end = c[0][0], c[-1][1] + 0.15
        if i + 1 < len(chunks):
            end = min(end, chunks[i + 1][0][0])
        out.append((start, max(end, start + 0.3), " ".join(x[2] for x in c)))
    return out


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, (a, b, text) in enumerate(chunks, 1):
            f.write(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n\n")


def subtitle_filter(w, h):
    """Крупный жирный белый текст с чёрной обводкой снизу по центру. Размер зависит от формата кадра."""
    vertical = h > w
    px = 0.05 * min(w, h)
    font_size = round(px * 288 / h, 1)
    margin_v = 52 if vertical else 20
    style = (f"FontName=DejaVu Sans,Bold=1,FontSize={font_size},"
             f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,"
             f"Outline=2.5,Shadow=0,Alignment=2,MarginV={margin_v}")
    return f"subtitles=subs.srt:force_style='{style}'"


# ---------- основной монтаж ----------

def process(job_file):
    job = json.loads(pathlib.Path(job_file).read_text(encoding="utf-8"))
    work = pathlib.Path(tempfile.mkdtemp())

    inputs = []
    for i, src in enumerate(job["inputs"]):
        dst = work / f"in{i}{pathlib.Path(src).suffix}"
        run(["rclone", "copyto", f"{REMOTE}:{src}", dst])
        inputs.append(dst)
    n = len(inputs)

    cut_cfg = job.get("cut_silence")
    auto_cfg = job.get("auto_subtitles")
    if cut_cfg and n != 1:
        raise ValueError("cut_silence работает только с одним входным видео")
    if auto_cfg and n != 1:
        raise ValueError("auto_subtitles работает только с одним входным видео")
    if cut_cfg is True:
        cut_cfg = {}
    if auto_cfg is True:
        auto_cfg = {}

    w, h = (job.get("resolution") or "1920x1080").split("x")
    fps = int(job.get("fps", 30))
    fit = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
           f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1")

    parts = []
    vf, af = [], []
    segs = None

    if n > 1:
        for i in range(n):
            parts.append(f"[{i}:v]{fit},fps={fps},format=yuv420p[v{i}]")
            parts.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}]")
        parts.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[vc][ac]")
        vin, ain = "vc", "ac"
    elif cut_cfg is not None and cut_cfg is not False:
        src_w, src_h, duration = probe(inputs[0])
        segs = speech_segments(inputs[0], duration, cut_cfg)
        parts += cut_graph(segs, True)
        vin, ain = "vc", "ac"
        if job.get("resolution"):
            vf.append(fit)
    else:
        vin, ain = "0:v", "0:a"
        if job.get("resolution"):
            vf.append(fit)

    # размеры итогового кадра (нужны для оформления субтитров)
    if job.get("resolution"):
        out_w, out_h = int(w), int(h)
    else:
        out_w, out_h, _ = probe(inputs[0])

    if auto_cfg is not None and auto_cfg is not False:
        wav = work / "speech.wav"
        if segs:
            graph = work / "audio_graph.txt"
            graph.write_text(";\n".join(cut_graph(segs, False)), encoding="utf-8")
            run(["ffmpeg", "-y", "-i", inputs[0], "-filter_complex_script", graph,
                 "-map", "[ac]", "-vn", "-ac", "1", "-ar", "16000", wav])
        else:
            run(["ffmpeg", "-y", "-i", inputs[0], "-vn", "-ac", "1", "-ar", "16000", wav])
        words = transcribe(wav, auto_cfg.get("language", "ru"), auto_cfg.get("model", "small"))
        max_chars = 26 if out_h > out_w else 40
        chunks = build_chunks(words, max_chars)
        write_srt(chunks, work / "subs.srt")
        vf.append(subtitle_filter(out_w, out_h))
    elif job.get("subtitles"):
        run(["rclone", "copyto", f"{REMOTE}:{job['subtitles']}", work / "subs.srt"])
        vf.append("subtitles=subs.srt")

    if job.get("denoise_audio"):
        af.append("afftdn")
    if job.get("normalize_audio", True):
        af.append("loudnorm=I=-16:TP=-1.5:LRA=11")

    parts.append(f"[{vin}]{','.join(vf) or 'null'}[v]")
    parts.append(f"[{ain}]{','.join(af) or 'anull'}[a]")

    graph_file = work / "graph.txt"
    graph_file.write_text(";\n".join(parts), encoding="utf-8")

    out = work / "out.mp4"
    cmd = ["ffmpeg", "-y"]
    for p in inputs:
        cmd += ["-i", p]
    cmd += ["-filter_complex_script", graph_file, "-map", "[v]", "-map", "[a]"]

    trim = job.get("trim") or {}
    if trim.get("start"):
        cmd += ["-ss", trim["start"]]
    if trim.get("end"):
        cmd += ["-to", trim["end"]]

    cmd += ["-c:v", "libx264", "-preset", job.get("preset", "veryfast"),
            "-crf", job.get("crf", 20), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]
    cmd += job.get("extra_args", [])
    cmd.append(out)

    run(cmd, cwd=work)
    run(["rclone", "copyto", out, f"{REMOTE}:{job['output']}"])
    print(f"Готово: {job['output']}")


# ---------- очередь заданий на Google Drive ----------

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
