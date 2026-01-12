#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests


def format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def extract_audio(video_path: Path, wav_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-y",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_service_ready(service_url: str, timeout_s: int = 3600, poll_s: int = 2) -> None:
    start = time.time()
    while True:
        try:
            r = requests.get(f"{service_url}/health", timeout=5)
            if r.status_code == 200 and r.json().get("ready"):
                return
        except Exception:
            pass

        if time.time() - start > timeout_s:
            raise TimeoutError(f"service not ready after {timeout_s}s: {service_url}")
        time.sleep(poll_s)


def init_models(service_url: str, asr_model_size: str, translation_model: str, use_gpu: bool) -> None:
    try:
        requests.post(
            f"{service_url}/init",
            json={
                "asr_model_size": asr_model_size,
                "translation_model": translation_model,
                "use_gpu": bool(use_gpu),
            },
            timeout=10,
        )
    except Exception:
        # If init fails (service down / already loading), let wait_for_service_ready handle it.
        pass


def transcribe(service_url: str, audio_path: Path) -> dict:
    with audio_path.open("rb") as f:
        r = requests.post(f"{service_url}/transcribe", files={"audio": f}, timeout=3600)
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(data.get("error") or "transcribe failed")
    return data


def translate_text(service_url: str, text: str, source_lang: str, target_lang: str, max_retries: int = 3) -> str:
    if not text.strip():
        return ""

    last_err = None
    for _ in range(max_retries):
        try:
            r = requests.post(
                f"{service_url}/translate",
                json={"text": text, "source_language": source_lang, "target_language": target_lang},
                timeout=90,
            )
            if r.status_code == 200:
                return r.json().get("translated_text", "") or ""
            last_err = RuntimeError(f"HTTP {r.status_code}: {r.text}")
        except Exception as e:
            last_err = e
        time.sleep(2)

    # Fallback: return source text if translation keeps failing
    return text


def write_srt(segments: list[dict], srt_path: Path, translation_only: bool) -> None:
    with srt_path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n")
            if translation_only:
                f.write(f"{seg.get('translated', seg.get('text', '')).strip()}\n\n")
            else:
                f.write(f"{seg.get('text', '').strip()}\n")
                f.write(f"{seg.get('translated', seg.get('text', '')).strip()}\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate 1.mp4 to Chinese subtitles (no polish).")
    parser.add_argument("--video", default="1.mp4", help="Video file path (default: 1.mp4)")
    parser.add_argument("--service-url", default="http://127.0.0.1:50515", help="Service URL")
    parser.add_argument("--target", default="zh", help="Target language (default: zh)")
    parser.add_argument("--source", default="auto", help="Source language (default: auto)")
    parser.add_argument(
        "--translation-only",
        action="store_true",
        help="Only write translated lines (default: bilingual)",
    )
    parser.add_argument(
        "--asr-model-size",
        default="reazonspeech",
        help="ASR model size for /init (reazonspeech / tiny / base / small / medium / large-v3)",
    )
    parser.add_argument(
        "--translation-model",
        default="SakuraLLM/Sakura-7B",
        help="Translation model for /init",
    )
    parser.add_argument("--use-gpu", action="store_true", help="Request GPU in /init (if available)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    out_srt = video_path.with_name(f"{video_path.stem}_{args.target}.srt")

    # Kick off model loading (safe even if already loading), then wait until ready.
    init_models(args.service_url, args.asr_model_size, args.translation_model, args.use_gpu)
    wait_for_service_ready(args.service_url, timeout_s=3600)

    tmp_wav_path = None
    try:
        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav_path = Path(tmp.name)
        extract_audio(video_path, tmp_wav_path)

        asr = transcribe(args.service_url, tmp_wav_path)
        detected_lang = asr.get("language") or "en"
        source_lang = detected_lang if args.source == "auto" else args.source

        segments = asr.get("segments") or []
        for seg in segments:
            seg["translated"] = translate_text(args.service_url, seg.get("text", ""), source_lang, args.target)

        write_srt(segments, out_srt, translation_only=bool(args.translation_only))
        print(f"OK: {out_srt}")
        return 0
    finally:
        if tmp_wav_path and tmp_wav_path.exists():
            tmp_wav_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
