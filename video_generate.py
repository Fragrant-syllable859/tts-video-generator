"""
Video Generator - Create subtitle-overlaid videos from script documents.

Pipeline: Word document -> sentence splitting -> ElevenLabs TTS with timestamps
-> ASS subtitle generation -> FFmpeg video composition (background image + audio + subtitles + BGM).

Usage:
    python video_generate.py                     # Generate all episodes
    python video_generate.py --single 29         # Generate only episode 29
    python video_generate.py --batch 10          # Generate up to 10 episodes
    python video_generate.py --list-backgrounds  # List available background images
    python video_generate.py --preview-subs      # Preview subtitle timestamps
"""

import os
import re
import sys
import json
import random
import subprocess
import tempfile
import requests
import base64
from docx import Document
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================
# Configuration (all configurable via .env)
# ============================================================
API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")

BASE_DIR = os.getenv("BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.getenv("AUDIO_DIR", os.path.join(BASE_DIR, "audio_output"))
BG_DIR = os.getenv("BG_DIR", os.path.join(BASE_DIR, "backgrounds"))
BGM_DIR = os.getenv("BGM_DIR", os.path.join(BASE_DIR, "bgm"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "video_output"))
DOCX_PATH = os.getenv("DOCX_PATH", os.path.join(BASE_DIR, "scripts.docx"))

START_FROM = int(os.getenv("START_FROM", "1"))
BGM_VOLUME = float(os.getenv("BGM_VOLUME", "0.12"))
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = os.getenv("FFPROBE_PATH", "ffprobe")

# Text extraction tags (customize for your language/format)
TEXT_TAG = os.getenv("TEXT_TAG", "【Text】")
END_TAGS = os.getenv("END_TAGS", "").split(",") if os.getenv("END_TAGS") else []

# Subtitle style configuration
SUBTITLE_FONT = os.getenv("SUBTITLE_FONT", "Arial")
TITLE_FONTSIZE = int(os.getenv("TITLE_FONTSIZE", "60"))
SUB_FONTSIZE = int(os.getenv("SUB_FONTSIZE", "48"))
SUB_MAX_LINE_CHARS = int(os.getenv("SUB_MAX_LINE_CHARS", "16"))
TITLE_MAX_LINE_CHARS = int(os.getenv("TITLE_MAX_LINE_CHARS", "14"))

# Video resolution
VIDEO_WIDTH = int(os.getenv("VIDEO_WIDTH", "1080"))
VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "1920"))

# ElevenLabs model
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5")
TTS_STABILITY = float(os.getenv("TTS_STABILITY", "0.5"))
TTS_SIMILARITY_BOOST = float(os.getenv("TTS_SIMILARITY_BOOST", "0.75"))

# Sentence splitting pattern (regex for sentence-ending punctuation)
SENTENCE_SPLIT_PATTERN = os.getenv("SENTENCE_SPLIT_PATTERN", r'(?<=[。！？!?.])')
# ============================================================


def build_background_map(all_eps):
    """Automatically assign background images to episodes in a round-robin fashion."""
    bg_files = sorted(
        f for f in os.listdir(BG_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    if not bg_files:
        return {}
    eps_sorted = sorted(all_eps)
    return {ep: bg_files[i % len(bg_files)] for i, ep in enumerate(eps_sorted)}


# ---------- Parse Word Document ----------

def parse_docx(docx_path):
    """Extract episode titles and text content from a Word document."""
    doc = Document(docx_path)
    titles = {}
    scripts = {}
    current_ep = None

    for para in doc.paragraphs:
        raw = para.text
        text = raw.strip().replace('\u200b', '')
        if not text:
            continue

        # Detect episode number + title line
        ep_match = re.match(r'^(\d+)\s*\S', text)
        if ep_match and TEXT_TAG not in text:
            current_ep = int(ep_match.group(1))
            title = re.sub(r'^\d+\s*', '', text).strip()
            titles[current_ep] = title
            continue

        # Content paragraph with text tag
        if TEXT_TAG in text and current_ep is not None:
            pattern = TEXT_TAG + r'\s*(.*?)'
            if END_TAGS:
                end_pattern = '|'.join(re.escape(t) for t in END_TAGS if t.strip())
                pattern += f'(?={end_pattern}|$)'
            else:
                pattern += '$'
            ja_match = re.search(pattern, text, re.DOTALL)
            if ja_match:
                content = ja_match.group(1).strip().replace('\u200b', '')
                scripts[current_ep] = content

    return titles, scripts


# ---------- Subtitle Timing: ElevenLabs Character-Level Timestamps ----------

def split_sentences(text):
    """Split text into sentences based on configured punctuation pattern."""
    lines = text.replace('\n', '').strip()
    parts = re.split(SENTENCE_SPLIT_PATTERN, lines)
    return [p.strip() for p in parts if p.strip()]


def wrap_subtitle(text, max_line=None):
    """
    Wrap subtitle text to fit within max_line characters per line (2 lines max).
    Tries to break at punctuation marks for natural reading.
    """
    if max_line is None:
        max_line = SUB_MAX_LINE_CHARS

    NO_LINE_START = set('」』）】〕〉》"\' 、。，！？…)].!?,;:')
    text = text.strip()
    n = len(text)

    if n <= max_line:
        return text

    min_pos = max(1, n - max_line)
    max_pos = min(n - 1, max_line)

    if min_pos > max_pos:
        return text[:n // 2] + "\\N" + text[n // 2:]

    target = max(min_pos, min(max_pos, n // 2))

    best_pos = target
    search_range = max_pos - min_pos + 1
    for offset in range(search_range):
        for i in [target - offset, target + offset]:
            if min_pos <= i <= max_pos and text[i - 1] in '。！？、，,…!?.;:':
                best_pos = i
                break
        else:
            continue
        break

    while best_pos < max_pos and text[best_pos] in NO_LINE_START:
        best_pos += 1

    return text[:best_pos] + "\\N" + text[best_pos:]


def get_sentence_timing(ep, text, sentences):
    """
    Call ElevenLabs /with-timestamps to get audio + character-level timestamps.
    Saves both MP3 and timing JSON from the same synthesis (100% aligned).
    Returns [(start_time, end_time, sentence_text), ...]
    """
    cache_path = os.path.join(AUDIO_DIR, f"{ep}_timing.json")
    mp3_out = os.path.join(AUDIO_DIR, f"{ep}.mp3")

    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return [(i[0], i[1], i[2]) for i in json.load(f)]

    if not API_KEY:
        raise ValueError("ELEVENLABS_API_KEY is not set. Please configure it in .env file.")

    print(f"      -> ElevenLabs with-timestamps...", end=" ", flush=True)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/with-timestamps"
    headers = {"xi-api-key": API_KEY, "Content-Type": "application/json"}
    send_text = "".join(s.strip() for s in sentences)
    payload = {
        "text": send_text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {"stability": TTS_STABILITY, "similarity_boost": TTS_SIMILARITY_BOOST}
    }
    resp = requests.post(url, json=payload, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"ElevenLabs API Error {resp.status_code}: {resp.text}")

    data = resp.json()

    # Save audio (from the same synthesis as timestamps, 100% aligned)
    with open(mp3_out, 'wb') as f:
        f.write(base64.b64decode(data["audio_base64"]))

    alignment = data["alignment"]
    chars = alignment["characters"]
    char_starts = alignment["character_start_times_seconds"]
    char_ends = alignment["character_end_times_seconds"]
    print("Done")

    # Map character-level timestamps to sentences
    SKIP = set('\n\r\u200b\u200c\u200d\ufeff ')
    result = []
    char_idx = 0
    for sentence in sentences:
        clean = sentence.replace('\u200b', '').strip()
        if not clean:
            continue
        while char_idx < len(chars) and chars[char_idx] in SKIP:
            char_idx += 1
        if char_idx >= len(chars):
            break
        start_idx = char_idx
        end_idx = min(char_idx + len(clean) - 1, len(chars) - 1)
        result.append((
            round(char_starts[start_idx], 3),
            round(char_ends[end_idx], 3),
            sentence
        ))
        char_idx += len(clean)

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ---------- Generate ASS Subtitles ----------

def sec_to_ass(t):
    """Convert seconds to ASS time format H:MM:SS.cc"""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    cs = round((s - int(s)) * 100)
    return f"{h}:{m:02d}:{int(s):02d}.{cs:02d}"


def build_ass(title, segments, duration):
    """
    Generate an ASS subtitle file with:
      Title - top-area bold text with thick outline
      Sub   - upper-center area text with background box
    """
    WHITE = "&H00FFFFFF"
    BLACK = "&H00000000"
    TRANS = "&HFF000000"

    def wrap_title(t, max_line=None):
        if max_line is None:
            max_line = TITLE_MAX_LINE_CHARS
        NO_LINE_START_T = set('」』）】〕〉》"\' 、。，！？…)].!?,;:')
        t = t.strip()
        n = len(t)
        if n <= max_line:
            return t
        min_pos = max(1, n - max_line)
        max_pos = min(n - 1, max_line)
        if min_pos > max_pos:
            return t[:n // 2] + "\\N" + t[n // 2:]
        target = max(min_pos, min(max_pos, n // 2))
        best_pos = target
        search_range = max_pos - min_pos + 1
        for offset in range(search_range):
            for i in [target - offset, target + offset]:
                if min_pos <= i <= max_pos and t[i - 1] in '。！？、，,…」』!?.;:':
                    best_pos = i
                    break
            else:
                continue
            break
        while best_pos < max_pos and t[best_pos] in NO_LINE_START_T:
            best_pos += 1
        return t[:best_pos] + "\\N" + t[best_pos:]

    title_display = wrap_title(title)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {VIDEO_WIDTH}
PlayResY: {VIDEO_HEIGHT}
WrapStyle: 1
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Title,{SUBTITLE_FONT},{TITLE_FONTSIZE},{WHITE},{TRANS},{BLACK},{TRANS},-1,0,0,0,100,100,2,0,1,6,2,8,80,80,250,1
Style: Sub,{SUBTITLE_FONT},{SUB_FONTSIZE},{BLACK},{TRANS},{WHITE},{TRANS},0,0,0,0,100,100,1,0,1,4,1,8,110,110,720,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []
    # Title: fixed at the top for the entire duration
    t_end = sec_to_ass(duration + 0.5)
    events.append(f"Dialogue: 0,0:00:00.00,{t_end},Title,,0,0,0,,{title_display}")

    # Subtitles: sentence by sentence
    for start, end, text in segments:
        wrapped = wrap_subtitle(text.strip().replace('\n', ''))
        events.append(f"Dialogue: 0,{sec_to_ass(start)},{sec_to_ass(end)},Sub,,0,0,0,,{wrapped}")

    return header + "\n".join(events) + "\n"


# ---------- FFmpeg Video Composition ----------

def get_audio_duration(mp3_path):
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [FFPROBE_PATH, "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", mp3_path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def pick_random_bgm():
    """Pick a random audio file from the BGM directory."""
    if not os.path.isdir(BGM_DIR):
        return None
    files = [f for f in os.listdir(BGM_DIR) if f.lower().endswith(('.mp3', '.wav', '.m4a', '.aac'))]
    if not files:
        return None
    return os.path.join(BGM_DIR, random.choice(files))


def make_video(bg_path, mp3_path, ass_path, output_path, bgm_path=None):
    """Compose video using FFmpeg: background image + audio + subtitles + optional BGM."""
    vf = (
        f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},"
        f"ass={ass_path}"
    )

    if bgm_path:
        cmd = [
            FFMPEG_PATH, "-y",
            "-loop", "1", "-framerate", "30", "-i", bg_path,
            "-i", mp3_path,
            "-stream_loop", "-1", "-i", bgm_path,
            "-filter_complex",
                f"[1:a][2:a]amix=inputs=2:duration=first:weights=1 {BGM_VOLUME}[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            output_path
        ]
    else:
        cmd = [
            FFMPEG_PATH, "-y",
            "-loop", "1", "-framerate", "30", "-i", bg_path,
            "-i", mp3_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            output_path
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr[-500:]}")


# ---------- Main Pipeline ----------

def main(only_ep=None, force=False, max_count=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    titles, scripts = parse_docx(DOCX_PATH)

    all_eps = sorted(ep for ep in scripts if ep >= START_FROM)
    background_map = build_background_map(all_eps)

    if only_ep is not None:
        target_eps = [only_ep] if only_ep in scripts else []
        print(f"Single mode: processing episode {only_ep}\n")
    else:
        target_eps = all_eps
        if max_count:
            print(f"Batch mode: up to {max_count} episodes ({len(target_eps)} total)\n")
        else:
            print(f"{len(target_eps)} episodes to process (from episode {START_FROM})\n")

    for ep in target_eps:
        mp3_path = os.path.join(AUDIO_DIR, f"{ep}.mp3")
        out_path = os.path.join(OUTPUT_DIR, f"{ep}.mp4")

        if os.path.exists(out_path) and not force:
            print(f"  Episode {ep}: already exists, skipping")
            continue

        bg_file = background_map.get(ep)
        if not bg_file:
            print(f"  Episode {ep}: no background image assigned, skipping")
            continue

        bg_path = os.path.join(BG_DIR, bg_file)
        if not os.path.exists(bg_path):
            print(f"  Episode {ep}: background not found: {bg_path}, skipping")
            continue

        title = titles.get(ep, f"Episode {ep}")
        print(f"  Episode {ep}: processing...")

        # 1. Split sentences + get ElevenLabs timestamps (also generates MP3)
        print(f"    1) Getting subtitle timestamps...")
        text = scripts.get(ep, "")
        sentences = split_sentences(text)
        segments = get_sentence_timing(ep, text, sentences)
        duration = get_audio_duration(mp3_path)
        print(f"      -> {len(segments)} sentences, duration {duration:.1f}s")

        # 2. Generate ASS subtitles
        print(f"    2) Generating subtitles...", end=" ", flush=True)
        ass_content = build_ass(title, segments, duration)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ass',
                                         delete=False, encoding='utf-8') as f:
            f.write(ass_content)
            ass_path = f.name
        print(f"Done")

        # 3. Compose video (with random BGM)
        bgm_path = pick_random_bgm()
        bgm_name = os.path.basename(bgm_path) if bgm_path else "None"
        print(f"    3) Composing video (BGM: {bgm_name})...", end=" ", flush=True)
        try:
            make_video(bg_path, mp3_path, ass_path, out_path, bgm_path=bgm_path)
            print(f"Done -> {out_path}")
            if only_ep is not None:
                print(f"    4) Opening video preview...")
                subprocess.run(["open", out_path])
        except RuntimeError as e:
            print(f"Failed!\n{e}")
            continue
        finally:
            os.unlink(ass_path)

        if max_count is not None:
            max_count -= 1
            if max_count <= 0:
                print(f"\nBatch complete. Review results, then run again for next batch.")
                return

    print("\nAll done!")


# ---------- Utility Commands ----------

def list_backgrounds():
    """List all available background images."""
    files = sorted(f for f in os.listdir(BG_DIR)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    print(f"\n=== Background Images ({len(files)} total) ===")
    print(f"Directory: {BG_DIR}\n")
    for i, f in enumerate(files, 1):
        print(f"  {i:2d}. {f}")


def preview_subtitles():
    """Preview subtitle timestamps for the first episode (uses cached data if available)."""
    titles, scripts = parse_docx(DOCX_PATH)
    ep = min(ep for ep in scripts if ep >= START_FROM)
    text = scripts.get(ep, "")
    sentences = split_sentences(text)
    segments = get_sentence_timing(ep, text, sentences)
    print(f"\nEpisode {ep} subtitle preview ({len(segments)} sentences)\n")
    for start, end, text in segments:
        print(f"  [{start:6.3f}s -> {end:6.3f}s]  {text}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "--list-backgrounds":
            list_backgrounds()
        elif cmd == "--preview-subs":
            preview_subtitles()
        elif cmd == "--single":
            if len(sys.argv) < 3:
                print("Usage: python video_generate.py --single <episode_number>")
                print("Example: python video_generate.py --single 29")
            else:
                main(only_ep=int(sys.argv[2]), force=True)
        elif cmd == "--batch":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            main(max_count=n)
        else:
            print(f"Unknown command: {cmd}")
            print("Available commands:")
            print("  --single <N>         Generate only episode N")
            print("  --batch <N>          Generate up to N episodes, then pause")
            print("  --list-backgrounds   List background images")
            print("  --preview-subs       Preview subtitle timestamps")
    else:
        main()
