"""
TTS Generator - Generate speech audio from script documents using ElevenLabs API.

Reads a structured Word document (.docx) with tagged sections,
extracts text content, and generates TTS audio files via ElevenLabs.

Usage:
    python tts_generate.py                # Generate all episodes
    python tts_generate.py --preview      # Preview extracted text (no API calls)
"""

import os
import re
import sys
import time
import requests
from docx import Document
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==============================
# Configuration
# ==============================
API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")

# Script document path (Word .docx file)
DOCX_PATH = os.getenv("DOCX_PATH", "scripts.docx")
# Output directory for generated audio
OUTPUT_DIR = os.getenv("AUDIO_OUTPUT_DIR", "audio_output")
# Start from this episode number
START_FROM = int(os.getenv("START_FROM", "1"))
# Tag used to mark the text section in the docx (e.g. "【正文日语】" for Japanese)
TEXT_TAG = os.getenv("TEXT_TAG", "【Text】")
# Optional end tags to stop extraction (comma-separated)
END_TAGS = os.getenv("END_TAGS", "").split(",") if os.getenv("END_TAGS") else []
# ==============================


def extract_text_content(text, text_tag, end_tags):
    """Extract content after the text tag, stopping at any end tag."""
    pattern = text_tag + r'\s*(.*?)'
    if end_tags:
        end_pattern = '|'.join(re.escape(t) for t in end_tags if t.strip())
        pattern += f'(?={end_pattern}|$)'
    else:
        pattern += '$'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        content = match.group(1).strip().replace('\u200b', '')
        return content
    return None


def extract_scripts(docx_path, text_tag, end_tags):
    """Extract episode scripts from a Word document.

    Expected format per episode:
        <number> <title text>
        <text_tag> content here...

    Returns dict: {episode_number: text_content}
    """
    doc = Document(docx_path)
    scripts = {}
    current_ep = None

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Detect episode header (starts with a number)
        ep_match = re.match(r'^(\d+)[「『\s]', text)
        if ep_match:
            current_ep = int(ep_match.group(1))
            continue

        # Detect content paragraph with text tag
        if text_tag in text and current_ep is not None:
            content = extract_text_content(text, text_tag, end_tags)
            if content:
                scripts[current_ep] = content

    return scripts


def text_to_speech(text):
    """Call ElevenLabs API to generate speech audio. Returns audio bytes."""
    if not API_KEY:
        raise ValueError("ELEVENLABS_API_KEY is not set. Please configure it in .env file.")
    if not VOICE_ID:
        raise ValueError("ELEVENLABS_VOICE_ID is not set. Please configure it in .env file.")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": os.getenv("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5"),
        "voice_settings": {
            "stability": float(os.getenv("TTS_STABILITY", "0.5")),
            "similarity_boost": float(os.getenv("TTS_SIMILARITY_BOOST", "0.75"))
        }
    }
    resp = requests.post(url, json=payload, headers=headers)
    if resp.status_code == 200:
        return resp.content
    else:
        raise Exception(f"API Error {resp.status_code}: {resp.text}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Parsing script document...")
    scripts = extract_scripts(DOCX_PATH, TEXT_TAG, END_TAGS)
    print(f"Found {len(scripts)} episodes, starting from episode {START_FROM}\n")

    target_eps = sorted(ep for ep in scripts if ep >= START_FROM)

    for ep_num in target_eps:
        output_path = os.path.join(OUTPUT_DIR, f"{ep_num}.mp3")

        if os.path.exists(output_path):
            print(f"  Episode {ep_num}: already exists, skipping")
            continue

        text = scripts[ep_num]
        print(f"  Episode {ep_num} ({len(text)} chars)...", end=" ", flush=True)

        try:
            audio = text_to_speech(text)
            with open(output_path, 'wb') as f:
                f.write(audio)
            print(f"Done -> {output_path}")
            time.sleep(0.5)  # Rate limit protection
        except Exception as e:
            print(f"Failed: {e}")

    print("\nAll done!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--preview':
        scripts = extract_scripts(DOCX_PATH, TEXT_TAG, END_TAGS)
        for ep in sorted(scripts.keys()):
            print(f"\n=== Episode {ep} ({len(scripts[ep])} chars) ===")
            print(scripts[ep][:150] + "...")
    else:
        main()
