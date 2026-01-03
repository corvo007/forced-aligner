"""
CTC Forced Aligner Demo - SRT Alignment Tool
=============================================
This demo aligns SRT subtitles with audio using the ctc-forced-aligner library.

Usage:
    # SRT Mode (default)
    python align.py --audio "audio.mp3" --srt "subtitles.srt" --language "eng"
    python align.py --audio "audio.mp3" --srt "subtitles.srt" --language "cmn" --romanize
    python align.py --audio "audio.mp3" --srt "subtitles.srt" --language "eng" --debug
    
    # JSON Mode
    python align.py --audio "audio.mp3" --json-input "input.json" --json-output "output.json" --language "eng"
    python align.py --audio "audio.mp3" --json-input - --json-output - --language "eng"  # stdin/stdout

Features:
    - Automatically converts audio to WAV format (requires ffmpeg)
    - Reads subtitles from SRT file or JSON input
    - Outputs aligned SRT file or JSON with corrected timestamps
    - Supports stdin/stdout for JSON mode (use "-" as file path)
    - Saves intermediate results for debugging (--debug flag)
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime

import torch

# Explicitly import transformers model classes to ensure they are packaged and loaded
try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Config, Wav2Vec2Processor
    import transformers.models.wav2vec2.modeling_wav2vec2
except ImportError:
    pass

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    # load_audio,  # Skip this - use custom loader to avoid torchcodec issues
    postprocess_results,
    preprocess_text,
)


def load_audio_wav(audio_path: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Load audio file using scipy (avoids torchcodec issues on Windows).
    
    The audio file should already be converted to 16kHz mono WAV format.
    """
    from scipy.io import wavfile
    import numpy as np
    
    sample_rate, audio_data = wavfile.read(audio_path)
    
    # Convert to float32 normalized to [-1, 1]
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    elif audio_data.dtype == np.float32:
        pass  # Already float32
    else:
        audio_data = audio_data.astype(np.float32)
    
    # Handle stereo by taking mean
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Resample if needed (should already be 16kHz from ffmpeg conversion)
    if sample_rate != 16000:
        logger.warning(f"Audio sample rate is {sample_rate}Hz, expected 16000Hz")
    
    # Convert to tensor
    waveform = torch.from_numpy(audio_data).to(dtype=dtype, device=device)
    
    return waveform


# ============================================================
# Logging Setup
# ============================================================

def setup_logging(debug: bool = False, log_file: str = None) -> logging.Logger:
    """Setup logging with console and optional file output."""
    logger = logging.getLogger("ctc_aligner")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if debug mode)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# Global logger (will be initialized in main)
logger: logging.Logger = None


# ============================================================
# Data Classes
# ============================================================

@dataclass
class SrtSegment:
    """Represents a single SRT subtitle segment."""
    index: int
    start: float
    end: float
    text: str


@dataclass
class AlignmentResult:
    """Stores all intermediate and final results."""
    audio_path: str
    srt_path: str
    language: str
    romanize: bool
    
    # Intermediate results
    original_segments: list = None
    full_text: str = None
    tokens_starred: list = None
    text_starred: list = None
    word_timestamps: list = None
    aligned_segments: list = None
    
    # Metadata
    audio_duration: float = None
    num_segments: int = None
    num_words: int = None
    processing_time: float = None


# ============================================================
# SRT Parsing and Writing
# ============================================================

def parse_srt_time(time_str: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    match = re.match(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})", time_str.strip())
    if not match:
        raise ValueError(f"Invalid SRT time format: {time_str}")
    hours, minutes, seconds, millis = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_srt(srt_path: str) -> list[SrtSegment]:
    """Parse an SRT file and return a list of SrtSegment objects."""
    logger.info(f"Parsing SRT file: {srt_path}")
    
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    logger.debug(f"SRT file size: {len(content)} bytes")
    
    # Split by blank lines (handles different line endings)
    blocks = re.split(r"\n\s*\n", content.strip())
    segments = []
    
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            logger.debug(f"Skipping invalid block (less than 3 lines): {block[:50]}...")
            continue
        
        try:
            index = int(lines[0].strip())
            time_match = re.match(
                r"(.+?)\s*-->\s*(.+)", 
                lines[1].strip()
            )
            if not time_match:
                logger.warning(f"Invalid time format in segment {index}: {lines[1]}")
                continue
            
            start = parse_srt_time(time_match.group(1))
            end = parse_srt_time(time_match.group(2))
            text = "\n".join(lines[2:]).strip()
            
            segments.append(SrtSegment(
                index=index,
                start=start,
                end=end,
                text=text
            ))
            
            logger.debug(
                f"Parsed segment {index}: [{format_srt_time(start)} --> {format_srt_time(end)}] "
                f"{text[:30]}{'...' if len(text) > 30 else ''}"
            )
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing segment: {e}")
            continue
    
    logger.info(f"Successfully parsed {len(segments)} segments")
    return segments


def write_srt(segments: list[SrtSegment], output_path: str) -> None:
    """Write SRT segments to a file."""
    logger.info(f"Writing SRT to: {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(seg.start)} --> {format_srt_time(seg.end)}\n")
            f.write(f"{seg.text}\n\n")
    
    logger.info(f"Wrote {len(segments)} segments to {output_path}")


# ============================================================
# JSON Input/Output
# ============================================================

def parse_json_input(json_input: str) -> list[SrtSegment]:
    """
    Parse JSON input and return a list of SrtSegment objects.
    
    Args:
        json_input: Path to JSON file or '-' for stdin
        
    Expected JSON format:
    {
        "segments": [
            {"index": 1, "start": 0.0, "end": 1.5, "text": "Hello world"},
            {"index": 2, "start": 1.5, "end": 3.0, "text": "How are you"}
        ]
    }
    
    Returns:
        List of SrtSegment objects
    """
    if json_input == "-":
        logger.info("Reading JSON from stdin...")
        content = sys.stdin.read()
    else:
        logger.info(f"Parsing JSON file: {json_input}")
        with open(json_input, "r", encoding="utf-8") as f:
            content = f.read()
    
    logger.debug(f"JSON content size: {len(content)} bytes")
    
    data = json.loads(content)
    
    # Support both {"segments": [...]} and direct array [...]
    if isinstance(data, list):
        segments_data = data
    elif isinstance(data, dict) and "segments" in data:
        segments_data = data["segments"]
    else:
        raise ValueError(
            "Invalid JSON format. Expected {\"segments\": [...]} or [...] array"
        )
    
    segments = []
    for i, seg_data in enumerate(segments_data):
        # Allow index to be optional (auto-generate if missing)
        index = seg_data.get("index", i + 1)
        
        # Validate required fields
        if "text" not in seg_data:
            raise ValueError(f"Segment {index} missing required 'text' field")
        
        # Start/end can be optional (will use 0.0 as placeholder)
        start = seg_data.get("start", 0.0)
        end = seg_data.get("end", 0.0)
        text = seg_data["text"]
        
        segments.append(SrtSegment(
            index=index,
            start=start,
            end=end,
            text=text
        ))
        
        logger.debug(
            f"Parsed segment {index}: [{format_srt_time(start)} --> {format_srt_time(end)}] "
            f"{text[:30]}{'...' if len(text) > 30 else ''}"
        )
    
    logger.info(f"Successfully parsed {len(segments)} segments from JSON")
    return segments


def write_json_output(
    segments: list[SrtSegment], 
    json_output: str,
    include_metadata: bool = True,
    processing_time: float = None
) -> None:
    """
    Write aligned segments to JSON output.
    
    Args:
        segments: List of aligned SrtSegment objects
        json_output: Path to output JSON file or '-' for stdout
        include_metadata: Whether to include metadata in output
        processing_time: Processing time in seconds (optional)
    """
    output_data = {
        "segments": [asdict(seg) for seg in segments]
    }
    
    if include_metadata:
        output_data["metadata"] = {
            "count": len(segments),
            "processing_time": processing_time
        }
    
    json_str = json.dumps(output_data, ensure_ascii=False, indent=2)
    
    if json_output == "-":
        logger.info("Writing JSON to stdout...")
        print(json_str)
    else:
        logger.info(f"Writing JSON to: {json_output}")
        with open(json_output, "w", encoding="utf-8") as f:
            f.write(json_str)
        logger.info(f"Wrote {len(segments)} segments to {json_output}")


# ============================================================
# Audio Conversion
# ============================================================

def convert_to_wav(
    audio_path: str, 
    output_path: str = None, 
    ffmpeg_path: str = "ffmpeg"
) -> str:
    """
    Convert audio file to WAV format using ffmpeg.
    
    Args:
        audio_path: Path to the input audio file
        output_path: Path for the output WAV file (optional)
        ffmpeg_path: Exact path to ffmpeg executable
        
    Returns:
        Path to the WAV file
    """
    if output_path is None:
        base = os.path.splitext(audio_path)[0]
        output_path = f"{base}_converted.wav"
    
    logger.info(f"Converting audio: {audio_path} -> {output_path}")
    logger.debug(f"Using FFmpeg: {ffmpeg_path}")
    
    cmd = [
        ffmpeg_path,
        "-i", audio_path,
        "-ar", "16000",      # 16kHz sample rate
        "-ac", "1",          # Mono
        "-c:a", "pcm_s16le", # 16-bit PCM
        "-y",                # Overwrite output
        output_path
    ]
    
    logger.debug(f"FFmpeg command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True,
            text=True
        )
        logger.debug(f"FFmpeg stderr: {result.stderr[:500] if result.stderr else 'None'}")
        
        # Get file size
        file_size = os.path.getsize(output_path)
        logger.info(f"Audio converted successfully: {output_path} ({file_size / 1024 / 1024:.2f} MB)")
        
        return output_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr}")
        raise RuntimeError(f"ffmpeg conversion failed: {e.stderr}")
    except FileNotFoundError:
        logger.error("ffmpeg not found in PATH")
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg and add it to your PATH."
        )


# ============================================================
# Alignment Functions
# ============================================================

def align_srt_with_audio(
    audio_path: str,
    srt_segments: list[SrtSegment],
    language: str = "eng",
    romanize: bool = True,
    batch_size: int = 4,
    model_path: str = None,
    result: AlignmentResult = None,
) -> list[SrtSegment]:
    """
    Align SRT segments with audio and return segments with corrected timestamps.
    
    Args:
        audio_path: Path to the WAV audio file
        srt_segments: List of SRT segments to align
        language: ISO 639-3 language code
        romanize: Whether to romanize non-latin scripts
        batch_size: Batch size for inference
        model_path: Path to alignment model (HuggingFace name or local path)
        result: AlignmentResult object to store intermediate results
        
    Returns:
        List of SrtSegment with aligned timestamps
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    logger.info(f"Device: {device}, dtype: {dtype}")
    logger.info(f"Language: {language}, Romanize: {romanize}, Batch size: {batch_size}")
    
    # Load model and tokenizer
    model_name = model_path or "MahmoudAshraf/mms-300m-1130-forced-aligner"
    logger.info(f"Loading alignment model: {model_name}")
    model, tokenizer = load_alignment_model(device, model_name, dtype=dtype)
    logger.debug(f"Model loaded: {type(model).__name__}")
    
    # Load audio
    logger.info(f"Loading audio: {audio_path}")
    audio_waveform = load_audio_wav(audio_path, model.dtype, model.device)
    logger.debug(f"Audio waveform shape: {audio_waveform.shape}")
    logger.debug(f"Audio duration: {audio_waveform.shape[-1] / 16000:.2f} seconds")
    
    if result:
        result.audio_duration = audio_waveform.shape[-1] / 16000
    
    # Generate emissions
    logger.info("Generating emissions...")
    emissions, stride = generate_emissions(
        model, 
        audio_waveform, 
        batch_size=batch_size
    )
    logger.debug(f"Emissions shape: {emissions.shape}")
    logger.debug(f"Stride: {stride}")
    
    # Combine all text for alignment
    full_text = " ".join(seg.text.replace("\n", " ") for seg in srt_segments)
    logger.info(f"Full text length: {len(full_text)} characters")
    logger.debug(f"Full text preview: {full_text[:200]}...")
    
    if result:
        result.full_text = full_text
    
    # Preprocess text
    logger.info("Preprocessing text...")
    tokens_starred, text_starred = preprocess_text(
        full_text,
        romanize=romanize,
        language=language,
        split_size="word",
    )
    
    logger.debug(f"Tokens starred count: {len(tokens_starred)}")
    logger.debug(f"Text starred count: {len(text_starred)}")
    logger.debug(f"Text starred preview: {text_starred[:20]}...")
    
    if result:
        result.tokens_starred = tokens_starred
        result.text_starred = text_starred
        result.num_words = len(text_starred)
    
    # Get alignments
    logger.info("Getting alignments...")
    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        tokenizer,
    )
    logger.debug(f"Segments count: {len(segments)}")
    logger.debug(f"Blank token: {blank_token}")
    
    # Get spans
    logger.info("Getting spans...")
    spans = get_spans(tokens_starred, segments, blank_token)
    logger.debug(f"Spans count: {len(spans)}")
    
    # Postprocess results
    logger.info("Postprocessing results...")
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    logger.info(f"Generated {len(word_timestamps)} word timestamps")
    
    # Log sample word timestamps
    for i, wt in enumerate(word_timestamps[:5]):
        logger.debug(
            f"  Word {i}: [{wt['start']:.3f}s - {wt['end']:.3f}s] '{wt['text']}'"
        )
    if len(word_timestamps) > 5:
        logger.debug(f"  ... and {len(word_timestamps) - 5} more words")
    
    if result:
        result.word_timestamps = word_timestamps
    
    # Map word timestamps back to SRT segments
    logger.info("Mapping timestamps to SRT segments...")
    aligned_segments = map_timestamps_to_srt(srt_segments, word_timestamps)
    
    if result:
        result.aligned_segments = [asdict(s) for s in aligned_segments]
    
    return aligned_segments


def map_timestamps_to_srt(
    srt_segments: list[SrtSegment],
    word_timestamps: list[dict],
) -> list[SrtSegment]:
    """
    Map character-level timestamps back to SRT segment boundaries.
    
    The CTC aligner generates timestamps for each character (for CJK languages)
    or word (for space-separated languages). This function finds the start and
    end timestamps for each SRT segment based on the aligned characters/words.
    """
    aligned = []
    char_idx = 0
    
    # Build a mapping of text position to timestamp index
    # First, reconstruct the full text and track boundaries
    logger.info(f"Mapping {len(word_timestamps)} tokens to {len(srt_segments)} segments")
    
    # Calculate character counts for each segment (excluding spaces used as separators)
    for seg in srt_segments:
        # Get the segment text and normalize it (same as preprocessing)
        seg_text = seg.text.replace("\n", " ").strip()
        
        # Count actual characters in this segment
        # We need to match how preprocess_text splits the text
        num_chars = len(seg_text)
        
        # Account for the space separator added between segments  
        # (except for the first segment)
        space_offset = 1 if char_idx > 0 else 0
        
        logger.debug(
            f"Segment {seg.index}: {num_chars} chars, "
            f"original: [{format_srt_time(seg.start)} --> {format_srt_time(seg.end)}]"
        )
        
        if num_chars == 0 or char_idx >= len(word_timestamps):
            # Keep original timestamps if no chars or exhausted
            logger.warning(
                f"Segment {seg.index}: Keeping original timestamps "
                f"(no chars or char_idx exhausted)"
            )
            aligned.append(SrtSegment(
                index=seg.index,
                start=seg.start,
                end=seg.end,
                text=seg.text
            ))
            continue
        
        # Skip the space separator between segments
        if space_offset > 0 and char_idx < len(word_timestamps):
            if word_timestamps[char_idx]["text"].strip() == "":
                char_idx += 1
        
        # Find start time from first character of this segment
        if char_idx >= len(word_timestamps):
            logger.warning(f"Segment {seg.index}: No more timestamps available")
            aligned.append(SrtSegment(
                index=seg.index,
                start=seg.start,
                end=seg.end,
                text=seg.text
            ))
            continue
            
        start_time = word_timestamps[char_idx]["start"]
        
        # Find end time from last character of this segment
        end_idx = min(char_idx + num_chars - 1, len(word_timestamps) - 1)
        end_time = word_timestamps[end_idx]["end"]
        
        logger.debug(
            f"  -> Aligned: [{format_srt_time(start_time)} --> {format_srt_time(end_time)}] "
            f"(chars {char_idx}-{end_idx})"
        )
        
        aligned.append(SrtSegment(
            index=seg.index,
            start=start_time,
            end=end_time,
            text=seg.text
        ))
        
        char_idx = end_idx + 1
    
    logger.info(f"Alignment complete: {len(aligned)} segments aligned")
    
    return aligned


# ============================================================
# Debug Output Functions
# ============================================================

def save_intermediate_results(
    result: AlignmentResult,
    output_dir: str,
) -> None:
    """Save all intermediate results to files for debugging."""
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving intermediate results to: {output_dir}")
    
    # 1. Save original segments
    if result.original_segments:
        path = os.path.join(output_dir, "01_original_segments.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(s) for s in result.original_segments],
                f, ensure_ascii=False, indent=2
            )
        logger.debug(f"Saved: {path}")
    
    # 2. Save full text
    if result.full_text:
        path = os.path.join(output_dir, "02_full_text.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(result.full_text)
        logger.debug(f"Saved: {path}")
    
    # 3. Save tokens starred (as string representation)
    if result.tokens_starred:
        path = os.path.join(output_dir, "03_tokens_starred.json")
        with open(path, "w", encoding="utf-8") as f:
            # Convert tensor to list if needed
            tokens = result.tokens_starred
            if hasattr(tokens, 'tolist'):
                tokens = tokens.tolist()
            json.dump(tokens, f, ensure_ascii=False, indent=2)
        logger.debug(f"Saved: {path}")
    
    # 4. Save text starred
    if result.text_starred:
        path = os.path.join(output_dir, "04_text_starred.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.text_starred, f, ensure_ascii=False, indent=2)
        logger.debug(f"Saved: {path}")
    
    # 5. Save word timestamps
    if result.word_timestamps:
        path = os.path.join(output_dir, "05_word_timestamps.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.word_timestamps, f, ensure_ascii=False, indent=2)
        logger.debug(f"Saved: {path}")
    
    # 6. Save aligned segments
    if result.aligned_segments:
        path = os.path.join(output_dir, "06_aligned_segments.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.aligned_segments, f, ensure_ascii=False, indent=2)
        logger.debug(f"Saved: {path}")
    
    # 7. Save summary
    summary = {
        "audio_path": result.audio_path,
        "srt_path": result.srt_path,
        "language": result.language,
        "romanize": result.romanize,
        "audio_duration": result.audio_duration,
        "num_segments": result.num_segments,
        "num_words": result.num_words,
        "processing_time": result.processing_time,
    }
    path = os.path.join(output_dir, "00_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.debug(f"Saved: {path}")
    
    logger.info(f"All intermediate results saved to: {output_dir}")


# ============================================================
# Helper Functions
# ============================================================

def get_ffmpeg_path() -> str:
    """
    Locate ffmpeg.exe in likely bundled locations.
    
    Priority:
    1. _MEIPASS/ffmpeg.exe (PyInstaller one-file temp dir)
    2. ./ffmpeg.exe (Same directory as script/exe)
    3. ./bin/ffmpeg.exe (Subdirectory)
    4. System PATH
    """
    # Base directory depends on if we are frozen (exe) or script
    if getattr(sys, 'frozen', False):
        base_dir = parse_base_dir_frozen()
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        os.path.join(base_dir, "ffmpeg.exe"),
        os.path.join(base_dir, "bin", "ffmpeg.exe"),
    ]
    
    # If running as PyInstaller one-file, also check _MEIPASS
    if hasattr(sys, '_MEIPASS'):
        candidates.insert(0, os.path.join(sys._MEIPASS, "ffmpeg.exe"))

    for path in candidates:
        if os.path.exists(path):
            logger.info(f"Found bundled ffmpeg: {path}")
            return path
            
    # Fallback to system PATH
    import shutil
    if shutil.which("ffmpeg"):
        logger.warning("Bundled ffmpeg not found, using system ffmpeg.")
        return "ffmpeg"
        
    raise FileNotFoundError(
        "ffmpeg.exe not found. Please place it in the same directory as this executable or in a 'bin' subdirectory."
    )

def parse_base_dir_frozen():
    """Get the base directory when running as a frozen executable."""
    return os.path.dirname(sys.executable)


# ============================================================
# Main Function
# ============================================================

def main():
    """Main entry point for the SRT alignment tool."""
    global logger
    
    parser = argparse.ArgumentParser(
        description="Align SRT subtitles with audio using CTC forced alignment"
    )
    parser.add_argument(
        "--audio", "-a",
        required=True,
        help="Path to the audio file (any format supported by ffmpeg)"
    )
    parser.add_argument(
        "--srt", "-s",
        default=None,
        help="Path to the SRT subtitle file (required unless using --json-input)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output SRT file path (default: input_aligned.srt)"
    )
    parser.add_argument(
        "--language", "-l",
        default="eng",
        help="Language code (ISO 639-3), e.g., eng, cmn, jpn (default: eng)"
    )
    parser.add_argument(
        "--romanize", "-r",
        action="store_true",
        help="Romanize non-latin scripts (required for default model)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)"
    )
    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help="Keep the converted WAV file after alignment"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode: verbose logging and save intermediate results"
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Directory to save debug files (default: <srt_name>_debug/)"
    )
    parser.add_argument(
        "--model", "-m",
        required=True, 
        help="[REQUIRED] Path to local alignment model directory (containing config.json, model.safetensors, etc.)"
    )
    parser.add_argument(
        "--json-input", "-ji",
        default=None,
        help="Path to JSON input file (use '-' for stdin). Format: {\"segments\": [{\"index\": 1, \"start\": 0.0, \"end\": 1.0, \"text\": \"...\"}]}"
    )
    parser.add_argument(
        "--json-output", "-jo",
        default=None,
        help="Path to JSON output file (use '-' for stdout). Outputs aligned segments as JSON array."
    )
    
    args = parser.parse_args()
    
    # Determine input mode: JSON or SRT
    json_mode = args.json_input is not None
    
    # Validate: must have either --srt or --json-input
    if not json_mode and args.srt is None:
        parser.error("Either --srt or --json-input is required")
    
    # Setup logging
    log_file = None
    if args.debug:
        if json_mode:
            # For JSON mode, use audio file as base for debug dir
            base = os.path.splitext(args.audio)[0]
        else:
            base = os.path.splitext(args.srt)[0]
        debug_dir = args.debug_dir or f"{base}_debug"
        os.makedirs(debug_dir, exist_ok=True)
        log_file = os.path.join(debug_dir, "alignment.log")
    
    logger = setup_logging(debug=args.debug, log_file=log_file)
    
    # Validate inputs
    if not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        return 1
    
    # Validate SRT file (only if not in JSON mode)
    if not json_mode and not os.path.exists(args.srt):
        logger.error(f"SRT file not found: {args.srt}")
        return 1
    
    # Validate JSON input file (only if in JSON mode and not stdin)
    if json_mode and args.json_input != "-" and not os.path.exists(args.json_input):
        logger.error(f"JSON input file not found: {args.json_input}")
        return 1
        
    # Strict Model Validation
    if not os.path.isdir(args.model):
        logger.error(f"Model path must be a directory: {args.model}")
        return 1
    if not os.path.exists(os.path.join(args.model, "config.json")):
        logger.error(f"Invalid model directory (config.json not found): {args.model}")
        return 1

    # Locate ffmpeg
    try:
        ffmpeg_path = get_ffmpeg_path()
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    
    # Set output path (only for SRT mode)
    if not json_mode and args.output is None:
        base = os.path.splitext(args.srt)[0]
        args.output = f"{base}_aligned.srt"
    
    # Determine input source for display
    input_source = args.json_input if json_mode else args.srt
    output_target = args.json_output if json_mode else args.output
    mode_label = "JSON" if json_mode else "SRT"
    
    # Print header
    logger.info("=" * 60)
    logger.info(f"CTC Forced Aligner - {mode_label} Alignment Tool (CLI)")
    logger.info("=" * 60)
    logger.info(f"Audio:     {args.audio}")
    logger.info(f"Input:     {input_source} ({mode_label})")
    logger.info(f"Output:    {output_target}")
    logger.info(f"Language:  {args.language}")
    logger.info(f"Model:     {args.model}")
    logger.info(f"FFmpeg:    {ffmpeg_path}")
    logger.info("=" * 60)
    
    # Initialize result tracker for debug mode
    result = AlignmentResult(
        audio_path=args.audio,
        srt_path=args.srt if not json_mode else args.json_input,
        language=args.language,
        romanize=args.romanize,
    ) if args.debug else None
    
    start_time = datetime.now()
    
    # Convert audio to WAV
    wav_path = convert_to_wav(args.audio, ffmpeg_path=ffmpeg_path)
    
    try:
        # Parse input (JSON or SRT)
        if json_mode:
            srt_segments = parse_json_input(args.json_input)
        else:
            srt_segments = parse_srt(args.srt)
        
        if result:
            result.original_segments = srt_segments
            result.num_segments = len(srt_segments)
        
        # Perform alignment
        aligned_segments = align_srt_with_audio(
            audio_path=wav_path,
            srt_segments=srt_segments,
            language=args.language,
            romanize=args.romanize,
            batch_size=args.batch_size,
            model_path=args.model,
            result=result,
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        if result:
            result.processing_time = processing_time
        
        # Write output (JSON or SRT)
        if json_mode:
            write_json_output(
                aligned_segments, 
                args.json_output or "-",  # Default to stdout if not specified
                processing_time=processing_time
            )
            output_msg = args.json_output if args.json_output and args.json_output != "-" else "stdout"
        else:
            write_srt(aligned_segments, args.output)
            output_msg = args.output
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Aligned {mode_label} saved to: {output_msg}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info("=" * 60)
        
        # Show comparison of original vs aligned (skip if outputting to stdout for cleaner output)
        if not (json_mode and (args.json_output is None or args.json_output == "-")):
            logger.info("")
            logger.info("Alignment comparison (first 5 segments):")
            logger.info("-" * 60)
            for orig, aligned in zip(srt_segments[:5], aligned_segments[:5]):
                logger.info(f"Segment {orig.index}:")
                logger.info(
                    f"  Original: [{format_srt_time(orig.start)} --> {format_srt_time(orig.end)}]"
                )
                logger.info(
                    f"  Aligned:  [{format_srt_time(aligned.start)} --> {format_srt_time(aligned.end)}]"
                )
                delta_start = aligned.start - orig.start
                delta_end = aligned.end - orig.end
                logger.info(f"  Delta:    start {delta_start:+.3f}s, end {delta_end:+.3f}s")
            
            if len(aligned_segments) > 5:
                logger.info(f"  ... and {len(aligned_segments) - 5} more segments")
        
        # Save debug files
        if args.debug and result:
            if json_mode:
                debug_dir = args.debug_dir or f"{os.path.splitext(args.audio)[0]}_debug"
            else:
                debug_dir = args.debug_dir or f"{os.path.splitext(args.srt)[0]}_debug"
            save_intermediate_results(result, debug_dir)
        
    finally:
        # Cleanup converted WAV if not keeping
        if not args.keep_wav and wav_path != args.audio and os.path.exists(wav_path):
            os.remove(wav_path)
            logger.info("Cleaned up temporary WAV file")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

