import os
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(filepath: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file and remove DC offset.

    Equivalent to MATLAB:
        [AudioData, Fs] = audioread(filepath)
        AudioData = AudioData - mean(AudioData)

    Supports .wav, .flac, .mp3 (via soundfile/libsndfile).
    Returns mono float64 signal and sample rate.
    """
    data, fs = sf.read(filepath, dtype="float64", always_2d=False)

    if data.ndim == 2:
        # Stereo or multi-channel: take first channel
        data = data[:, 0]

    data = data - np.mean(data)   # ADC offset correction
    return data, fs


def get_audio_info(filepath: str) -> dict:
    """
    Return basic metadata without loading the full file.
    Equivalent to MATLAB audioinfo().
    """
    info = sf.info(filepath)
    return {
        "sample_rate": info.samplerate,
        "duration_seconds": info.duration,
        "duration_minutes": info.duration / 60.0,
        "channels": info.channels,
        "format": info.format,
    }


def chunk_audio(
    audio_data: np.ndarray,
    fs: int,
    file_chunk_number: int,
    label_frame_length: float,
) -> tuple[list[np.ndarray], int, int, int]:
    """
    Split audio into N big chunks, each containing an equal number of
    complete label frames.  The last chunk gets all remaining samples
    (including any partial frame).

    Mirrors the MATLAB LoadAudioData nested function exactly.

    Returns
    -------
    chunks               : list of np.ndarray, length = file_chunk_number
    frames_per_chunk     : int  (frames in chunks 0 … N-2)
    frames_in_last_chunk : int  (frames in chunk N-1, may differ)
    frame_size           : int  (samples per label frame)
    """
    frame_size = int(np.floor(label_frame_length * fs))
    total_complete_frames = int(np.floor(len(audio_data) / frame_size))
    frames_per_chunk = int(np.floor(total_complete_frames / file_chunk_number))
    frames_in_last_chunk = (
        total_complete_frames - frames_per_chunk * (file_chunk_number - 1)
    )

    chunks: list[np.ndarray] = []

    for i in range(file_chunk_number):
        if i < file_chunk_number - 1:
            # Chunks 0 … N-2: exact frames_per_chunk complete frames
            start_sample = i * frames_per_chunk * frame_size
            end_sample   = (i + 1) * frames_per_chunk * frame_size
            chunks.append(audio_data[start_sample:end_sample])
        else:
            # Last chunk: everything that remains
            start_sample = i * frames_per_chunk * frame_size
            chunks.append(audio_data[start_sample:])

    _log_chunk_info(chunks, fs, frame_size, frames_per_chunk, frames_in_last_chunk)

    return chunks, frames_per_chunk, frames_in_last_chunk, frame_size


def resample_audio(audio_data: np.ndarray, fs_original: int, fs_target: int) -> np.ndarray:
    """
    Polyphase resample, equivalent to MATLAB resample(signal, target_Fs, Fs).
    Uses the GCD to keep up/down integers as small as possible.
    """
    if fs_original == fs_target:
        return audio_data.copy()

    common = gcd(fs_original, fs_target)
    up     = fs_target   // common
    down   = fs_original // common
    return resample_poly(audio_data, up, down)


def pad_chunk(chunk: np.ndarray, window_size: int, overlap_samples: int) -> np.ndarray:
    """
    Zero-pad a chunk that is shorter than the minimum required for spectrogram
    computation.

    Matches MATLAB pad logic:
        padTarget = minRequiredSamples + (minRequiredSamples - overlap_samples_SA)
    """
    pad_target = window_size + (window_size - overlap_samples)
    if len(chunk) < pad_target:
        padding = np.zeros(pad_target - len(chunk), dtype=chunk.dtype)
        return np.concatenate([chunk, padding])
    return chunk


def check_existing_labels(audio_filepath: str) -> bool:
    """
    Return True if a _labels.csv already exists alongside the audio file.
    Equivalent to MATLAB checkExistingLabels nested function.
    """
    directory = os.path.dirname(audio_filepath)
    stem      = os.path.splitext(os.path.basename(audio_filepath))[0]
    csv_path  = os.path.join(directory, f"{stem}_labels.csv")
    return os.path.isfile(csv_path)


# Internal helpers                                                     

def _log_chunk_info(
    chunks: list[np.ndarray],
    fs: int,
    frame_size: int,
    frames_per_chunk: int,
    frames_in_last_chunk: int,
) -> None:
    """Print the same diagnostic messages as the MATLAB version."""
    for i, ch in enumerate(chunks):
        complete_frames = len(ch) // frame_size
        remaining       = len(ch) - complete_frames * frame_size
        duration        = len(ch) / fs

        if i < len(chunks) - 1:
            print(
                f"Chunk {i + 1}: {frames_per_chunk} complete frames, "
                f"{duration:.1f} s"
            )
        else:
            if remaining > 0:
                print(
                    f"Chunk {i + 1}: {complete_frames} complete frames + "
                    f"partial ({remaining / fs:.2f} s)"
                )
            else:
                print(
                    f"Chunk {i + 1}: {complete_frames} complete frames "
                    f"(perfect fit)"
                )