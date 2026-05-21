import numpy as np
import pytest
import tempfile, os
import soundfile as sf

from hydroseek.audio import (
    load_audio,
    get_audio_info,
    chunk_audio,
    resample_audio,
    pad_chunk,
    check_existing_labels,
)



# Fixtures                                                             #

@pytest.fixture
def sine_wav(tmp_path):
    """Write a 10-second 1kHz sine wave at 48000 Hz, return path."""
    fs       = 48000
    duration = 10.0
    t        = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal   = 0.5 * np.sin(2 * np.pi * 1000 * t).astype(np.float64)
    path     = tmp_path / "test_sine.wav"
    sf.write(str(path), signal, fs)
    return str(path), fs, duration


@pytest.fixture
def stereo_wav(tmp_path):
    """Write a 5-second stereo file, return path."""
    fs       = 44100
    duration = 5.0
    n        = int(fs * duration)
    signal   = np.random.randn(n, 2).astype(np.float64) * 0.1
    path     = tmp_path / "stereo.wav"
    sf.write(str(path), signal, fs)
    return str(path), fs



# load_audio                                                           #


def test_load_audio_shape_and_dtype(sine_wav):
    path, fs, duration = sine_wav
    data, loaded_fs = load_audio(path)
    assert loaded_fs == fs
    assert data.ndim == 1
    assert data.dtype == np.float64
    assert len(data) == pytest.approx(fs * duration, abs=1)


def test_load_audio_dc_removed(sine_wav):
    path, fs, _ = sine_wav
    data, _ = load_audio(path)
    assert abs(np.mean(data)) < 1e-10


def test_load_audio_stereo_becomes_mono(stereo_wav):
    path, fs = stereo_wav
    data, loaded_fs = load_audio(path)
    assert data.ndim == 1
    assert loaded_fs == fs



# chunk_audio                                                          #


def test_chunk_count(sine_wav):
    path, fs, _ = sine_wav
    data, loaded_fs = load_audio(path)
    chunks, _, _, _ = chunk_audio(data, loaded_fs, file_chunk_number=4, label_frame_length=2.0)
    assert len(chunks) == 4


def test_chunks_cover_full_signal(sine_wav):
    path, fs, _ = sine_wav
    data, loaded_fs = load_audio(path)
    chunks, _, _, _ = chunk_audio(data, loaded_fs, file_chunk_number=3, label_frame_length=2.0)
    total_samples = sum(len(c) for c in chunks)
    assert total_samples == len(data)


def test_frames_per_chunk_consistency(sine_wav):
    path, fs, _ = sine_wav
    data, loaded_fs = load_audio(path)
    chunks, frames_per_chunk, frames_last, frame_size = chunk_audio(
        data, loaded_fs, file_chunk_number=4, label_frame_length=2.0
    )
    for ch in chunks[:-1]:
        assert len(ch) == frames_per_chunk * frame_size


def test_single_chunk(sine_wav):
    path, fs, _ = sine_wav
    data, loaded_fs = load_audio(path)
    chunks, frames_per_chunk, frames_last, frame_size = chunk_audio(
        data, loaded_fs, file_chunk_number=1, label_frame_length=2.0
    )
    assert len(chunks) == 1
    assert len(chunks[0]) == len(data)


# resample_audio                                                       #


def test_resample_length(sine_wav):
    path, fs, duration = sine_wav
    data, loaded_fs = load_audio(path)
    target_fs = 5000
    resampled = resample_audio(data, loaded_fs, target_fs)
    expected  = int(np.round(len(data) * target_fs / loaded_fs))
    # Allow ±2 samples for polyphase filter edge effects
    assert abs(len(resampled) - expected) <= 2


def test_resample_same_fs_returns_copy(sine_wav):
    path, fs, _ = sine_wav
    data, loaded_fs = load_audio(path)
    resampled = resample_audio(data, loaded_fs, loaded_fs)
    np.testing.assert_array_equal(data, resampled)


# pad_chunk                                                            #


def test_pad_chunk_short_signal():
    chunk    = np.ones(100)
    result   = pad_chunk(chunk, window_size=512, overlap_samples=256)
    expected = 512 + (512 - 256)   # 768
    assert len(result) == expected
    assert np.all(result[100:] == 0.0)


def test_pad_chunk_already_long_enough():
    chunk  = np.ones(1000)
    result = pad_chunk(chunk, window_size=256, overlap_samples=128)
    assert len(result) == 1000     # unchanged



# check_existing_labels                                                #


def test_check_existing_labels_present(tmp_path):
    wav_path = tmp_path / "file.wav"
    csv_path = tmp_path / "file_labels.csv"
    wav_path.touch()
    csv_path.touch()
    assert check_existing_labels(str(wav_path)) is True


def test_check_existing_labels_absent(tmp_path):
    wav_path = tmp_path / "file.wav"
    wav_path.touch()
    assert check_existing_labels(str(wav_path)) is False