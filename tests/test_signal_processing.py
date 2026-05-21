import numpy as np
import pytest
from hydroseek.signal_processing import (
    compute_spectrogram,
    compute_mel_spectrogram,
    overlap_percent_to_samples,
    format_frequency_ticks,
    get_frequency_axis_label,
)
# Shared test signal                                                   


@pytest.fixture
def sine_signal():
    """
    A 5-second 1kHz sine wave at 48kHz.
    Long enough for all window/overlap combinations to produce
    multiple time frames.
    """
    fs = 48000
    duration = 5.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 1000 * t).astype(np.float64)
    return signal, fs


@pytest.fixture
def short_signal():
    """A short signal for testing with smaller window sizes."""
    fs = 8000
    duration = 2.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float64)
    return signal, fs

# compute_spectrogram                                                  


def test_spectrogram_output_shapes(sine_signal):
    signal, fs = sine_signal
    window_size = 2048
    overlap = overlap_percent_to_samples(75, window_size)
    S_db, freqs, times = compute_spectrogram(signal, fs, window_size, overlap)

    # S_db must be 2D with shape (n_freqs, n_times)
    assert S_db.ndim == 2
    # Frequency axis length = window_size // 2 + 1 for a real-valued signal
    assert len(freqs) == window_size // 2 + 1
    # Shape must be consistent: rows = freqs, cols = times
    assert S_db.shape == (len(freqs), len(times))


def test_spectrogram_frequency_axis_range(sine_signal):
    signal, fs = sine_signal
    window_size = 2048
    overlap = overlap_percent_to_samples(75, window_size)
    _, freqs, _ = compute_spectrogram(signal, fs, window_size, overlap)

    # Lowest frequency should be 0 Hz
    assert freqs[0] == pytest.approx(0.0)
    # Highest frequency should be the Nyquist frequency
    assert freqs[-1] == pytest.approx(fs / 2.0)


def test_spectrogram_time_axis_range(sine_signal):
    signal, fs = sine_signal
    window_size = 2048
    overlap = overlap_percent_to_samples(75, window_size)
    _, _, times = compute_spectrogram(signal, fs, window_size, overlap)

    # Time axis must start at or near 0
    assert times[0] >= 0.0
    # Time axis must not exceed the signal duration
    duration = len(signal) / fs
    assert times[-1] <= duration


def test_spectrogram_values_are_db(sine_signal):
    """
    A pure sine wave should produce a clear peak at 1kHz in dB.
    All values should be finite (no log(0) = -inf leaking through).
    """
    signal, fs = sine_signal
    window_size = 2048
    overlap = overlap_percent_to_samples(75, window_size)
    S_db, freqs, _ = compute_spectrogram(signal, fs, window_size, overlap)

    assert np.all(np.isfinite(S_db)), "S_db contains non-finite values"

    # The peak energy should be near 1000 Hz
    mean_power_per_freq = S_db.mean(axis=1)
    peak_freq_index = np.argmax(mean_power_per_freq)
    assert abs(freqs[peak_freq_index] - 1000.0) < 100.0


def test_spectrogram_different_window_sizes(sine_signal):
    """Changing the window size should change the frequency resolution."""
    signal, fs = sine_signal
    overlap_small = overlap_percent_to_samples(50, 512)
    overlap_large = overlap_percent_to_samples(50, 2048)

    _, freqs_small, _ = compute_spectrogram(signal, fs, 512, overlap_small)
    _, freqs_large, _ = compute_spectrogram(signal, fs, 2048, overlap_large)

    # Larger window = more frequency bins
    assert len(freqs_large) > len(freqs_small)


# compute_mel_spectrogram                                              


def test_mel_spectrogram_output_shapes(short_signal):
    signal, fs = short_signal
    window_size = 512
    overlap = overlap_percent_to_samples(50, window_size)
    S_mel_db, mel_freqs, times = compute_mel_spectrogram(
        signal, fs, window_size, overlap, n_mels=128
    )

    assert S_mel_db.ndim == 2
    assert S_mel_db.shape[0] == 128       # n_mels rows
    assert len(mel_freqs) == 128
    assert S_mel_db.shape == (len(mel_freqs), len(times))


def test_mel_spectrogram_frequency_axis_monotonic(short_signal):
    """Mel frequencies should always increase from low to high."""
    signal, fs = short_signal
    window_size = 512
    overlap = overlap_percent_to_samples(50, window_size)
    _, mel_freqs, _ = compute_mel_spectrogram(signal, fs, window_size, overlap)

    diffs = np.diff(mel_freqs)
    assert np.all(diffs > 0), "Mel frequency axis is not monotonically increasing"


def test_mel_spectrogram_values_finite(short_signal):
    signal, fs = short_signal
    window_size = 512
    overlap = overlap_percent_to_samples(50, window_size)
    S_mel_db, _, _ = compute_mel_spectrogram(signal, fs, window_size, overlap)

    assert np.all(np.isfinite(S_mel_db))


def test_mel_spectrogram_n_mels_respected(short_signal):
    signal, fs = short_signal
    window_size = 512
    overlap = overlap_percent_to_samples(50, window_size)

    _, mel_freqs_64, _  = compute_mel_spectrogram(signal, fs, window_size, overlap, n_mels=64)
    _, mel_freqs_128, _ = compute_mel_spectrogram(signal, fs, window_size, overlap, n_mels=128)

    assert len(mel_freqs_64) == 64
    assert len(mel_freqs_128) == 128


# overlap_percent_to_samples                                           


def test_overlap_75_percent():
    assert overlap_percent_to_samples(75.0, 2048) == 1536


def test_overlap_50_percent():
    assert overlap_percent_to_samples(50.0, 1024) == 512


def test_overlap_0_percent():
    assert overlap_percent_to_samples(0.0, 2048) == 0


def test_overlap_100_percent():
    # 100% overlap = window_size samples (edge case, not used in practice)
    assert overlap_percent_to_samples(100.0, 2048) == 2048


# format_frequency_ticks                                               


def test_format_ticks_hz_range():
    ticks = np.array([0, 250, 500, 750])
    labels = format_frequency_ticks(ticks)
    assert labels == ["0", "250", "500", "750"]


def test_format_ticks_khz_range():
    ticks = np.array([1000, 2000, 10000, 24000])
    labels = format_frequency_ticks(ticks)
    assert labels == ["1", "2", "10", "24"]


def test_format_ticks_mixed_range():
    ticks = np.array([500, 1000, 5000])
    labels = format_frequency_ticks(ticks)
    assert labels == ["500", "1", "5"]



# get_frequency_axis_label                                             #

def test_label_hz():
    assert get_frequency_axis_label(500.0) == "Frequency (Hz)"


def test_label_khz():
    assert get_frequency_axis_label(48000.0) == "Frequency (kHz)"


def test_label_boundary():
    assert get_frequency_axis_label(1000.0) == "Frequency (kHz)"