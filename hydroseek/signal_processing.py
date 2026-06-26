import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.signal.windows import hann
import librosa


def compute_spectrogram(
    signal: np.ndarray,
    fs: int,
    window_size: int,
    overlap_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a short-time Fourier transform spectrogram and convert to dB.

    Equivalent to MATLAB:
        [S, frq, tm] = spectrogram(chunk, hann(windowsize), overlap_samples, [], fs, 'yaxis')
        S = 20 * log10(abs(S))

    The nfft argument is left equal to window_size 

    Parameters
    signal          : 1-D float64 array
    fs              : sample rate in Hz
    window_size     : FFT window length in samples
    overlap_samples : number of samples of overlap between windows

    Returns
    S_db    : (n_freqs, n_times) array in dB
    freqs   : frequency axis in Hz
    times   : time axis in seconds
    """
    win = hann(window_size)

    freqs, times, Sxx = scipy_spectrogram(
        signal,
        fs=fs,
        window=win,
        nperseg=window_size,
        noverlap=overlap_samples,
        nfft=window_size,
        scaling="spectrum",
    )

    # Add a small epsilon before log to avoid log(0) = -inf
    S_db = 20.0 * np.log10(np.abs(Sxx) + 1e-12)
    return S_db, freqs, times


def compute_mel_spectrogram(
    signal: np.ndarray,
    fs: int,
    window_size: int,
    overlap_samples: int,
    n_mels: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a mel-scaled spectrogram and convert to dB.

    Equivalent to MATLAB:
        [S_mel, melFrqs, tm] = melSpectrogram(chunk, fs,
            'Window', hann(windowsize),
            'OverlapLength', overlap_samples,
            'FFTLength', windowsize * 2,
            'NumBands', 128)
        S_mel_dB = 20 * log10(abs(S_mel))

    The hop length (step between windows) is derived from the overlap,
    matching the MATLAB convention: hop = window_size - overlap_samples.

    Parameters
   
    signal          : 1-D float64 array
    fs              : sample rate in Hz
    window_size     : analysis window length in samples
    overlap_samples : number of overlapping samples between windows
    n_mels          : number of mel filter banks (128 matches MATLAB default)

    Returns
    S_mel_db : (n_mels, n_times) array in dB
    mel_freqs: centre frequency of each mel band in Hz
    times    : time axis in seconds
    """
    fft_length = window_size * 2        # matches MATLAB FFTLength argument
    hop_length = window_size - overlap_samples

    S_mel = librosa.feature.melspectrogram(
        y=signal.astype(np.float32),
        sr=fs,
        n_fft=fft_length,
        hop_length=hop_length,
        win_length=window_size,
        window="hann",
        n_mels=n_mels,
    )

    S_mel_db = 20.0 * np.log10(np.abs(S_mel) + 1e-12)

    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=fs / 2.0)
    times = librosa.frames_to_time(
        np.arange(S_mel.shape[1]),
        sr=fs,
        hop_length=hop_length,
    )

    return S_mel_db, mel_freqs, times


def overlap_percent_to_samples(overlap_percent: float, window_size: int) -> int:
    """
    Convert ercentage overlap to a sample count.

    """
    return int(round((overlap_percent / 100.0) * window_size))


def format_frequency_ticks(tick_values: np.ndarray) -> list[str]:
    """
    Format y-axis tick labels as Hz or kHz depending on magnitude.

    Examples
    --------
    [500, 1000, 2000, 10000] -> ['500', '1.0', '2.0', '10.0']
    [1000, 1200, 1400, 1600] -> ['1.0', '1.2', '1.4', '1.6']
    [200, 500, 800]          -> ['200', '500', '800']
    """
    # Separate Hz and kHz values to determine appropriate kHz precision.
    khz_vals = [v / 1000.0 for v in tick_values if v >= 1000]

    if len(khz_vals) >= 2:
        span = max(khz_vals) - min(khz_vals)
        # Choose decimal places so adjacent ticks are distinguishable.
        if span == 0:
            decimals = 1
        elif span < 0.5:
            decimals = 2
        elif span < 5.0:
            decimals = 1
        else:
            decimals = 0
    elif len(khz_vals) == 1:
        # Single kHz tick — show 1 decimal place for clarity.
        decimals = 1
    else:
        decimals = 0

    fmt = f"{{:.{decimals}f}}"

    labels = []
    for val in tick_values:
        if val >= 1000:
            labels.append(fmt.format(val / 1000.0))
        else:
            labels.append(f"{int(round(val))}")
    return labels


def get_frequency_axis_label(max_freq: float) -> str:
    """Return 'Frequency (kHz)' or 'Frequency (Hz)' based on range."""
    return "Frequency (kHz)" if max_freq >= 1000 else "Frequency (Hz)"