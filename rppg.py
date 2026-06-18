import cv2
import numpy as np
from dataclasses import dataclass
from scipy.signal import butter, filtfilt, detrend

@dataclass
class VitalsResult:
    heart_rate_bpm: float
    respiration_rate_bpm: float
    signal_quality: float
    samples: int
    fps: float
    filtered_signal: list
    frequencies: list
    spectrum: list


# -----------------------------
# Helper: Bandpass Filter
# -----------------------------
def bandpass(signal, fs, low, high, order=3):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


# -----------------------------
# Extract green channel signal
# -----------------------------
def extract_signal(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    signal = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (320, 240))

        # Take central face region (simple ROI)
        h, w, _ = frame.shape
        roi = frame[h//4:h//2, w//4:3*w//4]

        green_mean = np.mean(roi[:, :, 1])
        signal.append(green_mean)

    cap.release()
    return np.array(signal), fps


# -----------------------------
# Main Analysis Function
# -----------------------------
def analyse_video(video_path):
    raw_signal, fps = extract_signal(video_path)

    if len(raw_signal) < 50:
        raise Exception("Video too short / insufficient frames")

    # Remove trend
    signal = detrend(raw_signal)

    # Normalize
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

    # -----------------------------
    # Heart Rate (0.7 - 3.5 Hz)
    # -----------------------------
    filtered = bandpass(signal, fps, 0.7, 3.5)

    freqs = np.fft.rfftfreq(len(filtered), d=1/fps)
    spectrum = np.abs(np.fft.rfft(filtered))

    # Convert to BPM range
    bpm_freqs = freqs * 60

    valid_idx = np.where((bpm_freqs >= 40) & (bpm_freqs <= 200))[0]

    if len(valid_idx) == 0:
        hr = 0
    else:
        peak_idx = valid_idx[np.argmax(spectrum[valid_idx])]
        hr = bpm_freqs[peak_idx]

    # -----------------------------
    # Respiration (0.1 - 0.5 Hz)
    # -----------------------------
    resp_signal = bandpass(signal, fps, 0.1, 0.5)

    resp_freqs = np.fft.rfftfreq(len(resp_signal), d=1/fps)
    resp_spectrum = np.abs(np.fft.rfft(resp_signal))

    resp_bpm_freqs = resp_freqs * 60
    valid_resp = np.where((resp_bpm_freqs >= 6) & (resp_bpm_freqs <= 30))[0]

    if len(valid_resp) == 0:
        rr = 0
    else:
        peak_idx = valid_resp[np.argmax(resp_spectrum[valid_resp])]
        rr = resp_bpm_freqs[peak_idx]

    # -----------------------------
    # Signal Quality (simple SNR)
    # -----------------------------
    signal_power = np.max(spectrum)
    noise_power = np.mean(spectrum)

    quality = signal_power / (noise_power + 1e-6)
    quality = min(1.0, quality / 10)  # normalize to 0–1

    return VitalsResult(
        heart_rate_bpm=float(hr),
        respiration_rate_bpm=float(rr),
        signal_quality=float(quality),
        samples=len(signal),
        fps=fps,
        filtered_signal=filtered.tolist(),
        frequencies=freqs.tolist(),
        spectrum=spectrum.tolist()
    )