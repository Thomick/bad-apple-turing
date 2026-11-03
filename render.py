# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python", "numpy", "tqdm"]
# ///
"""Bad Apple rendered as a living Turing pattern.

A persistent sharpen/blur feedback field evolves continuously; each video frame
only steers its local mean, which selects the pattern morphology (spots /
labyrinth / holes).
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_SRC = Path("bad-apple.mp4")


def gblur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur. In: float32 field, sigma (px). Out: blurred field."""
    return cv2.GaussianBlur(x, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def turing_step(s: np.ndarray, target: np.ndarray, p: argparse.Namespace) -> np.ndarray:
    """One sharpen/blur iteration steered toward a target mean.

    In: state s in [0,1], target mean field in [0,1], params. Out: new state.
    """
    s = s + p.gain * (s - gblur(s, p.sigma))          # sharpen: amplifies wavelengths ~ sigma
    s = gblur(s, p.sigma * p.blur_ratio)              # blur: kills the finest scales
    s = s + p.force * (target - gblur(s, p.sigma * 3))  # steer local mean, not pixels
    return np.clip(s, 0.0, 1.0)


class CurlFlow:
    """Smooth, divergence-free, slowly evolving velocity field (curl noise).

    A coarse random stream function is cross-faded between keyframes every
    `period` frames, upsampled, and differentiated: v = (dpsi/dy, -dpsi/dx).
    """

    def __init__(self, size: tuple[int, int], speed: float, scale: float, period: int, seed: int):
        self.w, self.h = size
        self.speed, self.period = speed, period
        self.grid = (max(2, round(self.w / scale)), max(2, round(self.h / scale)))
        self.rng = np.random.default_rng(seed + 1)
        self.keys = [self._key(), self._key()]
        xs, ys = np.meshgrid(np.arange(self.w, dtype=np.float32), np.arange(self.h, dtype=np.float32))
        self.xs, self.ys = xs, ys

    def _key(self) -> np.ndarray:
        """Random coarse stream function (upsampled lazily)."""
        k = self.rng.standard_normal((self.grid[1], self.grid[0])).astype(np.float32)
        return cv2.resize(k, (self.w, self.h), interpolation=cv2.INTER_CUBIC)

    def maps(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """Backward-advection remap coordinates for one iteration at this frame."""
        phase = frame % self.period
        if phase == 0 and frame > 0:
            self.keys = [self.keys[1], self._key()]
        a = phase / self.period
        a = a * a * (3 - 2 * a)
        psi = (1 - a) * self.keys[0] + a * self.keys[1]
        dpy, dpx = np.gradient(psi)
        vx, vy = dpy, -dpx
        norm = self.speed / (np.sqrt(np.mean(vx**2 + vy**2)) + 1e-8)
        return self.xs - vx * norm, self.ys - vy * norm


def pattern_noise(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Unit-variance white noise smoothed to the pattern scale."""
    n = gblur(rng.standard_normal(shape, dtype=np.float32), sigma)
    return n / (n.std() + 1e-8)


def advect(s: np.ndarray, maps: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Semi-Lagrangian transport of the field along the flow."""
    return cv2.remap(s, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def target_field(frame: np.ndarray, size: tuple[int, int], p: argparse.Namespace) -> np.ndarray:
    """Map a video frame to the desired local mean. In: BGR frame. Out: float32 field."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g = cv2.resize(g, size, interpolation=cv2.INTER_LINEAR)
    g = gblur(g, p.edge_soft)
    return p.mean_black + (p.mean_white - p.mean_black) * g


def colorize(s: np.ndarray, p: argparse.Namespace) -> np.ndarray:
    """Field to output BGR uint8 (grayscale or a two-tone palette)."""
    s = np.clip((s - 0.5) * p.contrast + 0.5, 0, 1)
    if p.palette == "gray":
        return cv2.cvtColor((s * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    lo, hi = PALETTES[p.palette]
    out = np.array(lo, np.float32) + s[..., None] * (np.array(hi, np.float32) - np.array(lo, np.float32))
    return out.astype(np.uint8)


PALETTES = {  # BGR (dark, light)
    "ink": ((30, 20, 15), (225, 240, 245)),
    "ember": ((20, 10, 40), (120, 200, 255)),
    "cyan": ((40, 15, 10), (240, 230, 120)),
}


def open_writer(path: Path, size: tuple[int, int], fps: float, src: Path, start: float, dur: float | None):
    """ffmpeg pipe: raw BGR frames in, H.264 + source audio out."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
           "-ss", str(start), "-i", str(src)]
    if dur is not None:
        cmd += ["-t", str(dur)]
    cmd += ["-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("-o", "--out", type=Path, default=Path("output/bad_apple_turing.mp4"))
    ap.add_argument("--start", type=float, default=0.0, help="start time (s)")
    ap.add_argument("--duration", type=float, default=None, help="length (s); default: whole video")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--stills", type=Path, default=None, help="dump every Nth frame as PNG here instead of video")
    ap.add_argument("--still-every", type=int, default=30)
    # dynamics
    ap.add_argument("--sigma", type=float, default=4.0, help="pattern scale (px at output res)")
    ap.add_argument("--gain", type=float, default=4.0)
    ap.add_argument("--blur-ratio", type=float, default=0.5)
    ap.add_argument("--force", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=4, help="Turing iterations per video frame")
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--mean-black", type=float, default=0.08)
    ap.add_argument("--mean-white", type=float, default=0.92)
    ap.add_argument("--edge-soft", type=float, default=3.0)
    # motion (0 = v1 static behaviour)
    ap.add_argument("--flow", type=float, default=0.0, help="RMS drift speed (px per iteration)")
    ap.add_argument("--flow-scale", type=float, default=160.0, help="eddy size (px)")
    ap.add_argument("--flow-period", type=int, default=90, help="frames between flow keyframes")
    ap.add_argument("--noise", type=float, default=0.0, help="std of smoothed noise added per iteration")
    ap.add_argument("--noise-scale", type=float, default=0.5, help="noise smoothing, in units of sigma")
    ap.add_argument("--noise-corr", type=float, default=0.0,
                    help="frame-to-frame noise correlation (0 = fresh each iteration, ->1 = slowly wandering)")
    # look
    ap.add_argument("--palette", choices=["gray", *PALETTES], default="gray")
    ap.add_argument("--contrast", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    p = ap.parse_args()

    if not p.src.is_file():
        ap.error(f"source video not found: {p.src} (see README)")
    cap = cv2.VideoCapture(str(p.src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (round(w0 * p.height / h0 / 2) * 2, p.height)
    first = int(round(p.start * fps))
    n = n_total - first if p.duration is None else min(int(round(p.duration * fps)), n_total - first)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    ok, frame = cap.read()
    if not ok:
        ap.error(f"no frame at {p.start}s in {p.src}")
    target = target_field(frame, size, p)
    rng = np.random.default_rng(p.seed)
    s = np.clip(target + 0.1 * rng.standard_normal(target.shape).astype(np.float32), 0, 1)
    for _ in range(p.warmup):
        s = turing_step(s, target, p)

    flow = CurlFlow(size, p.flow, p.flow_scale, p.flow_period, p.seed) if p.flow > 0 else None
    writer = None
    if p.stills:
        p.stills.mkdir(parents=True, exist_ok=True)
    else:
        p.out.parent.mkdir(parents=True, exist_ok=True)
        writer = open_writer(p.out, size, fps, p.src, p.start, p.duration)

    for i in tqdm(range(n), unit="f"):
        if i > 0:
            ok, frame = cap.read()
            if not ok:
                break
            target = target_field(frame, size, p)
        maps = flow.maps(i) if flow else None
        if p.noise > 0 and p.noise_corr > 0:  # Ornstein-Uhlenbeck: one slowly evolving field per frame
            fresh = pattern_noise(rng, s.shape, p.sigma * p.noise_scale)
            nfield = fresh if i == 0 else p.noise_corr * nfield + np.sqrt(1 - p.noise_corr**2) * fresh
        for _ in range(p.iters):
            if maps:
                s = advect(s, maps)
            if p.noise > 0:
                eps = nfield if p.noise_corr > 0 else pattern_noise(rng, s.shape, p.sigma * p.noise_scale)
                s = s + p.noise * eps
            s = turing_step(s, target, p)
        img = colorize(s, p)
        if writer:
            writer.stdin.write(img.tobytes())
        elif i % p.still_every == 0:
            cv2.imwrite(str(p.stills / f"f{first + i:05d}.png"), img)

    if writer:
        writer.stdin.close()
        writer.wait()
        print(f"wrote {p.out} ({n} frames, {size[0]}x{size[1]})")


if __name__ == "__main__":
    main()
