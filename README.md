# Bad Apple!! as one continuous Turing pattern

Bad Apple!! re-rendered through a sharpen/blur feedback loop. The pattern is
never reset: one field runs for the whole video, and each frame only steers its
local mean. That mean selects the morphology, so black regions grow white spots
on black, white regions black spots on white, and moving edges pass through
labyrinth stripes.

## Run

The source video is not included. Put a copy at `./bad-apple.mp4` or pass
`--src PATH`. Parameters were tuned on a 480x360, 30 fps version. Needs
[uv](https://docs.astral.sh/uv/) and `ffmpeg`.

```bash
uv run render.py --flow 0.08 --flow-scale 40                     # published version -> output/bad_apple_turing.mp4
uv run render.py --start 30 --duration 5 -o output/clip.mp4     # short test clip
```

The full video took about 18 min on 20 CPU cores.

## How it works

Each video frame runs `iters` iterations of:

1. sharpen: `s += gain * (s - G_sigma(s))`
2. blur: `s = G_{sigma * blur_ratio}(s)`
3. steer: `s += force * (m(frame) - G_{3 sigma}(s))`, where `m` maps black and white to `mean_black` and `mean_white`
4. clip to [0, 1]

Without `--flow`, the field freezes wherever the source does not change.
`--flow` advects it along a slowly evolving divergence-free noise field, so the
stripes keep drifting.

## Prior art

Other videos render Bad Apple with blur/sharpen Turing patterns, following
patorjk's `video-to-turing-pattern`, which processes each frame independently.
Here the pattern is carried across frames, its morphology is steered by
brightness, and it drifts.

## Credits

Music: Bad Apple!! feat. nomico, Alstroemeria Records (arr. Masayoshi
Minoshima, lyrics Haruka), original by ZUN (Touhou Project). Original
shadow-art PV: Anira. Neither is included here.
