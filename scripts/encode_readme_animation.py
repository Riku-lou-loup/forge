"""Encode one frame at a time to keep animation export memory bounded."""

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import GifImagePlugin, Image, ImageDraw

root = Path(__file__).resolve().parents[1]
work = root / ".cache/readme-motion"
checks = {}
for theme, matte in [("light", "#ffffff"), ("dark", "#0d1117")]:
    paths = sorted((work / "frames" / theme).glob("*.png"))
    assert len(paths) == 360
    samples = Image.new("RGB", (1200, 460), matte)
    for n, index in enumerate(range(0, 360, 10)):
        with Image.open(paths[index]) as image:
            rgba = image.convert("RGBA")
            rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, matte), rgba).convert("RGB")
            samples.paste(rgb.resize((200, 76)), ((n % 6) * 200, (n // 6) * 76))
    # Downsampled palette samples can lose the color at the center of thin
    # strokes. Reserve the authored ink colors before adding antialias shades.
    inks = (
        [
            "#e7b969",
            "#ee886b",
            "#8dc5aa",
            "#e5eee8",
            "#a1b7a9",
            "#547665",
            "#385144",
            "#16261e",
            "#294336",
        ]
        if theme == "dark"
        else [
            "#925e12",
            "#b8442c",
            "#287450",
            "#263c2f",
            "#506b59",
            "#729985",
            "#b9cfc0",
            "#f0f5ee",
            "#d3e3d4",
        ]
    )
    fixed = [channel for color in inks for channel in tuple(bytes.fromhex(color[1:]))]
    palette = samples.quantize(colors=246)
    colors = fixed + palette.getpalette()[:738]
    palette.putpalette(colors + colors[:3])
    target = work / f"workflow-{theme}.gif"
    with target.open("wb") as output:
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                rgba = image.convert("RGBA")
            rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, matte), rgba).convert("RGB")
            pixels = np.asarray(rgb.quantize(palette=palette, dither=Image.Dither.NONE)).copy()
            pixels[pixels == 255] = 0
            alpha = np.asarray(rgba.getchannel("A"))
            pixels[alpha < 64] = 255
            assert not ((alpha == 255) & (pixels == 255)).any()
            frame = Image.fromarray(pixels)
            frame.putpalette(colors + [0, 0, 0])
            if index == 0:
                header, _ = GifImagePlugin.getheader(
                    frame,
                    info={"loop": 0, "transparency": 255, "background": 255, "version": b"GIF89a"},
                )
                output.write(b"".join(header))
            output.write(
                b"".join(GifImagePlugin.getdata(frame, duration=40, disposal=2, transparency=255))
            )
        output.write(b";")
    selected = [0, 60, 112, 159, 226, 280]
    sheet = Image.new("RGB", (1200, 752), matte)
    with Image.open(target) as gif:
        assert gif.n_frames == 360
        total = 0
        for i in range(gif.n_frames):
            gif.seek(i)
            image = gif.convert("RGBA")
            assert gif.disposal_method == 2
            assert all(
                image.getpixel(p)[3] == 0 for p in [(0, 0), (1199, 0), (0, 459), (1199, 459)]
            )
            total += gif.info["duration"]
            if i in selected:
                flat = Image.alpha_composite(Image.new("RGBA", image.size, matte), image).convert(
                    "RGB"
                )
                flat.save(work / f"frame-{theme}-{i:03}.png")
                n = selected.index(i)
                sheet.paste(
                    flat.resize((600, 230), Image.Resampling.LANCZOS),
                    ((n % 2) * 600, (n // 2) * 250),
                )
                ImageDraw.Draw(sheet).text(
                    ((n % 2) * 600 + 12, (n // 2) * 250 + 232),
                    f"{i / 25:.2f}s",
                    fill="#a1b7a9" if theme == "dark" else "#506b59",
                )
        assert total == 14400
    sheet.save(work / f"contact-{theme}.png")
    with Image.open(paths[280]) as image:
        image.save(work / f"workflow-static-{theme}.png")
    checks[theme] = {
        "frames": 360,
        "duration_ms": total,
        "bytes": target.stat().st_size,
        "transparent_corners": True,
    }
    print(json.dumps({theme: checks[theme]}), flush=True)
(work / "gif-validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

for source, destination in {
    "workflow-light.gif": "workflow.gif",
    "workflow-dark.gif": "workflow-dark.gif",
    "workflow-static-light.png": "workflow-static.png",
    "workflow-static-dark.png": "workflow-static-dark.png",
    "motion-tokens.json": "motion-tokens.json",
}.items():
    shutil.copy2(work / source, root / "docs/assets" / destination)
