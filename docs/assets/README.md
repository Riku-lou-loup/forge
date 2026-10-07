# README media

- `app.png` is a browser capture of the local Streamlit application.
- `recorded-signals.png` uses the pinned SKAB development sample and its source annotations. Coral points are annotations, not predictions.
- `grouped-audit.png` reads `docs/generalization-results.json`. It reports pooled outer-fold results on the previously inspected development pool.
- The architecture is a native Mermaid code block in the project README.
- `workflow.gif` and `workflow-dark.gif` illustrate a selected interval, supporting notes and a cited draft. The signals and timing are illustrative. Note titles come from `knowledge/playbook.json`; their order is not a retrieval benchmark. The outer background is transparent.
- `workflow.html` is the editable canvas source, with pause, seeking and a static reduced-motion state. It contains no project title or wordmark in the animation. The app has a separate compact scene in `src/forge/ui/motion.py`.

The README's picture element selects the light or dark GIF and substitutes the matching static PNG when the browser requests reduced motion. Static links remain available. The animation loops in 14.4 seconds at 25 frames per second. Its opening and closing decoded frames match.

## Rebuild

Rebuild the data figures with `.venv/Scripts/python.exe scripts/build_readme_media.py` after downloading the sample. SKAB attribution and source checksums are recorded in `data/sample-manifest.json`.

To rebuild the animation, use a Node environment with Playwright and its Chromium browser (`npx playwright install chromium`), plus the project Python environment with Pillow and NumPy:

```powershell
node scripts/render_readme_animation.cjs
.\.venv\Scripts\python.exe scripts/encode_readme_animation.py
```

`FORGE_PLAYWRIGHT_PATH` can select an existing Playwright installation and `FORGE_BROWSER_BINARY` can select a browser executable. Frames and validation captures go to ignored `.cache/readme-motion/`. The encoder processes one frame at a time to bound memory use, then writes the GIFs, static images and token export here.

[Motion tokens and choreography](motion.md) describe the timing rules and link to seekable examples in the local HTML source.
