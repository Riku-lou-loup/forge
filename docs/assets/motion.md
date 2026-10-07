# README motion

The selected interval stays recognizable as the recording is inspected, the evidence is gathered and the draft is assembled. One action leads at a time. Citation numbers stay attached to their passages. The draft remains unreviewed.

The source defines its motion tokens next to its color tokens. `motion-tokens.json` is exported by the renderer. These tokens belong to this README scene; they do not change the Streamlit app.

| Token | Value | Use | Local HTML example |
| --- | --- | --- | --- |
| `duration-fast` | 160 ms | Read-head exit | [Signal sweep](workflow.html?time=1.7) |
| `duration-normal` | 320 ms | Passage highlight and review-status entry | [Highlight](workflow.html?time=4.8) |
| `duration-slow` | 640 ms | Notes gathering and citation transfer | [Citations](workflow.html?time=8.95) |
| `duration-deliberate` | 800 ms | Interval isolation and report unfolding | [Interval](workflow.html?time=2.5) |
| `stagger` | 40 ms | Note entrances, 80 ms across three items | [Notes](workflow.html?time=3.9) |

`ease-standard` is cubic-bezier(.2, 0, 0, 1) for changes of position. `ease-enter` is cubic-bezier(.16, 1, .3, 1) for arrivals. `ease-exit` is cubic-bezier(.4, 0, 1, 1) for departure. Endpoints return exactly zero or one so completed packets disappear and the loop closes cleanly.

## Choreography

The eight traces begin visible. A read head crosses the recording and a bracket selects an interval. That same interval expands while the surrounding channels contract. Three source notes fan out, passages are highlighted, and the pages fold into excerpts. The selected waveform moves into the draft; citation IDs follow their curved routes to the corresponding evidence rows. The assembled report holds before the scene returns to the original signal field.

Text stays still within each object. There is no ambient particle field, flashing status, animated wordmark or simulated approval. Signals and pacing explain the workflow rather than reproduce a timed inference run.

## Reduced motion and verification

The root reduced-motion rule sets movement durations and stagger to zero. Playback starts paused on the assembled report. The README uses matching static PNGs for the same preference because a GIF cannot change its own playback behavior. The HTML source also provides pause and seek controls.

Verification covers all 360 decoded frames in each theme, transparent corners, frame disposal, total duration, matching loop endpoints and preservation of the selected interval. Source controls and reduced motion were checked in headless Chrome. A 4× CPU-throttled render check stayed under a 40 ms frame budget; this is an emulated check, not a claim about every device.
