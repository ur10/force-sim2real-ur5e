# Pipeline overview provenance

The README overview is a presentation of existing project evidence, not a new experiment or synchronized comparison.

| Panel | Source | Processing |
| --- | --- | --- |
| Simulate | [Fin-ray simulation reference](finray_closeup.jpg) | Static frame, resized; labeled as a reference still. This is a privileged-expert example, not a residual-policy evaluation. |
| Calibrate | [Payload calibration](videos/payload_calibration.mp4) | Seconds 18–30 at original speed; camera area cropped from the existing annotated recording. |
| Deploy | [Recording 0, camera 4](hardware/videos/0/4.mp4) | Seconds 12–24 at original speed; resized and cropped for the panel. Policy identity is not inferred from the video. |

The 12-second loop places all three stages in view from the first frame. Arrows express the workflow, not a claim that the excerpts were captured together. Calibration measurements feed back into the simulator and training; complete real-contact parameter fitting remains future work. Numerical payload and controller captions come from the [calibration provenance](../CALIBRATION.md) and existing deployment implementation.

Outputs: [GIF](pipeline_overview.gif) for inline GitHub playback, [MP4](pipeline_overview.mp4) for a controlled player, and [PNG](pipeline_overview.png) for a static view. The animation is silent and carries its explanation in visible labels and README alt text. Original evidence files are unchanged.

Rebuild from the repository root:

```bash
python docs/tools/make_pipeline_overview.py
```

Requires FFmpeg with `libx264`, `drawtext`, and the DejaVu Sans font files used by the script. Output is 1200 × 650 at 10 fps; the simulation panel remains still and the hardware clips retain 1× timing.
