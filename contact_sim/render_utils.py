"""Close-up rendering for contact rollouts (import after the Isaac app is launched with cameras).

add_visuals(): dim dome + key light and a table per env.  make_cameras(): one close-up camera per
env looking at the rod tip (create before sim.reset()).  aim(): set the poses after the reset.
Tile helpers label frames and write H.264 videos with ffmpeg.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils


def add_visuals(n: int, table_z: float):
    sim_utils.DomeLightCfg(intensity=450.0).func("/World/DomeLight", sim_utils.DomeLightCfg(intensity=450.0, color=(0.95, 0.95, 1.0)))
    sim_utils.DistantLightCfg(intensity=900.0).func("/World/Sun", sim_utils.DistantLightCfg(intensity=900.0, angle=1.0),
                                                    orientation=(0.85, 0.3, 0.3, 0.3))
    table = sim_utils.CuboidCfg(size=(1.0, 1.4, 0.04), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.78, 0.66, 0.5)))
    for i in range(n):
        table.func(f"/World/envs/env_{i}/Table", table, translation=(0.05, -0.55, table_z - 0.02))


def make_cameras(height: int = 360, width: int = 480):
    from isaaclab.sensors import Camera, CameraCfg
    spawn = sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 20.0))
    return Camera(CameraCfg(prim_path="/World/envs/env_.*/CloseCam", height=height, width=width, data_types=["rgb"], spawn=spawn))


def aim(cam, origins: torch.Tensor, rod_xy, tip_z: float):
    dev = origins.device
    tip = torch.tensor([rod_xy[0], rod_xy[1], tip_z], device=dev)
    cam.set_world_poses_from_view(origins + tip + torch.tensor([0.11, -0.10, 0.05], device=dev),
                                  origins + tip + torch.tensor([0.0, 0.0, 0.004], device=dev))


def grab(sim, cam, dt: float) -> np.ndarray:
    sim.render()
    cam.update(dt, force_recompute=True)
    return cam.data.output["rgb"][..., :3].cpu().numpy()


def label(img: np.ndarray, lines: list[str], footer: str | None = None, footer_ok: bool | None = None) -> np.ndarray:
    import cv2
    im = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
    band = im.copy()
    cv2.rectangle(band, (0, 0), (im.shape[1], 14 + 20 * len(lines)), (25, 25, 25), -1)
    im = cv2.addWeighted(band, 0.6, im, 0.4, 0)
    for j, line in enumerate(lines):
        cv2.putText(im, line, (8, 20 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
    if footer:
        col = (60, 200, 60) if footer_ok else (40, 40, 230) if footer_ok is False else (230, 230, 230)
        cv2.putText(im, footer, (8, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(im, footer, (8, im.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)
    return im


def write_mp4(frames_bgr: list[np.ndarray], path: Path, fps: int = 30):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=path.parent))
    for k, f in enumerate(frames_bgr):
        f = f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2]
        cv2.imwrite(str(tmp / f"f{k:05d}.png"), f)
    subprocess.run(["/usr/bin/ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "f%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "22", str(path)], check=True)
    for f in tmp.glob("*.png"):
        f.unlink()
    tmp.rmdir()
