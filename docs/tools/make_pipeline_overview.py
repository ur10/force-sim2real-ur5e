"""Build the README hero from existing project evidence with FFmpeg.

Run from any directory: python docs/tools/make_pipeline_overview.py
Requires ffmpeg with libx264/drawtext and DejaVu Sans fonts. No robot connection.
The simulation panel is explicitly a still; hardware excerpts play at 1x.
"""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / 'docs/media'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
DURATION = 12


def run(*args):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *map(str, args)], check=True)


def main():
    with tempfile.TemporaryDirectory(prefix='pipeline-overview-') as tmp:
        filters = [
            '[0:v]scale=352:264,setsar=1[sim]',
            '[1:v]crop=850:480:0:0,scale=352:198,pad=352:264:0:33:color=0x152235,setsar=1[cal]',
            '[2:v]scale=352:326,crop=352:264:0:24,setsar=1[real]',
            'color=c=0x0b1321:s=1200x650:r=10:d=12[bg]',
            '[bg]drawbox=x=16:y=116:w=368:h=416:color=0x152235:t=fill,'
            'drawbox=x=416:y=116:w=368:h=416:color=0x152235:t=fill,'
            'drawbox=x=816:y=116:w=368:h=416:color=0x152235:t=fill[canvas]',
            '[canvas][sim]overlay=24:185[a]',
            '[a][cal]overlay=424:185[b]',
            '[b][real]overlay=824:185[c]',
        ]
        labels = []

        def label(text, x, y, size=20, color='e8eff8', bold=False):
            p = Path(tmp) / f'label-{len(labels)}.txt'
            p.write_text(text)
            labels.append(f"drawtext=fontfile={BOLD if bold else FONT}:textfile={p}:expansion=none:"
                          f'x={x}:y={y}:fontsize={size}:fontcolor=0x{color}')

        label('FORCE-AWARE MANIPULATION  /  UR5e', 24, 20, 16, '73dac5', True)
        label('From simulated contact to real insertion', 24, 52, 37, bold=True)
        for x, title, sub, color in [
            (40, '01  SIMULATE', 'ISAAC LAB  /  REFERENCE STILL', '86b9ff'),
            (440, '02  CALIBRATE', 'REAL ROBOT  /  PAYLOAD FIT', '73dac5'),
            (840, '03  DEPLOY', 'REAL ROBOT  /  EVALUATION CLIP', 'ffca82'),
        ]:
            label(title, x, 132, 25, color, True)
            label(sub, x, 165, 13, color)
        label('→', 386, 305, 30, '73dac5', True)
        label('→', 786, 305, 30, 'ffca82', True)
        label('Contact physics + grasp uncertainty', 30, 466, 17, bold=True)
        label('DAgger + bounded residual PPO', 30, 494, 17, 'b4c3d6')
        label('Measure payload, sensing + frames', 430, 466, 17, bold=True)
        label('Fitted tool payload: 1.35 kg', 430, 494, 17, 'b4c3d6')
        label('Export the policy to the UR5e', 830, 466, 17, bold=True)
        label('30 Hz policy / 120 Hz control', 830, 494, 17, 'b4c3d6')
        label('←  Real measurements feed back into simulation and policy training', 24, 557, 22, '73dac5', True)
        label('Same observation processing in simulation and deployment.', 24, 594, 20, 'b4c3d6')
        label('Independent excerpts at 1× speed; not a synchronized or paired benchmark.', 24, 626, 14, '8b9bb0')
        filters.append('[c]' + ','.join(labels) + '[out]')
        graph = Path(tmp) / 'filters.txt'
        graph.write_text(';\n'.join(filters))
        run('-loop', '1', '-framerate', '10', '-i', MEDIA / 'finray_closeup.jpg',
            '-ss', '18', '-i', MEDIA / 'videos/payload_calibration.mp4',
            '-ss', '12', '-i', MEDIA / 'hardware/videos/0/4.mp4',
            '-filter_complex_threads', '1', '-filter_complex_script', graph,
            '-map', '[out]', '-t', DURATION, '-r', '10', '-an', '-c:v', 'libx264',
            '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', MEDIA / 'pipeline_overview.mp4')
    run('-i', MEDIA / 'pipeline_overview.mp4', '-filter_complex',
        '[0:v]split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];'
        '[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle',
        '-loop', '0', MEDIA / 'pipeline_overview.gif')
    run('-ss', '6', '-i', MEDIA / 'pipeline_overview.mp4', '-frames:v', '1', MEDIA / 'pipeline_overview.png')
    for suffix in ('gif', 'mp4', 'png'):
        p = MEDIA / f'pipeline_overview.{suffix}'
        print(f'{p.relative_to(ROOT)}: {p.stat().st_size / 1024**2:.2f} MiB')


if __name__ == '__main__':
    main()
