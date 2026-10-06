#!/usr/bin/env python3
"""Assemble and verify renders made with --mounted --mount-angle 15/32/42/55."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/camera-short-tilt-study'
ANGLES=[15,32,42,55]
SCENES=['bench_grasp','bench_transfer','pool_end','pool_end_back','pool_side','pool_side_back']
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
reports={a:json.loads((OUT/f'angle_{a}/report.json').read_text()) for a in ANGLES}
rows=[]
for scene in SCENES:
    full=Image.new('RGB',(2000,4*850))
    for i,a in enumerate(ANGLES):
        row=next(r for r in reports[a]['results'] if r['scene']==scene)
        assert abs(row['pitch_deg']-a)<1e-8
        if i==0:q=np.array(row['robot_q'])
        else:np.testing.assert_allclose(row['robot_q'],q,atol=1e-12)
        with Image.open(OUT/f'angle_{a}/{scene}_mounted_pair.jpg') as im:
            im.load();full.paste(im,(0,850*i))
        rows.append(row)
    full.save(OUT/f'{scene}_tilt_comparison.jpg',quality=94)
summary=Image.new('RGB',(1920,90+6*325),'#111c2c');draw=ImageDraw.Draw(summary)
draw.text((18,12),'SAME SHORT SUPPORT: 65 mm shelf datum, triangular ribs | only shelf/camera tilt changes',font=font,fill='white')
for j,a in enumerate(ANGLES):draw.text((480*j+12,53),f'{a} degrees downward',font=font,fill='#86dce8')
for i,scene in enumerate(SCENES):
    for j,a in enumerate(ANGLES):
        row=next(r for r in reports[a]['results'] if r['scene']==scene)
        x,y=480*j,90+325*i
        with Image.open(OUT/f'angle_{a}/{scene}_short{a}.png') as im:summary.paste(im.resize((480,270)),(x,y))
        draw.text((x+10,y+273),scene.replace('_',' '),font=small,fill='white')
        draw.text((x+10,y+296),f"Corners {row['surface_corners_in_image']}/4 | targets {row['objects_unoccluded']}/{row['objects_total']}",font=small,fill='#a8e4c9')
summary.save(OUT/'short_support_tilt_summary.jpg',quality=94)
# A single close-up row makes the preserved structure and changed tilt explicit.
closeup=Image.new('RGB',(2000,390),'#111c2c');draw=ImageDraw.Draw(closeup)
for j,a in enumerate(ANGLES):
    draw.text((j*500+15,12),f'65 mm shelf datum | {a} deg',font=font,fill='white')
    with Image.open(OUT/f'angle_{a}/bench_grasp_mount_detail.png') as im:closeup.paste(im.resize((500,290)),(j*500,60))
draw.text((15,360),'Same base, rib spacing/thickness and lift position. Upper support edges meet each tilted shelf.',font=small,fill='#b8cddd')
closeup.save(OUT/'short_support_four_angles.jpg',quality=95)
(OUT/'comparison_report.json').write_text(json.dumps({'simulation_only':True,'shelf_midplane_mm':65,'angles_deg':ANGLES,'robot_poses_equal_across_angles':True,'results':rows},indent=2))
print('Assembled 24 angle/scenario cases; all robot joint poses identical across angles.')
