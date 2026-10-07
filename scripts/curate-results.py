#!/usr/bin/env python3
"""Rank exported predictions against GT boxes and package an auditable showcase.

Requires Pillow. Usage: python scripts/curate-results.py ../Laox-source
No inference, confidence-based ranking, GT masks, or artificial mask growth.
"""
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'det_compass_xp': 'DET-COMPASS', 'CLCXray': 'CLCXray', 'DvXray': 'DvXray',
         'PIXray': 'PIXray', 'HiXray': 'HiXray', 'pidray': 'PIDray'}


def box_iou(pred, gt):
    if pred is None:
        return 0.0
    x, y, w, h = gt
    a, b, c, d = pred  # Pillow bbox has exclusive right/bottom edges.
    intersection = max(0, min(c, x + w) - max(a, x)) * max(0, min(d, y + h) - max(b, y))
    union = (c - a) * (d - b) + w * h - intersection
    return intersection / union if union > 0 else 0.0


def score_case(case, source):
    objects = {str(o['annotation_id']): o for o in case['objects']}
    winners = {}
    for pi, prompt in enumerate(case['prompts']):
        key = str(prompt['annotation_id'])
        assert key in objects
        assert [o['granularity'] for o in prompt['outputs']] == [i / 10 for i in range(1, 11)]
        prompt['excluded_g1'] = prompt['outputs'][-1]['mask_area_pixels'] < prompt['outputs'][-2]['mask_area_pixels']
        for gi, output in enumerate(prompt['outputs']):
            with Image.open(source / output['mask']) as mask:
                assert mask.size == (case['width'], case['height']), output['mask']
                assert mask.mode == 'L', output['mask']
                # Exported masks use 0/255. Thresholding also makes the foreground rule explicit.
                mask = mask.point(lambda v: 255 if v > 127 else 0)
                bbox = mask.getbbox()
                area = mask.histogram()[255]
                assert area == output['mask_area_pixels'], output['mask']
            output['bbox_xyxy'] = bbox
            output['box_iou'] = box_iou(bbox, objects[key]['bbox'])
            if gi == 9 and prompt['excluded_g1']:
                continue
            candidate = {'prompt_index': pi, 'output_index': gi, 'box_iou': output['box_iou']}
            # Deterministic ties: first source prompt, then first granularity.
            if key not in winners or candidate['box_iou'] > winners[key]['box_iou']:
                winners[key] = candidate
        if prompt['excluded_g1']:
            removed = prompt['outputs'].pop()
            prompt['_excluded_output'] = removed
            prompt['exclusion'] = {'granularity': 1.0, 'area': removed['mask_area_pixels'], 'g09_area': prompt['outputs'][-1]['mask_area_pixels'], 'reason': 'g1_area_less_than_g09'}
        prompt['best_output'] = max(range(len(prompt['outputs'])), key=lambda i: prompt['outputs'][i]['box_iou'])
    assert set(winners) == set(objects)
    for obj in case['objects']:
        obj['best'] = winners[str(obj['annotation_id'])]
    scores = [v['box_iou'] for v in winners.values()]
    case['score'] = sum(scores) / len(scores)
    case['min_object_iou'] = min(scores)
    best = max(winners.values(), key=lambda v: v['box_iou'])
    case['default_prompt'] = best['prompt_index']
    case['default_output'] = best['output_index']
    return case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    manifest_bytes = (source / 'manifest.json').read_bytes()
    data = json.loads(manifest_bytes)
    source_prediction_count = sum(len(p['outputs']) for c in data['cases'] for p in c['prompts'])
    assert len(data['cases']) == 600
    assert set(Counter(c['dataset'] for c in data['cases']).values()) == {100}
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, case in enumerate(pool.map(lambda c: score_case(c, source), data['cases']), 1):
            results.append(case)
            if i % 100 == 0:
                print(f'Scored {i}/600 images', flush=True)
    ranked = []
    for dataset in data['datasets']:
        group = sorted((c for c in results if c['dataset'] == dataset), key=lambda c: (-c['score'], -c['min_object_iou'], str(c['id'])))
        for rank, case in enumerate(group, 1):
            case['rank'] = rank
            ranked.append(case)
    selected = [c for c in ranked if c['rank'] <= 10]
    out_dir = ROOT / 'assets' / 'real'
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / 'research'
    report_dir.mkdir(exist_ok=True)
    with (report_dir / 'selected-top10.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'rank', 'image_id', 'file_name', 'mean_best_box_iou', 'gt_objects', 'default_point_index', 'default_granularity', 'image'])
        for c in selected:
            prompt = c['prompts'][c['default_prompt']]
            output = prompt['outputs'][c['default_output']]
            image = f"assets/real/{c['dataset']}/{c['id']}/image.png"
            writer.writerow([c['dataset'], c['rank'], c['id'], c['file_name'], c['score'], len(c['objects']), prompt['point_index'], output['granularity'], image])
    with (report_dir / 'image-ranking.csv').open('w', newline='') as f:
        fields = ['dataset', 'rank', 'id', 'file_name', 'score', 'min_object_iou', 'object_count', 'selected']
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for c in ranked:
            writer.writerow({**{k: c[k] for k in fields[:6]}, 'object_count': len(c['objects']), 'selected': c['rank'] <= 10})
    report = {
        'source_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'checkpoint_sha256': data['checkpoint_sha256'],
        'metric': 'Mean across GT objects of best tight-mask-box IoU over supplied prompts and eligible granularities.',
        'exclusion_rule': 'For each point independently, omit g=1 if its foreground area is strictly smaller than g=0.9. Recompute prompt optima, object optima and image ranking.',
        'excluded': [{'dataset': c['dataset'], 'image_id': c['id'], 'annotation_id': p['annotation_id'], 'point_index': p['point_index'], **p['exclusion']} for c in results for p in c['prompts'] if p['excluded_g1']],
        'foreground': 'mask > 127; xyxy uses exclusive right/bottom edges; GT is unrounded xywh; no GT clipping.',
        'ties': 'Image score descending, minimum object IoU descending, string image ID ascending. Output ties use source order.',
        'scope': 'GT-assisted qualitative selection from 100 supplied images per dataset; not mask IoU, AP, or an unbiased benchmark.',
        'gt_type': 'bounding boxes only',
        'image_count': len(results),
        'prediction_count': sum(len(p['outputs']) for c in results for p in c['prompts']),
        'source_prediction_count': source_prediction_count,
        'cases': [{k: c[k] for k in ['dataset', 'id', 'file_name', 'rank', 'score', 'min_object_iou', 'objects', 'default_prompt', 'default_output']} for c in ranked]
    }
    (report_dir / 'selection-report.json').write_text(json.dumps(report, indent=2))
    packaged = []
    sheet = Image.new('RGB', (1800, 6 * 170), '#edf3f9')
    draw = ImageDraw.Draw(sheet)
    for i, c in enumerate(selected):
        target = out_dir / c['dataset'] / str(c['id'])
        target.mkdir(parents=True, exist_ok=True)
        public = {k: c[k] for k in ['dataset', 'id', 'file_name', 'width', 'height', 'rank', 'score', 'min_object_iou', 'objects', 'default_prompt', 'default_output']}
        public['image'] = str((target / 'image.png').relative_to(ROOT))
        shutil.copy2(source / c['image'], target / 'image.png')
        public['prompts'] = []
        for p in c['prompts']:
            if p['excluded_g1']:
                removed = p['_excluded_output']
                # Delete only generated website copies. Never modify the source export.
                for name in [Path(removed['mask']).name, Path(removed['overlay']).name, Path(removed['mask']).stem + '_display.png']:
                    (target / name).unlink(missing_ok=True)
            prompt = {k: p[k] for k in ['annotation_id', 'point_index', 'point', 'label', 'excluded_g1', 'best_output']}
            prompt['outputs'] = []
            for o in p['outputs']:
                output = {k: o[k] for k in ['granularity', 'predicted_iou', 'mask_area_pixels', 'bbox_xyxy', 'box_iou']}
                for key in ['mask', 'overlay']:
                    path = target / Path(o[key]).name
                    shutil.copy2(source / o[key], path)
                    output[key] = str(path.relative_to(ROOT))
                # Color only: preserve every predicted foreground pixel at full alpha,
                # allowing the website to set any display opacity independently.
                display_path = target / (Path(o['mask']).stem + '_display.png')
                with Image.open(source / o['mask']) as binary:
                    rgba = Image.new('RGBA', binary.size, (0, 210, 170, 0))
                    rgba.putalpha(binary)
                    rgba.save(display_path, optimize=True)
                output['display'] = str(display_path.relative_to(ROOT))
                prompt['outputs'].append(output)
            public['prompts'].append(prompt)
        with Image.open(source / c['image']) as im:
            thumb = im.convert('RGBA')
        for obj in c['objects']:
            best = obj['best']
            path = c['prompts'][best['prompt_index']]['outputs'][best['output_index']]['overlay']
            with Image.open(source / path) as overlay:
                thumb = Image.alpha_composite(thumb, overlay)
        thumb = ImageOps.contain(thumb.convert('RGB'), (320, 210))
        thumb.save(target / 'thumb.webp', quality=85)
        public['thumbnail'] = str((target / 'thumb.webp').relative_to(ROOT))
        small = ImageOps.contain(thumb, (174, 130))
        x, y = (i % 10) * 180, (i // 10) * 170
        sheet.paste(small, (x + (180 - small.width) // 2, y))
        draw.text((x + 4, y + 132), f"{NAMES[c['dataset']]} #{c['rank']} {c['score']:.3f}", fill='#18334d')
        draw.text((x + 4, y + 147), str(c['id'])[:24], fill='#18334d')
        packaged.append(public)
    payload = {'datasets': [{'id': d, 'name': NAMES[d]} for d in data['datasets']], 'granularities': data['granularities'], 'cases': packaged}
    (out_dir / 'manifest.json').write_text(json.dumps(payload, separators=(',', ':')))
    (out_dir / 'manifest.js').write_text('window.LAOX_REAL = ' + json.dumps(payload, separators=(',', ':')) + ';\n')
    sheet.save(report_dir / 'selected-contact-sheet.jpg', quality=92)
    print(f"Packaged 60 images. Excluded {len(report['excluded'])} shrinking g=1 outputs across all source prompts.")
    for d in data['datasets']:
        group = [c for c in selected if c['dataset'] == d]
        print(f"{NAMES[d]}: {group[0]['score']:.6f} – {group[-1]['score']:.6f}")


if __name__ == '__main__':
    main()
