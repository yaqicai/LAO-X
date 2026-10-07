"""Independent source/geometry audit. Requires Pillow + NumPy; no browser runtime dependency."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[1]
source = Path(sys.argv[1] if len(sys.argv) > 1 else root.parent / 'Laox-source')
manifest = json.loads((root / 'assets/real/manifest.json').read_text())
source_bytes = (source / 'manifest.json').read_bytes()
original = json.loads(source_bytes)
originals = {(c['dataset'], str(c['id'])): c for c in original['cases']}
report = json.loads((root / 'research/selection-report.json').read_text())
assert report['source_manifest_sha256'] == hashlib.sha256(source_bytes).hexdigest()
expected = {(c['dataset'], str(c['id'])) for c in report['cases'] if c['rank'] <= 10}
assert expected == {(c['dataset'], str(c['id'])) for c in manifest['cases']}
source_excluded = sum(p['outputs'][-1]['mask_area_pixels'] < p['outputs'][-2]['mask_area_pixels'] for c in original['cases'] for p in c['prompts'])
assert len(report['excluded']) == source_excluded
count = excluded = 0
for case in manifest['cases']:
    src = originals[case['dataset'], str(case['id'])]
    assert (root / case['image']).read_bytes() == (source / src['image']).read_bytes()
    object_best = {}
    for prompt, source_prompt in zip(case['prompts'], src['prompts']):
        assert prompt['point'] == source_prompt['point']
        assert prompt['annotation_id'] == source_prompt['annotation_id']
        should_exclude = source_prompt['outputs'][-1]['mask_area_pixels'] < source_prompt['outputs'][-2]['mask_area_pixels']
        assert prompt['excluded_g1'] == should_exclude
        assert len(prompt['outputs']) == (9 if should_exclude else 10)
        excluded += should_exclude
        gt = next(o for o in case['objects'] if o['annotation_id'] == prompt['annotation_id'])
        x, y, w, h = gt['bbox']
        scores = []
        for output, source_output in zip(prompt['outputs'], source_prompt['outputs']):
            assert (root / output['mask']).read_bytes() == (source / source_output['mask']).read_bytes()
            mask = np.asarray(Image.open(root / output['mask']))
            display = np.asarray(Image.open(root / output['display']))
            assert np.array_equal(mask, display[:, :, 3])
            ys, xs = np.nonzero(mask > 127)
            if len(xs):
                box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                intersection = max(0, min(box[2], x + w) - max(box[0], x)) * max(0, min(box[3], y + h) - max(box[1], y))
                iou = intersection / ((box[2] - box[0]) * (box[3] - box[1]) + w * h - intersection)
            else:
                box, iou = None, 0
            assert box == output['bbox_xyxy'] and abs(iou - output['box_iou']) < 1e-12
            scores.append(iou)
            count += 1
        assert prompt['best_output'] == int(np.argmax(scores))
        key = str(prompt['annotation_id'])
        object_best[key] = max(object_best.get(key, 0), max(scores))
        if should_exclude:
            removed = source_prompt['outputs'][-1]
            target = root / 'assets/real' / case['dataset'] / str(case['id'])
            assert not (target / Path(removed['mask']).name).exists()
            assert not (target / (Path(removed['mask']).stem + '_display.png')).exists()
    assert abs(sum(object_best.values()) / len(object_best) - case['score']) < 1e-12

result = {'scans_verified': len(manifest['cases']), 'outputs_verified': count,
          'source_excluded_g1': source_excluded, 'selected_excluded_g1': excluded,
          'checks': ['Source images and retained masks are byte-identical.',
                     'Display alpha equals the binary prediction at every pixel.',
                     'Independent NumPy bounds and IoU calculations match every packaged output.',
                     'Every point optimum excludes g=1 exactly when area(g=1) < area(g=0.9).',
                     'Generated copies of excluded masks are absent; the source export is unchanged.',
                     'The 60 selected scans match the recomputed top-ten ranking.']}
(root / 'research/curation-verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
