# LAO-X project website

Project website for **Localize Any Object in X-Ray Security Scans without Human Annotation**, NeurIPS 2026.

[Website](https://yaqicai.github.io/LAO-X/) · [Paper](https://arxiv.org/pdf/2610.07326v1) · [Model](https://huggingface.co/YAQICAI/LAO-X) · [MHUG](https://mhug.disi.unitn.it/)

## Branches

- `website-source`: editable HTML, CSS, JavaScript, assets, validation and export scripts.
- `website`: deployable static files. GitHub Pages publishes this branch from `/`.
- `main`: model implementation and repository landing page, maintained separately.

## Development

Requires Node.js 18 or newer. The website has no npm dependencies.

```sh
git clone --branch website-source https://github.com/yaqicai/LAO-X.git
cd LAO-X
npm run dev
```

Open http://localhost:4174/. `parallel.html` contains the same current hero; `single.html` preserves the earlier single-scan animation for comparison.

```sh
npm run check
npm run build
```

`npm run build` validates the source and exports HTML, CSS, JavaScript, `.nojekyll`, and `assets/` to `dist/`. Publish the contents of `dist/` at the root of `website`, preserving its Git history. Build scripts and research records are not included in the deployed site. All site assets use relative paths so the `/LAO-X/` project prefix works.

## Interactive results

The explorer includes 60 scans (10 per dataset), 198 supplied point prompts, and 1,857 retained predictions. It supports granularity selection, an IoU curve, the best granularity for the current point, mask opacity, predicted and GT boxes, mask download, and animated help.

GT annotations contain **bounding boxes**, not pixel-level masks. Explorer IoU is calculated between each prediction's tight mask bounding box and the GT box. It is not segmentation IoU. For a point whose g=1 mask is smaller than its g=0.9 mask, g=1 is excluded before computing the best granularity. Showcase selection uses GT-assisted ranking and is separate from benchmark results. Selection records are in `research/`.

Hero and explorer masks are actual precomputed predictions, not live inference. Transitions blend their display opacity without changing prediction geometry, scores, or downloadable masks. Reduced-motion settings are respected.

To reproduce curation from a separately available source export (Python with Pillow; verification also requires NumPy):

```sh
python scripts/curate-results.py /path/to/Laox-source
python scripts/verify-curation.py /path/to/Laox-source
```

## Research content and assets

Benchmark values come from the paper: Table 1 for box localization and Table 3 for segmentation. The hero's +12.8 mAP-point gain over SAM2 is the equally weighted mean difference across six test-only datasets (24.1633 versus 11.3767 box mAP). Figures and qualitative examples are extracted from the paper. The CT workstation is a schematic illustration; X-ray images and masks are real exported results.

- MHUG logo: [official website](https://mhug.disi.unitn.it/).
- NeurIPS logo: [official Media Kit](https://neurips.cc/public/MediaKit), unchanged SVG artwork with a separate 2026 label.
- GitHub and arXiv icons: Simple Icons v16.0.0 (CC0).
- Hugging Face icon: official Hugging Face website.

The local source PDF is retained for figure provenance. The public paper link opens arXiv. Bibliographic text in the three HTML pages and `assets/lao-x.bib` should be updated together.
