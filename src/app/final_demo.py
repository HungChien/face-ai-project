from __future__ import annotations

import argparse
import html
import json
import os
import webbrowser
from dataclasses import asdict, dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "final_demo"


@dataclass(frozen=True)
class ModuleResult:
    key: str
    title: str
    stage: str
    status: str
    summary: str
    metrics: tuple[str, ...]
    media: tuple[str, ...]
    reports: tuple[str, ...]
    entrypoints: tuple[str, ...]


MODULES = (
    ModuleResult(
        key="datasets",
        title="Dataset exploration",
        stage="Data foundations",
        status="complete",
        summary="LFW and CelebA formats, annotations, distributions, pairs, and preprocessing were inspected with OpenCV and Matplotlib.",
        metrics=("LFW: 13,233 images / 5,749 identities", "CelebA: 202,599 images / 40 attributes"),
        media=("outputs/images/lfw_opencv_pair_examples.jpg", "outputs/images/celeba_opencv_annotation_examples.jpg"),
        reports=("outputs/reports/lfw_exploration_result.txt", "outputs/reports/celeba_exploration_result.txt"),
        entrypoints=("src/datasets/explore_lfw.py", "src/datasets/explore_lfw_pairs.py", "src/datasets/explore_celeba.py"),
    ),
    ModuleResult(
        key="detection",
        title="Face detection",
        stage="Detection stack",
        status="partial",
        summary="MMDetection inference and a RetinaNet WIDER FACE training pipeline were completed; the formal run was limited to one CPU epoch.",
        metrics=("WIDER FACE epoch-1 mAP/AP50: 0.3283", "MMDetection test image: 1 face"),
        media=("outputs/mmdetection_face_detection/vis/face_test.jpg", "outputs/images/widerface_epoch1_checkpoint/face_test_debug_det.jpg"),
        reports=("outputs/reports/mmdetection_face_detection_result.txt", "outputs/reports/widerface_epoch1_eval_result.txt"),
        entrypoints=("src/mmdetection/run_face_detection_mmdet.py", "src/mmdetection/train_widerface.py", "src/mmdetection/evaluate_widerface.py"),
    ),
    ModuleResult(
        key="landmarks",
        title="Landmarks and alignment",
        stage="Geometry stack",
        status="complete",
        summary="A 68-point 300W regressor, five-point extraction, calibrated alignment template, and GT-versus-prediction comparison were implemented.",
        metrics=("Best validation NME: 0.1721", "Best checkpoint epoch: 29 / 30"),
        media=("outputs/landmarks/landmark_cnn_300w_aug30_predictions.jpg", "outputs/landmarks/alignment_compare_300w/gt_vs_pred_alignment_grid.jpg"),
        reports=("outputs/reports/landmark_300w_aug30_training_result.txt", "outputs/reports/landmark_300w_alignment_compare_result.txt"),
        entrypoints=("src/landmarks/train_landmark_regressor.py", "src/landmarks/align_with_landmark_model.py", "src/landmarks/compare_gt_pred_alignment.py"),
    ),
    ModuleResult(
        key="recognition",
        title="Face recognition",
        stage="Recognition stack",
        status="partial",
        summary="ResNet50 + ArcFace was trained on a 10,000-identity aligned MS1M subset and evaluated with the official LFW 6,000-pair 10-fold protocol.",
        metrics=("Closed-set validation accuracy: 97.94%", "Self-trained LFW accuracy: 76.98%", "Face-pretrained reference: 96.77%"),
        media=("outputs/images/resnet50_ms1m10000_reproduce_server_curves.jpg",),
        reports=(
            "outputs/reports/resnet50_ms1m10000_reproduce_server_result.txt",
            "outputs/reports/resnet50_ms1m10000_reproduce_server_lfw_10fold_result.txt",
            "outputs/reports/facenet_vggface2_lfw_10fold_result.txt",
        ),
        entrypoints=("src/recognition/train_arcface_celeba_subset.py", "src/recognition/evaluate_lfw_10fold_resnet_arcface.py", "src/recognition/evaluate_lfw_10fold_facenet.py"),
    ),
    ModuleResult(
        key="optimization",
        title="Optimization and deployment",
        stage="Deployment stack",
        status="complete",
        summary="The final ArcFace model was dynamically quantized, exported to ONNX, and benchmarked with output-consistency checks.",
        metrics=("Quantized: -20.21% size / 1.174x CPU speedup", "ONNXRuntime: 1.170x CPU speedup"),
        media=(),
        reports=("outputs/reports/resnet50_ms1m10000_dynamic_quantization_report.txt", "outputs/reports/resnet50_ms1m10000_onnx_export_report.txt"),
        entrypoints=("src/optimization/quantize_arcface_model.py", "src/optimization/export_arcface_onnx.py"),
    ),
    ModuleResult(
        key="stargan",
        title="Face attribute editing",
        stage="Generative editing",
        status="complete",
        summary="StarGAN was trained on aligned CelebA for hair color, gender, and age-related editing, then refined to 30 epochs.",
        metrics=("Training images: 100,000", "FID: 66.40", "Inception Score: 2.178"),
        media=("outputs/stargan/images/stargan_celeba_attr5_refine/epoch_030.jpg", "outputs/stargan/curves/stargan_celeba_attr5_refine_quality_summary.jpg"),
        reports=("outputs/reports/stargan_celeba_attr5_refine_result.txt", "outputs/reports/stargan_celeba_attr5_refine_quality_eval.txt"),
        entrypoints=("src/stargan/train_stargan_celeba.py", "src/stargan/sample_stargan_celeba.py", "src/stargan/evaluate_stargan_quality.py"),
    ),
    ModuleResult(
        key="reconstruction",
        title="3D face reconstruction",
        stage="3D geometry",
        status="complete",
        summary="Official 3DDFA_V2 dense reconstruction was integrated and the resulting mesh was rendered from six views with OpenGL.",
        metrics=("Vertices: 38,365", "Triangles: 76,073", "OpenGL views: 6"),
        media=(
            "outputs/3d_reconstruction/3ddfa_v2/images/indoor_016_original_3ddfa_v2_projected_mesh.jpg",
            "outputs/3d_reconstruction/3ddfa_v2/opengl/indoor_016_original_3ddfa_v2_opengl_multiview_grid.jpg",
        ),
        reports=("outputs/reports/face_3d_reconstruction_3ddfa_v2_result.txt", "outputs/reports/face_3d_reconstruction_opengl_render_result.txt"),
        entrypoints=("src/reconstruction/run_3ddfa_v2_reconstruction.py", "src/reconstruction/render_3ddfa_mesh_opengl.py"),
    ),
    ModuleResult(
        key="effects",
        title="Dynamic face effects",
        stage="Real-time effects",
        status="complete",
        summary="Dense landmarks drive animated glasses, a hat, beauty smoothing, lipstick, and blush in a reproducible video demo.",
        metrics=("Output: 96 frames @ 24 FPS", "Processing throughput: 6.66 FPS"),
        media=(
            "outputs/effects/images/indoor_016_effects_grid.jpg",
            "outputs/effects/videos/dynamic_face_effects_contact_sheet.jpg",
            "outputs/effects/videos/dynamic_face_effects_demo.mp4",
        ),
        reports=("outputs/reports/effects_baseline_result.txt", "outputs/reports/dynamic_effects_demo_result.txt"),
        entrypoints=("src/effects/run_effect_demo.py", "src/effects/run_dynamic_effects_demo.py"),
    ),
)


def relative_href(target: str, output_dir: Path) -> str:
    return Path(os.path.relpath(PROJECT_ROOT / target, output_dir)).as_posix()


def media_markup(paths: Iterable[str], output_dir: Path) -> str:
    items = []
    for item in paths:
        path = PROJECT_ROOT / item
        if not path.exists():
            continue
        href = html.escape(relative_href(item, output_dir), quote=True)
        label = html.escape(path.name)
        if path.suffix.lower() in {".mp4", ".webm", ".mov"}:
            items.append(f'<figure><video controls preload="metadata" src="{href}"></video><figcaption>{label}</figcaption></figure>')
        else:
            items.append(f'<figure><a href="{href}"><img loading="lazy" src="{href}" alt="{label}"></a><figcaption>{label}</figcaption></figure>')
    return "".join(items) or '<p class="muted">No preview artifact is available.</p>'


def link_list(paths: Iterable[str], output_dir: Path) -> str:
    links = []
    for item in paths:
        exists = (PROJECT_ROOT / item).exists()
        css = "" if exists else ' class="missing"'
        href = html.escape(relative_href(item, output_dir), quote=True)
        links.append(f'<li><a{css} href="{href}">{html.escape(item)}</a></li>')
    return "".join(links)


def build_html(output_dir: Path) -> str:
    complete = sum(module.status == "complete" for module in MODULES)
    partial = len(MODULES) - complete
    sections = []
    for module in MODULES:
        status_label = "Complete" if module.status == "complete" else "Partial"
        metrics = "".join(f"<li>{html.escape(metric)}</li>" for metric in module.metrics)
        sections.append(
            f"""<section class="module" id="{html.escape(module.key)}">
<div class="module-head"><div><p class="stage">{html.escape(module.stage)}</p><h2>{html.escape(module.title)}</h2></div>
<span class="status {html.escape(module.status)}">{status_label}</span></div>
<p>{html.escape(module.summary)}</p><ul class="metrics">{metrics}</ul>
<div class="media">{media_markup(module.media, output_dir)}</div>
<details><summary>Evidence and entry points</summary><div class="evidence-grid">
<div><h3>Reports</h3><ul>{link_list(module.reports, output_dir)}</ul></div>
<div><h3>Code</h3><ul>{link_list(module.entrypoints, output_dir)}</ul></div>
</div></details></section>"""
        )
    nav = "".join(f'<a href="#{html.escape(module.key)}">{html.escape(module.title)}</a>' for module in MODULES)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face AI Project - Final Demo</title>
<style>
:root {{ color-scheme: light; --ink:#17212b; --muted:#607080; --line:#d6dde3; --panel:#f7f9fa; --green:#18794e; --amber:#8a5a00; --accent:#006d77; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:#fff; font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
header {{ border-bottom:1px solid var(--line); background:var(--panel); }} .shell {{ width:min(1180px,calc(100% - 32px)); margin:0 auto; }}
.intro {{ padding:36px 0 28px; }} h1 {{ margin:0 0 8px; font-size:32px; letter-spacing:0; }} h2 {{ margin:0; font-size:21px; letter-spacing:0; }}
h3 {{ margin:0 0 8px; font-size:14px; letter-spacing:0; }} p {{ margin:8px 0 14px; }} .summary {{ display:flex; gap:24px; color:var(--muted); }}
nav {{ display:flex; gap:6px; overflow-x:auto; padding:10px 0; border-top:1px solid var(--line); }} nav a {{ padding:7px 10px; color:var(--ink); text-decoration:none; white-space:nowrap; }}
nav a:hover {{ color:var(--accent); }} main {{ padding:28px 0 48px; }} .module {{ padding:24px 0 30px; border-bottom:1px solid var(--line); }}
.module-head {{ display:flex; align-items:start; justify-content:space-between; gap:16px; }} .stage {{ margin:0 0 2px; color:var(--muted); font-size:12px; font-weight:700; text-transform:uppercase; }}
.status {{ min-width:72px; padding:4px 8px; border:1px solid currentColor; border-radius:4px; text-align:center; font-size:12px; font-weight:700; }}
.status.complete {{ color:var(--green); }} .status.partial {{ color:var(--amber); }} .metrics {{ display:flex; flex-wrap:wrap; gap:8px 24px; margin:0 0 18px; padding-left:20px; font-weight:600; }}
.media {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; align-items:start; }} figure {{ margin:0; }}
img,video {{ display:block; width:100%; max-height:520px; object-fit:contain; background:#eef2f4; border:1px solid var(--line); }}
figcaption {{ padding-top:5px; color:var(--muted); font-size:12px; }} details {{ margin-top:16px; }} summary {{ cursor:pointer; font-weight:700; }}
.evidence-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:20px; padding:14px 0 0; }} .evidence-grid ul {{ margin:0; padding-left:18px; overflow-wrap:anywhere; }}
a {{ color:var(--accent); }} a.missing {{ color:#a33; text-decoration-style:dotted; }} .muted {{ color:var(--muted); }} footer {{ padding:22px 0; border-top:1px solid var(--line); color:var(--muted); }}
@media (max-width:680px) {{ .summary,.evidence-grid {{ display:block; }} .summary span {{ display:block; }} .media {{ grid-template-columns:1fr; }} h1 {{ font-size:26px; }} }}
</style></head><body><header><div class="shell"><div class="intro"><h1>Face AI Project - Final Demo</h1>
<p class="muted">A single evidence dashboard for face analysis, recognition, generation, reconstruction, and effects.</p>
<div class="summary"><span><strong>{len(MODULES)}</strong> integrated modules</span><span><strong>{complete}</strong> complete</span><span><strong>{partial}</strong> partial</span></div>
</div><nav aria-label="Module navigation">{nav}</nav></div></header><main class="shell">{''.join(sections)}</main>
<footer><div class="shell">Generated from local experiment artifacts. Missing links are shown in red.</div></footer></body></html>"""


def generate(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.html"
    manifest_path = output_dir / "manifest.json"
    index_path.write_text(build_html(output_dir), encoding="utf-8")
    manifest = {
        "project": PROJECT_ROOT.name,
        "modules": [
            {
                **asdict(module),
                "missing_media": [path for path in module.media if not (PROJECT_ROOT / path).exists()],
                "missing_reports": [path for path in module.reports if not (PROJECT_ROOT / path).exists()],
                "missing_entrypoints": [path for path in module.entrypoints if not (PROJECT_ROOT / path).exists()],
            }
            for module in MODULES
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return index_path, manifest_path


def serve(port: int, open_browser: bool) -> None:
    os.chdir(PROJECT_ROOT)
    url = f"http://127.0.0.1:{port}/outputs/final_demo/index.html"
    if open_browser:
        webbrowser.open(url)
    print(f"Final demo: {url}")
    print("Press Ctrl+C to stop.")
    ThreadingHTTPServer(("127.0.0.1", port), SimpleHTTPRequestHandler).serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and optionally serve the unified final project demo.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--open", action="store_true", dest="open_browser")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    index_path, manifest_path = generate(output_dir)
    print(f"Generated: {index_path}")
    print(f"Manifest: {manifest_path}")
    if args.serve:
        serve(args.port, args.open_browser)


if __name__ == "__main__":
    main()
