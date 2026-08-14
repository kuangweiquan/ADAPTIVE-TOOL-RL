"""Phase 0: extract single-image MCQA subset from AdaTooler-V-300k RL data.

Runs CPU-only on the remote box, memory-safe under the 2GB cgroup RAM cap of
no-GPU mode:
  - parses adatooler_v_rl.json INCREMENTALLY with ijson (no full-file json.load)
  - reservoir-samples the filtered rows in one pass
  - fetches images from the HF repo: image files are stored inside category
    zip archives (e.g. General_Image/General_Image.zip), so we read them via
    HTTP Range requests with remotezip -- only the needed bytes are downloaded
  - joins delta_s from adatooler_v_rl_with_deltaS.json by (problem, path)

Official prepare_train.py hardcodes /home/wangcy paths and assumes the dataset
is fully materialized; this variant works from raw JSON + on-demand images.

Usage (remote, atr env, network_turbo sourced):
  python prepare_subset.py \
    --rl_json /root/autodl-tmp/datasets/AdaTooler-V-300k/adatooler_v_rl.json \
    --deltaS_json /root/autodl-tmp/datasets/AdaTooler-V-300k/adatooler_v_rl_with_deltaS.json \
    --out_dir /root/autodl-tmp/datasets/adatooler_v_subset --n_samples 3000
  # smoke first:
  python prepare_subset.py ... --n_samples 10 --out_dir /tmp/subset_smoke
"""
import argparse
import json
import random
import re
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

import ijson
from remotezip import RemoteZip

# --- prompt constants copied verbatim from their prepare_train.py (fidelity) ---
TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
}

system_prompt = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "crop_image", "description": "Zoom in on the image based on the bounding box coordinates.", "parameters": {"type": "object", "properties": {"bbox_2d": {"type": "array", "description": "coordinates for bounding box of the area you want to zoom in. minimum value is 0 and maximum value is the width/height of the image.", "items": {"type": "number"}}, "target_image": {"type": "number", "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."}}, "required": ["bbox_2d", "target_image"]}}}
{"type": "function", "function": {"name": "select_frames", "description": "Select frames from a video.", "parameters": {"type": "object", "properties": {"target_frames": {"type": "array", "description": "List of frame indices to select from the video (no more than 8 frames in total).", "items": {"type": "integer", "description": "Frame index from 1 to 16."}}}, "required": ["target_frames"]}}}
{"type": "function", "function": {"name": "PathTracer", "description": "Plot movement or connections between two points on the specified image.", "parameters": {"type": "object", "properties": {"target_image": {"type": "number", "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."}, "start_point_2d": {"type": "array", "description": "Starting point coordinates [x1, y1] of the path. minimum value is 0 and maximum value is the width/height of the image.", "items": {"type": "number"}}, "end_point_2d": {"type": "array", "description": "Ending point coordinates [x2, y2] of the path. minimum value is 0 and maximum value is the width/height of the image.", "items": {"type": "number"}}}, "required": ["start_point_2d", "end_point_2d", "target_image"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

guideline = """Guidelines: Understand the given visual information and the user query.

Determine if it is beneficial to employ the given visual operations (tools).

Determine which tool to use based on the input:
- For a single image, use `crop_image` or `PathTracer`.
- For a video, use `select_frames`, `crop_image`, or `PathTracer`.

Reason with the visual information step by step.
You should:
1. Explain why a tool is necessary.
2. Call the tool.
3. Continue reasoning based on the tool output.
4. Provide the final answer.

Place your text reasoning process within the <think> </think> tags.
Place any function calls within the <tool_call></tool_call> tags.
Place your final answer within the <answer> </answer> tags.
"""

IMAGE_SEP = "<image>"
REPO_BASE = "https://huggingface.co/datasets/AdaTooler-V/AdaTooler-V-300k/resolve/main/"
API_BASE = "https://huggingface.co/api/datasets/AdaTooler-V/AdaTooler-V-300k/tree/main/"
ZIP_PREFIX = "home/sig95vg/remote_data/wangcy/"  # prefix inside the category zips

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)


def norm_paths(path):
    """path may be a str or a list of str (multi-image rows)."""
    if isinstance(path, str):
        return (path.lstrip("./"),)
    return tuple(p.lstrip("./") for p in path)


def norm_key(problem, path):
    return (problem, norm_paths(path))


def strip_answer(solution: str) -> str:
    m = ANSWER_RE.search(solution or "")
    return m.group(1).strip() if m else (solution or "").strip()


def build_question(problem: str, options) -> str:
    q = problem
    if options:
        q += "\n" + "\n".join(f"{o}" for o in options)
    return q


class ImageFetcher:
    """Fetch dataset images stored inside HF repo zip archives.

    Fast path: category zip downloaded locally (zip_cache_dir) -> plain zipfile.
    Slow path: HTTP Range reads via remotezip.
    """

    def __init__(self, zip_cache_dir: Path = None):
        self._cat_zips = {}   # category -> [zip repo paths]
        self._zips = {}       # zip repo path -> RemoteZip | False
        self._local = {}      # zip repo path -> zipfile.ZipFile
        self._cache_dir = zip_cache_dir

    def _cat_zip_paths(self, category: str):
        if category in self._cat_zips:
            return self._cat_zips[category]
        zips = []
        try:
            with urllib.request.urlopen(API_BASE + quote(category, safe="/"), timeout=60) as r:
                tree = json.load(r)
            zips = [x["path"] for x in tree if x.get("path", "").endswith(".zip")]
        except Exception as e:
            print(f"  tree API failed for {category}: {type(e).__name__}")
        self._cat_zips[category] = zips
        return zips

    def _open_local(self, zip_path: str):
        if not self._cache_dir:
            return None
        if zip_path in self._local:
            return self._local[zip_path]
        import zipfile
        local_file = self._cache_dir / zip_path.split("/")[-1]
        zf = None
        if local_file.exists():
            try:
                zf = zipfile.ZipFile(local_file)
            except Exception as e:
                print(f"  local zip open failed {local_file}: {type(e).__name__}", flush=True)
        self._local[zip_path] = zf or False
        return zf

    def _open_zip(self, zip_path: str):
        if zip_path in self._zips:
            return self._zips[zip_path] or None
        rz = None
        try:
            rz = RemoteZip(REPO_BASE + quote(zip_path, safe="/"))
        except Exception as e:
            print(f"  RemoteZip open failed for {zip_path}: {type(e).__name__}")
        self._zips[zip_path] = rz or False
        return rz

    def fetch(self, json_path: str, out_path: Path, attempts: int = 2) -> bool:
        rel = json_path.lstrip("./")
        category = rel.split("/")[0]
        for zip_path in self._cat_zip_paths(category):
            zf_local = self._open_local(zip_path)
            if zf_local is not None:
                for candidate in (ZIP_PREFIX + rel, rel):
                    try:
                        data = zf_local.read(candidate)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(out_path, "wb") as f:
                            f.write(data)
                        return True
                    except KeyError:
                        continue
                    except Exception as e:
                        print(f"  local read failed {candidate}: {type(e).__name__}", flush=True)
                        return False
            rz = self._open_zip(zip_path)
            if rz is None:
                continue
            for candidate in (ZIP_PREFIX + rel, rel):
                for attempt in range(attempts):
                    try:
                        if candidate in rz.namelist():
                            data = rz.read(candidate)
                            if not data:
                                break
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(out_path, "wb") as f:
                                f.write(data)
                            return True
                    except Exception as e:
                        if attempt == attempts - 1:
                            print(f"  read failed {candidate}: {type(e).__name__}", flush=True)
                        time.sleep(1 + attempt)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl_json", type=str, required=True)
    ap.add_argument("--deltaS_json", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_samples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--problem_types", nargs="+", default=["multiple choice"])
    ap.add_argument("--categories", nargs="+", default=None,
                    help="restrict to these top-level category dirs (e.g. Spatial_Image General_Image)")
    ap.add_argument("--zip_cache_dir", type=str, default=None,
                    help="dir with locally downloaded category zips (plain zipfile, fast path)")
    ap.add_argument("--val_size", type=int, default=100)
    ap.add_argument("--max_candidates", type=int, default=0,
                    help="cap candidate scan (0 = unlimited)")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # delta_s content-key index (15k entries, small).
    deltaS = {}
    if args.deltaS_json:
        with open(args.deltaS_json) as f:
            for r in json.load(f):
                deltaS[norm_key(r.get("problem"), r.get("path"))] = r.get("delta_s")
        print(f"delta_s index: {len(deltaS)} entries", flush=True)

    # Pass 1: ijson stream + reservoir sample metadata rows (no image fetch).
    reservoir, n_candidates = [], 0
    with open(args.rl_json, "rb") as f:
        for r in ijson.items(f, "item"):
            if r.get("data_type") != "image" or r.get("problem_type") not in args.problem_types:
                continue
            if args.categories:
                cat = norm_paths(r.get("path"))[0].split("/")[0]
                if cat not in args.categories:
                    continue
            n_candidates += 1
            if args.max_candidates and n_candidates > args.max_candidates:
                break
            if len(reservoir) < args.n_samples:
                reservoir.append(r)
            else:
                j = rng.randrange(n_candidates)
                if j < args.n_samples:
                    reservoir[j] = r
    print(f"candidates (image, {args.problem_types}): {n_candidates}; kept {len(reservoir)}", flush=True)

    out = Path(args.out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    fetcher = ImageFetcher(Path(args.zip_cache_dir) if args.zip_cache_dir else None)
    # Warm zip listings for every category touched (central dir fetched once).
    cats = sorted({rel.split("/")[0] for r in reservoir for rel in norm_paths(r["path"])})
    for c in cats:
        fetcher._cat_zip_paths(c)
    print(f"categories: {cats}", flush=True)

    def fetch_row(r, i):
        qid = str(r.get("problem_id", 0))
        rel_paths = norm_paths(r["path"])
        abs_paths, ok_all = [], True
        for j, rel in enumerate(rel_paths):
            ext = Path(rel).suffix.lower() or ".jpg"
            # row index keeps filenames unique even when source problem_id duplicates
            img_name = f"{qid}_{i}_{j}{ext}" if len(rel_paths) > 1 else f"{qid}_{i}{ext}"
            if fetcher.fetch(rel, img_dir / img_name):
                abs_paths.append((img_dir / img_name).absolute().as_posix())
            else:
                ok_all = False
                print(f"  FAILED image {rel} -> skipping row", flush=True)
                break
        if not ok_all:
            return None
        question_raw = build_question(r["problem"], r.get("options")) + f"\n\n{guideline}" + TYPE_TEMPLATE[r["problem_type"]]
        return {
            "data_source": "AdaTooler-V/AdaTooler-V-300k",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": IMAGE_SEP * len(abs_paths) + question_raw},
            ],
            "images": [{"image": p} for p in abs_paths],
            "ability": "visual_reasoning",
            "reward_model": {"style": "rule", "ground_truth": strip_answer(r["solution"])},
            "extra_info": {
                "split": "train",
                "index": r.get("problem_id", 0),
                "qid": qid,
                "is_video": False,
                "images": abs_paths,
                "problem_type": r["problem_type"],
                "delta_s": deltaS.get(norm_key(r.get("problem"), r.get("path"))),
                "data_source": r.get("data_source"),
            },
        }

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = [x for x in ex.map(lambda t: fetch_row(t[1], t[0]), enumerate(reservoir)) if x is not None]

    fetched = sum(len(r["extra_info"]["images"]) for r in rows)
    failed = len(reservoir) - len(rows)

    print(f"fetched {fetched} images, failed {failed}, rows {len(rows)}", flush=True)
    train_rows, val_rows = rows[:-args.val_size], rows[-args.val_size:] if args.val_size else (rows, [])

    import pyarrow as pa
    import pyarrow.parquet as pq
    for name, r in (("train.parquet", train_rows), ("val.parquet", val_rows)):
        if not r:
            continue
        pq.write_table(pa.Table.from_pylist(r), out / name)
        print(f"wrote {out / name}: {len(r)} rows", flush=True)

    with open(out / "subset_stats.json", "w") as f:
        json.dump({"n_candidates": n_candidates, "n_kept": len(rows),
                   "n_val": len(val_rows), "n_failed_images": failed,
                   "seed": args.seed, "problem_types": args.problem_types,
                   "delta_s_hits": sum(1 for x in rows if x["extra_info"]["delta_s"] is not None)},
                  f, indent=2)
    print(f"stats -> {out / 'subset_stats.json'}", flush=True)


if __name__ == "__main__":
    main()
