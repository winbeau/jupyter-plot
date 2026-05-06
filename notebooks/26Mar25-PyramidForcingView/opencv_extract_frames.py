from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import ceil, sqrt
from multiprocessing import get_all_start_methods, get_context
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import DisplayHandle, display
from tqdm.auto import tqdm


VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm")
FRAME_FILENAME_TEMPLATE = "frame_{frame_idx:06d}.png"
SHEET_ROWS = 6
SHEET_COLS = 6
SHEET_DPI = 200
SHEET_FIGSIZE = (14, 14)
SHEET_FACE_COLOR = "white"
SHEET_OVERVIEW_MAX_COLS = 4
SUMMARY_FILENAME = "frame_export_summary.csv"
DEFAULT_MAX_WORKERS = max(1, min(8, os.cpu_count() or 1))


def resolve_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository root from the current working directory.")


def resolve_path(path_str: str) -> Path:
    raw = Path(path_str)
    return raw if raw.is_absolute() else resolve_repo_root() / raw


def list_local_videos(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    videos = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not videos:
        raise FileNotFoundError(f"No local videos found in: {input_path}")
    return videos


def select_video_paths(input_path: Path, extract_all_videos: bool, target_video_name: str) -> list[Path]:
    videos = list_local_videos(input_path)
    if extract_all_videos:
        return videos

    target_path = input_path / target_video_name
    if target_path not in videos:
        raise FileNotFoundError(f"Could not find target video in {input_path}: {target_video_name}")
    return [target_path]


def normalize_video_id(video_path: Path, index: int) -> str:
    group_name = video_path.parent.name
    if video_path.stem.startswith("video_"):
        return f"{group_name}_{video_path.stem}"
    return f"{group_name}_video_{index:03d}"


def build_video_jobs(video_paths: list[Path], output_path: Path, frame_stride: int) -> list[dict]:
    jobs = []
    for index, video_path in enumerate(video_paths, start=1):
        jobs.append(
            {
                "index": index,
                "video_name": video_path.name,
                "video_path": str(video_path),
                "normalized_video_id": normalize_video_id(video_path, index),
                "output_dir": str(output_path),
                "frame_stride": frame_stride,
            }
        )
    return jobs


def chunked(items: list[Path], chunk_size: int) -> list[list[Path]]:
    return [items[idx : idx + chunk_size] for idx in range(0, len(items), chunk_size)]


def render_image_grid(image_paths: list[Path], output_path: Path, rows: int, cols: int, title: str) -> None:
    fig, axes = plt.subplots(rows, cols, figsize=SHEET_FIGSIZE, dpi=SHEET_DPI)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    fig.patch.set_facecolor(SHEET_FACE_COLOR)
    fig.suptitle(title, fontsize=14)

    for axis, image_path in zip(axes, image_paths):
        axis.imshow(plt.imread(image_path))
        axis.axis("off")

    for axis in axes[len(image_paths) :]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=SHEET_DPI, facecolor=SHEET_FACE_COLOR, bbox_inches="tight")
    plt.close(fig)


def render_sheet_pages(
    normalized_video_id: str,
    frame_paths: list[Path],
    video_dir: Path,
    page_callback=None,
) -> list[Path]:
    pages = chunked(frame_paths, SHEET_ROWS * SHEET_COLS)
    sheet_paths = []
    for page_index, page_frame_paths in enumerate(pages, start=1):
        sheet_path = video_dir / f"sheet_{page_index:03d}.png"
        render_image_grid(
            page_frame_paths,
            sheet_path,
            rows=SHEET_ROWS,
            cols=SHEET_COLS,
            title=f"{normalized_video_id} | page {page_index}/{len(pages)}",
        )
        sheet_paths.append(sheet_path)
        if page_callback is not None:
            page_callback(page_index, len(pages), sheet_path)
    return sheet_paths


def render_sheet_overview(normalized_video_id: str, sheet_paths: list[Path], video_dir: Path) -> Path:
    overview_path = video_dir / "sheet_overview.png"
    if len(sheet_paths) == 1:
        shutil.copy2(sheet_paths[0], overview_path)
        return overview_path

    overview_cols = min(SHEET_OVERVIEW_MAX_COLS, max(1, ceil(sqrt(len(sheet_paths)))))
    overview_rows = ceil(len(sheet_paths) / overview_cols)
    render_image_grid(
        sheet_paths,
        overview_path,
        rows=overview_rows,
        cols=overview_cols,
        title=f"{normalized_video_id} | sheet overview",
    )
    return overview_path


def export_video_frames_job(job: dict) -> dict:
    video_path = Path(job["video_path"])
    output_dir = Path(job["output_dir"])
    normalized_video_id = job["normalized_video_id"]
    frame_stride = int(job["frame_stride"])

    video_dir = output_dir / normalized_video_id
    frames_dir = video_dir / "frames"
    video_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video with OpenCV: {video_path}")

    decoded_frame_count = 0
    frame_paths = []
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        decoded_frame_count += 1
        if (decoded_frame_count - 1) % frame_stride != 0:
            continue

        frame_output_path = frames_dir / FRAME_FILENAME_TEMPLATE.format(frame_idx=decoded_frame_count)
        if not cv2.imwrite(str(frame_output_path), frame_bgr):
            capture.release()
            raise RuntimeError(f"OpenCV failed to write frame PNG: {frame_output_path}")
        frame_paths.append(frame_output_path)

    capture.release()
    if not frame_paths:
        raise RuntimeError(f"No frames were exported for {video_path.name}")

    sheet_paths = render_sheet_pages(normalized_video_id, frame_paths, video_dir)
    overview_path = render_sheet_overview(normalized_video_id, sheet_paths, video_dir)
    return {
        "normalized_video_id": normalized_video_id,
        "video_name": job["video_name"],
        "video_path": str(video_path),
        "decoded_frame_count": decoded_frame_count,
        "exported_frame_count": len(frame_paths),
        "frame_stride": frame_stride,
        "frame_dir": str(frames_dir),
        "sheet_count": len(sheet_paths),
        "overview_path": str(overview_path),
        "status": "ok",
    }


def make_failed_result(job: dict, exc: Exception) -> dict:
    normalized_video_id = job["normalized_video_id"]
    video_dir = Path(job["output_dir"]) / normalized_video_id
    return {
        "normalized_video_id": normalized_video_id,
        "video_name": job["video_name"],
        "video_path": job["video_path"],
        "decoded_frame_count": 0,
        "exported_frame_count": 0,
        "frame_stride": job["frame_stride"],
        "frame_dir": str(video_dir / "frames"),
        "sheet_count": 0,
        "overview_path": str(video_dir / "sheet_overview.png"),
        "status": f"failed: {exc}",
    }


def build_progress_df(video_jobs: list[dict], results_by_id: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for job in video_jobs:
        result = results_by_id.get(job["normalized_video_id"])
        if result is None:
            rows.append(
                {
                    "video_name": job["video_name"],
                    "normalized_video_id": job["normalized_video_id"],
                    "status": "pending",
                    "frames": 0,
                    "sheets": 0,
                }
            )
        else:
            rows.append(
                {
                    "video_name": result["video_name"],
                    "normalized_video_id": result["normalized_video_id"],
                    "status": result["status"],
                    "frames": result["exported_frame_count"],
                    "sheets": result["sheet_count"],
                }
            )
    return pd.DataFrame(rows)


def build_single_video_progress_df(
    job: dict,
    stage: str,
    decoded_frame_count: int,
    exported_frame_count: int,
    sheet_pages_done: int,
    total_sheet_pages: int,
    last_output_path: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "video_name": job["video_name"],
                "normalized_video_id": job["normalized_video_id"],
                "stage": stage,
                "decoded_frames": decoded_frame_count,
                "saved_frames": exported_frame_count,
                "sheet_pages": f"{sheet_pages_done}/{total_sheet_pages}",
                "last_output": last_output_path,
            }
        ]
    )


def export_single_video_with_visualization(job: dict) -> dict:
    video_path = Path(job["video_path"])
    output_dir = Path(job["output_dir"])
    normalized_video_id = job["normalized_video_id"]
    frame_stride = int(job["frame_stride"])
    video_dir = output_dir / normalized_video_id
    frames_dir = video_dir / "frames"
    video_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video with OpenCV: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    decoded_frame_count = 0
    exported_frame_count = 0
    frame_paths = []
    last_output_path = ""
    progress_handle = DisplayHandle()
    progress_handle.display(
        build_single_video_progress_df(
            job,
            stage="saving_frames",
            decoded_frame_count=0,
            exported_frame_count=0,
            sheet_pages_done=0,
            total_sheet_pages=0,
            last_output_path="",
        )
    )

    with tqdm(
        total=total_frames or None,
        desc=f"Frames | {job['video_name']}",
        unit="frame",
    ) as frame_bar:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            decoded_frame_count += 1
            frame_bar.update(1)
            if (decoded_frame_count - 1) % frame_stride != 0:
                continue

            frame_output_path = frames_dir / FRAME_FILENAME_TEMPLATE.format(frame_idx=decoded_frame_count)
            if not cv2.imwrite(str(frame_output_path), frame_bgr):
                capture.release()
                raise RuntimeError(f"OpenCV failed to write frame PNG: {frame_output_path}")

            exported_frame_count += 1
            frame_paths.append(frame_output_path)
            last_output_path = str(frame_output_path)
            if exported_frame_count == 1 or exported_frame_count % 25 == 0:
                progress_handle.update(
                    build_single_video_progress_df(
                        job,
                        stage="saving_frames",
                        decoded_frame_count=decoded_frame_count,
                        exported_frame_count=exported_frame_count,
                        sheet_pages_done=0,
                        total_sheet_pages=0,
                        last_output_path=last_output_path,
                    )
                )

    capture.release()
    if not frame_paths:
        raise RuntimeError(f"No frames were exported for {video_path.name}")

    total_sheet_pages = ceil(len(frame_paths) / (SHEET_ROWS * SHEET_COLS))
    progress_handle.update(
        build_single_video_progress_df(
            job,
            stage="rendering_sheets",
            decoded_frame_count=decoded_frame_count,
            exported_frame_count=exported_frame_count,
            sheet_pages_done=0,
            total_sheet_pages=total_sheet_pages,
            last_output_path=last_output_path,
        )
    )

    with tqdm(
        total=total_sheet_pages + 1,
        desc=f"Sheets | {job['video_name']}",
        unit="sheet",
    ) as sheet_bar:

        def on_sheet_page(page_index: int, total_pages: int, sheet_path: Path) -> None:
            sheet_bar.update(1)
            progress_handle.update(
                build_single_video_progress_df(
                    job,
                    stage="rendering_sheets",
                    decoded_frame_count=decoded_frame_count,
                    exported_frame_count=exported_frame_count,
                    sheet_pages_done=page_index,
                    total_sheet_pages=total_pages,
                    last_output_path=str(sheet_path),
                )
            )

        sheet_paths = render_sheet_pages(
            normalized_video_id,
            frame_paths,
            video_dir,
            page_callback=on_sheet_page,
        )
        overview_path = render_sheet_overview(normalized_video_id, sheet_paths, video_dir)
        sheet_bar.update(1)

    progress_handle.update(
        build_single_video_progress_df(
            job,
            stage="done",
            decoded_frame_count=decoded_frame_count,
            exported_frame_count=exported_frame_count,
            sheet_pages_done=total_sheet_pages,
            total_sheet_pages=total_sheet_pages,
            last_output_path=str(overview_path),
        )
    )

    return {
        "normalized_video_id": normalized_video_id,
        "video_name": job["video_name"],
        "video_path": str(video_path),
        "decoded_frame_count": decoded_frame_count,
        "exported_frame_count": exported_frame_count,
        "frame_stride": frame_stride,
        "frame_dir": str(frames_dir),
        "sheet_count": len(sheet_paths),
        "overview_path": str(overview_path),
        "status": "ok",
    }


def extract_frames(
    input_dir: str,
    output_dir: str,
    frame_stride: int,
    extract_all_videos: bool,
    target_video_name: str,
    max_workers: int | None = None,
) -> pd.DataFrame:
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    input_path = resolve_path(input_dir)
    output_path = resolve_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / SUMMARY_FILENAME
    max_workers = DEFAULT_MAX_WORKERS if max_workers is None else max(1, int(max_workers))

    display(
        {
            "input_dir": str(input_path),
            "output_dir": str(output_path),
            "frame_stride": frame_stride,
            "extract_all_videos": extract_all_videos,
            "target_video_name": target_video_name,
            "max_workers": max_workers,
        }
    )

    video_paths = select_video_paths(input_path, extract_all_videos, target_video_name)
    video_jobs = build_video_jobs(video_paths, output_path, frame_stride)
    worker_count = 1 if len(video_jobs) <= 1 else min(max_workers, len(video_jobs))
    single_video_mode = (not extract_all_videos) and len(video_jobs) == 1

    display(
        {
            "mode": "all_videos" if extract_all_videos else "single_video",
            "selected_video_count": len(video_jobs),
            "worker_count": worker_count,
            "summary_path": str(summary_path),
        }
    )

    results = []
    results_by_id = {}

    if single_video_mode:
        result = export_single_video_with_visualization(video_jobs[0])
        results.append(result)
        results_by_id[result["normalized_video_id"]] = result
    else:
        progress_handle = DisplayHandle()
        progress_handle.display(build_progress_df(video_jobs, results_by_id))

        with tqdm(total=len(video_jobs), desc="Extracting videos", unit="video") as progress_bar:
            if worker_count == 1:
                for job in video_jobs:
                    try:
                        result = export_video_frames_job(job)
                    except Exception as exc:
                        result = make_failed_result(job, exc)
                    results.append(result)
                    results_by_id[result["normalized_video_id"]] = result
                    progress_handle.update(build_progress_df(video_jobs, results_by_id))
                    progress_bar.update(1)
            else:
                start_methods = get_all_start_methods()
                executor_kwargs = {"max_workers": worker_count}
                if "fork" in start_methods:
                    executor_kwargs["mp_context"] = get_context("fork")

                with ProcessPoolExecutor(**executor_kwargs) as executor:
                    future_to_job = {executor.submit(export_video_frames_job, job): job for job in video_jobs}
                    for future in as_completed(future_to_job):
                        job = future_to_job[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = make_failed_result(job, exc)
                        results.append(result)
                        results_by_id[result["normalized_video_id"]] = result
                        progress_handle.update(build_progress_df(video_jobs, results_by_id))
                        progress_bar.update(1)

    summary_df = pd.DataFrame(results).sort_values(["normalized_video_id"], kind="stable").reset_index(drop=True)
    summary_df.to_csv(summary_path, index=False)
    display(summary_df)

    failed_df = summary_df[summary_df["status"].str.startswith("failed:")].reset_index(drop=True)
    if not failed_df.empty:
        display(failed_df)
        failed_ids = ", ".join(failed_df["normalized_video_id"].tolist())
        raise RuntimeError(f"Frame export failed for: {failed_ids}")

    print(f"Saved frame export summary to {summary_path}")
    return summary_df
