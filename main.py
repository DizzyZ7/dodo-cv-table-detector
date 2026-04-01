"""
main.py
-------

Скрипт детектирует состояния одного столика по видео и сохраняет:
- видео с визуализацией;
- CSV с событиями;
- CSV с сегментами состояний;
- JSON со сводной статистикой;
- текстовый отчет.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class DetectionConfig:
    video_path: Path
    output_path: Path
    roi: Roi
    bg_frames: int
    threshold: float
    min_frames_for_change: int
    binary_threshold: int
    blur_size: int
    background_update_alpha: float
    progress_every: int


@dataclass(frozen=True)
class EventRecord:
    timestamp: float
    frame_index: int
    event: str


@dataclass(frozen=True)
class StateSegment:
    state: str
    start_time: float
    end_time: float
    duration: float
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class DetectionSummary:
    fps: float
    total_frames_processed: int
    video_duration_sec: float
    roi: dict
    events_detected: int
    approach_events: int
    empty_events: int
    occupied_segments: int
    occupied_time_sec: float
    empty_time_sec: float
    occupancy_ratio: float
    mean_reseat_delay_sec: Optional[float]
    median_reseat_delay_sec: Optional[float]
    min_reseat_delay_sec: Optional[float]
    max_reseat_delay_sec: Optional[float]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Детекция состояний одного столика по видео."
    )
    parser.add_argument("--video", required=True, help="Путь к входному видео")
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Координаты столика: x y w h",
    )
    parser.add_argument(
        "--select-roi",
        action="store_true",
        help="Выбрать столик мышкой на первом кадре",
    )
    parser.add_argument(
        "--output",
        default="output.mp4",
        help="Имя выходного видео",
    )
    parser.add_argument(
        "--bg-frames",
        type=int,
        default=30,
        help="Количество первых кадров для статического фона",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.04,
        help="Порог доли изменившихся пикселей для occupied",
    )
    parser.add_argument(
        "--min-frames-for-change",
        type=int,
        default=5,
        help="Количество кадров подряд для подтверждения смены состояния",
    )
    parser.add_argument(
        "--binary-threshold",
        type=int,
        default=25,
        help="Порог бинаризации разницы с фоном",
    )
    parser.add_argument(
        "--blur-size",
        type=int,
        default=5,
        help="Размер Gaussian blur для подавления шума",
    )
    parser.add_argument(
        "--background-update-alpha",
        type=float,
        default=0.02,
        help="Скорость адаптации фона во время empty; 0 отключает обновление",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=300,
        help="Печатать прогресс каждые N кадров; 0 отключает",
    )
    args = parser.parse_args()

    if args.roi is not None and args.select_roi:
        parser.error("Используй только один способ задания ROI: --roi или --select-roi.")
    if args.bg_frames <= 0:
        parser.error("--bg-frames должен быть больше 0.")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold должен быть в диапазоне [0, 1].")
    if args.min_frames_for_change <= 0:
        parser.error("--min-frames-for-change должен быть больше 0.")
    if not 0 <= args.binary_threshold <= 255:
        parser.error("--binary-threshold должен быть в диапазоне [0, 255].")
    if args.blur_size <= 0:
        parser.error("--blur-size должен быть больше 0.")
    if args.blur_size % 2 == 0:
        parser.error("--blur-size должен быть нечетным числом.")
    if not 0.0 <= args.background_update_alpha <= 1.0:
        parser.error("--background-update-alpha должен быть в диапазоне [0, 1].")
    if args.progress_every < 0:
        parser.error("--progress-every не может быть отрицательным.")

    return args


def select_roi_interactively(frame: np.ndarray) -> Roi:
    print("Откроется окно выбора столика.")
    print("Выдели столик мышкой и нажми Enter или Space.")
    print("Нажми C для сброса, Esc для отмены.")

    try:
        roi = cv2.selectROI("Select Table ROI", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
    except cv2.error as exc:
        raise RuntimeError(
            "Интерактивный выбор ROI недоступен в текущем окружении. Передай --roi вручную."
        ) from exc

    x, y, w, h = roi
    if w == 0 or h == 0:
        raise RuntimeError("ROI не выбран. Запусти снова и выдели столик.")
    return Roi(int(x), int(y), int(w), int(h))


def initialise_roi(frame: np.ndarray, roi_args: Optional[List[int]], select_roi_flag: bool) -> Roi:
    if roi_args is not None:
        x, y, w, h = roi_args
        return Roi(int(x), int(y), int(w), int(h))

    if select_roi_flag:
        return select_roi_interactively(frame)

    height, width = frame.shape[:2]
    w = width // 3
    h = height // 3
    x = (width - w) // 2
    y = (height - h) // 2
    print("ROI не передан, используется центральная область кадра.")
    return Roi(x, y, w, h)


def validate_roi(frame: np.ndarray, roi: Roi) -> Roi:
    height, width = frame.shape[:2]

    if roi.w <= 0 or roi.h <= 0:
        raise ValueError("Ширина и высота ROI должны быть больше 0.")
    if roi.x < 0 or roi.y < 0 or roi.x + roi.w > width or roi.y + roi.h > height:
        raise ValueError(
            f"ROI выходит за пределы кадра. Размер кадра: {width}x{height}, ROI: {roi}"
        )

    return roi


def build_background(
    cap: cv2.VideoCapture,
    roi: Roi,
    bg_frames_count: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    full_frames: List[np.ndarray] = []
    roi_frames: List[np.ndarray] = []

    for _ in range(bg_frames_count):
        ret, frame = cap.read()
        if not ret:
            break
        full_frames.append(frame.copy())
        roi_frame = frame[roi.y:roi.y + roi.h, roi.x:roi.x + roi.w]
        roi_frames.append(roi_frame.astype(np.float32))

    if not roi_frames:
        raise RuntimeError("Не удалось построить фон: видео слишком короткое или не читается.")

    background = np.median(np.stack(roi_frames, axis=0), axis=0).astype(np.float32)
    return background, full_frames


def finalize_segments(
    timeline_states: List[str],
    fps: float,
    start_frame_index: int,
) -> List[StateSegment]:
    if not timeline_states:
        return []

    segments: List[StateSegment] = []
    current_state = timeline_states[0]
    segment_start_idx = 0

    for idx, state in enumerate(timeline_states[1:], start=1):
        if state != current_state:
            start_frame = start_frame_index + segment_start_idx
            end_frame = start_frame_index + idx - 1
            start_time = start_frame / fps
            end_time = (end_frame + 1) / fps
            segments.append(
                StateSegment(
                    state=current_state,
                    start_time=round(start_time, 3),
                    end_time=round(end_time, 3),
                    duration=round(end_time - start_time, 3),
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )
            current_state = state
            segment_start_idx = idx

    start_frame = start_frame_index + segment_start_idx
    end_frame = start_frame_index + len(timeline_states) - 1
    start_time = start_frame / fps
    end_time = (end_frame + 1) / fps
    segments.append(
        StateSegment(
            state=current_state,
            start_time=round(start_time, 3),
            end_time=round(end_time, 3),
            duration=round(end_time - start_time, 3),
            start_frame=start_frame,
            end_frame=end_frame,
        )
    )
    return segments


def calculate_delay_stats(delays: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not delays:
        return None, None, None, None

    delay_array = np.array(delays, dtype=np.float32)
    return (
        float(np.mean(delay_array)),
        float(np.median(delay_array)),
        float(np.min(delay_array)),
        float(np.max(delay_array)),
    )


def detect_events(config: DetectionConfig) -> Tuple[pd.DataFrame, pd.DataFrame, DetectionSummary]:
    cap = cv2.VideoCapture(str(config.video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {config.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(config.output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Не удалось создать выходное видео: {config.output_path}")

    background = None
    initial_frames: List[np.ndarray] = []

    try:
        background, initial_frames = build_background(cap, config.roi, config.bg_frames)

        for frame in initial_frames:
            cv2.rectangle(
                frame,
                (config.roi.x, config.roi.y),
                (config.roi.x + config.roi.w, config.roi.y + config.roi.h),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "building background",
                (config.roi.x, max(20, config.roi.y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            out.write(frame)

        current_state = "empty"
        candidate_state = current_state
        candidate_count = 0
        frame_index = len(initial_frames)

        events: List[EventRecord] = []
        timeline_states: List[str] = []
        empty_timestamps: List[float] = []
        approach_timestamps: List[float] = []

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_index / fps
            roi_frame = frame[
                config.roi.y:config.roi.y + config.roi.h,
                config.roi.x:config.roi.x + config.roi.w,
            ]

            diff = cv2.absdiff(roi_frame.astype(np.float32), background.astype(np.float32))
            diff = diff.astype(np.uint8)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            if config.blur_size > 1:
                gray = cv2.GaussianBlur(gray, (config.blur_size, config.blur_size), 0)

            _, thresh = cv2.threshold(
                gray,
                config.binary_threshold,
                255,
                cv2.THRESH_BINARY,
            )
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            thresh = cv2.dilate(thresh, kernel, iterations=2)

            motion_ratio = float(cv2.countNonZero(thresh)) / float(config.roi.w * config.roi.h)
            instantaneous_state = "occupied" if motion_ratio > config.threshold else "empty"

            if instantaneous_state != current_state:
                if instantaneous_state == candidate_state:
                    candidate_count += 1
                else:
                    candidate_state = instantaneous_state
                    candidate_count = 1

                if candidate_count >= config.min_frames_for_change:
                    previous_state = current_state
                    current_state = instantaneous_state
                    candidate_state = current_state
                    candidate_count = 0

                    if current_state == "occupied" and previous_state == "empty":
                        events.append(
                            EventRecord(
                                timestamp=round(timestamp, 3),
                                frame_index=frame_index,
                                event="approach",
                            )
                        )
                        approach_timestamps.append(timestamp)
                    elif current_state == "empty" and previous_state == "occupied":
                        events.append(
                            EventRecord(
                                timestamp=round(timestamp, 3),
                                frame_index=frame_index,
                                event="empty",
                            )
                        )
                        empty_timestamps.append(timestamp)
            else:
                candidate_state = current_state
                candidate_count = 0

            if current_state == "empty" and config.background_update_alpha > 0:
                cv2.accumulateWeighted(
                    roi_frame.astype(np.float32),
                    background,
                    config.background_update_alpha,
                )

            timeline_states.append(current_state)

            color = (0, 255, 0) if current_state == "empty" else (0, 0, 255)
            cv2.rectangle(
                frame,
                (config.roi.x, config.roi.y),
                (config.roi.x + config.roi.w, config.roi.y + config.roi.h),
                color,
                2,
            )
            cv2.putText(
                frame,
                f"state: {current_state}",
                (config.roi.x, max(20, config.roi.y - 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"motion_ratio: {motion_ratio:.4f}",
                (config.roi.x, max(40, config.roi.y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

            out.write(frame)
            frame_index += 1

            if config.progress_every and frame_index % config.progress_every == 0:
                print(f"Обработано кадров: {frame_index}")

    finally:
        cap.release()
        out.release()

    delays: List[float] = []
    for empty_time in empty_timestamps:
        next_approaches = [t for t in approach_timestamps if t > empty_time]
        if next_approaches:
            delays.append(next_approaches[0] - empty_time)

    mean_delay, median_delay, min_delay, max_delay = calculate_delay_stats(delays)
    segments = finalize_segments(timeline_states, fps, len(initial_frames))

    event_rows = [asdict(event) for event in events]
    segment_rows = [asdict(segment) for segment in segments]

    events_df = pd.DataFrame(event_rows, columns=["timestamp", "frame_index", "event"])
    segments_df = pd.DataFrame(
        segment_rows,
        columns=["state", "start_time", "end_time", "duration", "start_frame", "end_frame"],
    )

    occupied_time = float(
        segments_df.loc[segments_df["state"] == "occupied", "duration"].sum()
    ) if not segments_df.empty else 0.0
    empty_time = float(
        segments_df.loc[segments_df["state"] == "empty", "duration"].sum()
    ) if not segments_df.empty else 0.0
    total_timeline_duration = occupied_time + empty_time
    occupancy_ratio = occupied_time / total_timeline_duration if total_timeline_duration else 0.0

    summary = DetectionSummary(
        fps=float(fps),
        total_frames_processed=int(frame_index),
        video_duration_sec=round(frame_index / fps, 3),
        roi=asdict(config.roi),
        events_detected=len(events),
        approach_events=sum(1 for event in events if event.event == "approach"),
        empty_events=sum(1 for event in events if event.event == "empty"),
        occupied_segments=int((segments_df["state"] == "occupied").sum()) if not segments_df.empty else 0,
        occupied_time_sec=round(occupied_time, 3),
        empty_time_sec=round(empty_time, 3),
        occupancy_ratio=round(occupancy_ratio, 4),
        mean_reseat_delay_sec=round(mean_delay, 3) if mean_delay is not None else None,
        median_reseat_delay_sec=round(median_delay, 3) if median_delay is not None else None,
        min_reseat_delay_sec=round(min_delay, 3) if min_delay is not None else None,
        max_reseat_delay_sec=round(max_delay, 3) if max_delay is not None else None,
    )

    return events_df, segments_df, summary


def save_report(
    events_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    summary: DetectionSummary,
    output_path: Path,
) -> None:
    base_name = output_path.with_suffix("")

    events_csv_path = base_name.with_name(f"{base_name.name}_events.csv")
    segments_csv_path = base_name.with_name(f"{base_name.name}_segments.csv")
    txt_path = base_name.with_name(f"{base_name.name}_report.txt")
    json_path = base_name.with_name(f"{base_name.name}_summary.json")

    events_df.to_csv(events_csv_path, index=False, encoding="utf-8-sig")
    segments_df.to_csv(segments_csv_path, index=False, encoding="utf-8-sig")

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(summary), file, ensure_ascii=False, indent=2)

    with txt_path.open("w", encoding="utf-8") as file:
        file.write("Отчет по детекции столика\n")
        file.write("=========================\n\n")

        file.write("Сводка:\n")
        file.write(f"- FPS: {summary.fps:.3f}\n")
        file.write(f"- Обработано кадров: {summary.total_frames_processed}\n")
        file.write(f"- Длительность видео: {summary.video_duration_sec:.2f} сек.\n")
        file.write(f"- ROI: {summary.roi}\n")
        file.write(f"- Событий: {summary.events_detected}\n")
        file.write(f"- Занятых сегментов: {summary.occupied_segments}\n")
        file.write(f"- Время occupied: {summary.occupied_time_sec:.2f} сек.\n")
        file.write(f"- Время empty: {summary.empty_time_sec:.2f} сек.\n")
        file.write(f"- Доля занятости: {summary.occupancy_ratio:.2%}\n")
        if summary.mean_reseat_delay_sec is not None:
            file.write(
                f"- Средняя задержка до следующего подхода: {summary.mean_reseat_delay_sec:.2f} сек.\n"
            )
            file.write(
                f"- Медианная задержка: {summary.median_reseat_delay_sec:.2f} сек.\n"
            )
        else:
            file.write("- Недостаточно данных для расчета задержки до следующего подхода.\n")

        file.write("\nСобытия:\n")
        if events_df.empty:
            file.write("События не обнаружены.\n")
        else:
            file.write(events_df.to_string(index=False))
            file.write("\n")

        file.write("\nСегменты состояний:\n")
        if segments_df.empty:
            file.write("Сегменты не обнаружены.\n")
        else:
            file.write(segments_df.to_string(index=False))
            file.write("\n")

    print(f"CSV событий сохранен: {events_csv_path}")
    print(f"CSV сегментов сохранен: {segments_csv_path}")
    print(f"JSON сводка сохранена: {json_path}")
    print(f"Текстовый отчет сохранен: {txt_path}")


def open_first_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        ret, first_frame = cap.read()
    finally:
        cap.release()

    if not ret or first_frame is None:
        raise RuntimeError(f"Не удалось прочитать первый кадр из {video_path}")
    return first_frame


def build_config(args: argparse.Namespace, first_frame: np.ndarray) -> DetectionConfig:
    roi = validate_roi(first_frame, initialise_roi(first_frame, args.roi, args.select_roi))
    return DetectionConfig(
        video_path=Path(args.video),
        output_path=Path(args.output),
        roi=roi,
        bg_frames=args.bg_frames,
        threshold=args.threshold,
        min_frames_for_change=args.min_frames_for_change,
        binary_threshold=args.binary_threshold,
        blur_size=args.blur_size,
        background_update_alpha=args.background_update_alpha,
        progress_every=args.progress_every,
    )


def print_summary(events_df: pd.DataFrame, summary: DetectionSummary, output_path: Path) -> None:
    print(f"Выходное видео сохранено: {output_path}")
    print(
        f"Событий: {summary.events_detected}, "
        f"occupied_time={summary.occupied_time_sec:.2f} сек., "
        f"occupancy_ratio={summary.occupancy_ratio:.2%}"
    )

    if events_df.empty:
        print("Событий не найдено.")
    else:
        print("\nОбнаруженные события:")
        print(events_df.to_string(index=False))

    if summary.mean_reseat_delay_sec is not None:
        print(
            "\nСреднее время между уходом и следующим подходом: "
            f"{summary.mean_reseat_delay_sec:.2f} сек."
        )
    else:
        print("\nСреднее время не удалось посчитать: недостаточно пар empty -> approach.")


def main() -> None:
    args = parse_arguments()
    video_path = Path(args.video)

    if not video_path.is_file():
        print(f"Ошибка: файл видео не найден: {video_path}")
        sys.exit(1)

    try:
        first_frame = open_first_frame(video_path)
        config = build_config(args, first_frame)
    except Exception as exc:
        print(f"Ошибка выбора ROI: {exc}")
        sys.exit(1)

    print(
        f"Используется ROI: x={config.roi.x}, y={config.roi.y}, "
        f"w={config.roi.w}, h={config.roi.h}"
    )

    try:
        events_df, segments_df, summary = detect_events(config)
        save_report(events_df, segments_df, summary, config.output_path)
    except Exception as exc:
        print(f"Ошибка обработки видео: {exc}")
        sys.exit(1)

    print_summary(events_df, summary, config.output_path)


if __name__ == "__main__":
    main()
