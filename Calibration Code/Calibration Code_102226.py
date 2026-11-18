import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
import cv2
import mediapipe.python.solutions.pose as mp_pose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.drawing.image import Image as OpenPyXLImage

# ==============================================================================
# SECTION 1: GLOBAL CONFIGURATIONS & CHARUCO DEFINITIONS
# ==============================================================================
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

CAMERA_CONFIGS = {
    "Frontal": {"label": "Frontal (Wide Lens)", "wide": True, "board_paths": [], "human_path": ""},
    "45Left": {"label": "45 Left (Linear)", "wide": False, "board_paths": [], "human_path": ""},
    "45Right": {"label": "45 Right (Linear)", "wide": False, "board_paths": [], "human_path": ""},
    "LateralLeft": {"label": "Lateral Left (Sagittal)", "wide": False, "board_paths": [], "human_path": ""},
    "LateralRight": {"label": "Lateral Right (Sagittal)", "wide": False, "board_paths": [], "human_path": ""},
    "Overhead": {"label": "Overhead (Linear Top-Down)", "wide": False, "board_paths": [], "human_path": ""}
}

# 36"x24" Board: 35x23 Grid, 1.00" Square (0.0254m), 0.75" Marker (0.01905m)
SQUARES_X = 35
SQUARES_Y = 23
SQUARE_LENGTH = 0.0254
MARKER_LENGTH = 0.01905

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)

try:
    board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
except AttributeError:
    board = cv2.aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, dictionary)

aruco_params = cv2.aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_params.adaptiveThreshWinSizeMin = 3
aruco_params.adaptiveThreshWinSizeMax = 53
aruco_params.adaptiveThreshWinSizeStep = 4
aruco_params.minMarkerPerimeterRate = 0.010

try:
    charuco_params = cv2.aruco.CharucoParameters() if hasattr(cv2.aruco, "CharucoParameters") else None
    if charuco_params:
        detector = cv2.aruco.CharucoDetector(board, charuco_params, aruco_params)
    else:
        detector = cv2.aruco.CharucoDetector(board, detectorParams=aruco_params)
except (TypeError, AttributeError):
    try:
        detector = cv2.aruco.CharucoDetector(board)
    except (TypeError, AttributeError):
        detector = None


def resolve_take_identity_dynamic(filename, dynamic_take_map):
    fn = os.path.basename(filename).lower().replace(" ", "").replace("_", "").replace("-", "")

    for take_key, cfg in dynamic_take_map.items():
        if take_key in fn:
            return take_key, cfg["spot"]

    # Takes 1-11 Automated Mapping
    if "take1" in fn:
        return "take1", "dynamic_sweep"
    if "take2" in fn:
        return "take2", "center"
    if "take3" in fn:
        return "take3", "planar_translation"
    if "take4" in fn or "inandout" in fn or "radial" in fn:
        return "take4", "radial_in_out"
    if "take5" in fn:
        return "take5", "center"
    if "take6" in fn:
        return "take6", "top_right"
    if "take7" in fn:
        return "take7", "top_left"
    if "take8" in fn:
        return "take8", "bottom_right"
    if "take9" in fn:
        return "take9", "bottom_left"
    if "take10" in fn or "centerswivel" in fn:
        return "take10", "center"
    if "take11" in fn or "apose" in fn or "human" in fn:
        return "take11", "human_baseline"

    if "topleft" in fn or "frontleft" in fn:
        return "custom", "top_left"
    if "topright" in fn or "frontright" in fn:
        return "custom", "top_right"
    if "bottomright" in fn or "backright" in fn:
        return "custom", "bottom_right"
    if "bottomleft" in fn or "backleft" in fn:
        return "custom", "bottom_left"
    if "center" in fn:
        return "custom", "center"

    return "unknown", None


def preprocess_for_detection(frame, is_wide):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if is_wide:
        gray = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def detect_charuco(gray_frame, is_upscaled=False):
    corners, ids = None, None
    if detector is not None:
        try:
            corners, ids, _, _ = detector.detectBoard(gray_frame)
        except Exception:
            corners, ids = None, None
    else:
        m_corners, m_ids, _ = cv2.aruco.detectMarkers(gray_frame, dictionary, parameters=aruco_params)
        if m_ids is not None and len(m_ids) > 0:
            _, corners, ids = cv2.aruco.interpolateCornersCharuco(m_corners, m_ids, gray_frame, board)

    if is_upscaled and corners is not None:
        corners = corners / 2.0

    return corners, ids


def is_non_degenerate(corners, min_span_px=25.0):
    if corners is None or len(corners) < 6:
        return False
    pts = corners.reshape(-1, 2)
    span_x = np.ptp(pts[:, 0])
    span_y = np.ptp(pts[:, 1])
    return bool((span_x >= min_span_px) and (span_y >= min_span_px))


def run_charuco_calibration(corners_list, ids_list, board_obj, img_size, is_wide):
    w, h = img_size
    init_f = float(max(w, h))
    K_init = np.array([
        [init_f, 0.0,    w / 2.0],
        [0.0,    init_f, h / 2.0],
        [0.0,    0.0,    1.0]
    ], dtype=np.float64)

    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    if is_wide:
        flags |= cv2.CALIB_RATIONAL_MODEL

    filtered_corners, filtered_ids = [], []
    for c, i in zip(corners_list, ids_list):
        if is_non_degenerate(c):
            filtered_corners.append(c)
            filtered_ids.append(i)

    if len(filtered_corners) < 5:
        raise ValueError(f"Only {len(filtered_corners)} valid frames found (minimum 5 required).")

    if hasattr(cv2.aruco, "calibrateCameraCharucoExtended"):
        res = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=filtered_corners,
            charucoIds=filtered_ids,
            board=board_obj,
            imageSize=img_size,
            cameraMatrix=K_init,
            distCoeffs=None,
            flags=flags
        )
        return res[0], res[1], res[2]
    elif hasattr(cv2.aruco, "calibrateCameraCharuco"):
        res = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=filtered_corners,
            charucoIds=filtered_ids,
            board=board_obj,
            imageSize=img_size,
            cameraMatrix=K_init,
            distCoeffs=None,
            flags=flags
        )
        return res[0], res[1], res[2]
    else:
        obj_points, img_points = [], []
        for c, i in zip(filtered_corners, filtered_ids):
            obj_p, img_p = board_obj.matchImagePoints(c, i)
            if obj_p is not None and len(obj_p) >= 6:
                obj_points.append(obj_p)
                img_points.append(img_p)
        ret, K, dist, _, _ = cv2.calibrateCamera(
            obj_points, img_points, img_size, K_init, None, flags=flags
        )
        return ret, K, dist


# ==============================================================================
# SECTION 2: HARDWARE SYNCHRONIZATION CHECKPOINT
# ==============================================================================
def validate_sync_checkpoint(inputs, frame_tolerance=5, time_tolerance_sec=120.0):
    active_paths = {cid: cfg["human_path"] for cid, cfg in inputs.items() if cfg["human_path"] and os.path.exists(cfg["human_path"])}
    if len(active_paths) < 2:
        return True, "Fewer than 2 human streams supplied; skipping cross-check."

    video_meta = {}
    for cid, path in active_paths.items():
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0
        cap.release()
        video_meta[cid] = {
            "fps": fps,
            "frames": frame_count,
            "duration": duration,
            "mtime": os.path.getmtime(path),
            "path": os.path.basename(path)
        }

    frame_counts = [m["frames"] for m in video_meta.values()]
    min_f, max_f = min(frame_counts), max(frame_counts)
    if (max_f - min_f) > frame_tolerance:
        err_msg = (
            f"Frame count mismatch across human streams (Take 11)!\n"
            f"Difference: {max_f - min_f} frames (Allowed: ±{frame_tolerance}).\n\n"
            + "\n".join([f"{cid}: {m['frames']} frames ({m['duration']:.2f}s) -> {m['path']}" for cid, m in video_meta.items()])
        )
        return False, err_msg

    fps_list = [round(m["fps"], 2) for m in video_meta.values()]
    if len(set(fps_list)) > 1:
        return False, "FPS mismatch across cameras:\n" + "\n".join([f"{cid}: {m['fps']:.2f} FPS" for cid, m in video_meta.items()])

    mtimes = [m["mtime"] for m in video_meta.values()]
    time_delta = max(mtimes) - min(mtimes)
    if time_delta > time_tolerance_sec:
        print(f"Warning: Timestamp delta is {time_delta:.1f}s. Frame counts are locked.")

    return True, f"Hardware sync verified: ~{min_f} frames across {len(active_paths)} streams at {fps_list[0]} FPS."


# ==============================================================================
# SECTION 3: CALIBRATION SCHEMATIC & QUALITY DIAGNOSTIC DASHBOARD
# ==============================================================================
def generate_calibration_diagram(table_dim_cm, corner_shifts_cm, measured_shifts_cm, calib_status_dict, output_png):
    fig, (ax_map, ax_diag) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=150, gridspec_kw={'width_ratios': [1.2, 1]})
    fig.patch.set_facecolor("#FAFAFB")

    tbl_w = table_dim_cm["length_cm"]
    tbl_h = table_dim_cm["width_cm"]

    ax_map.add_patch(plt.Rectangle(
        (-tbl_w / 2.0, -tbl_h / 2.0), tbl_w, tbl_h,
        fill=True, color="#EDF2F7", ec="#2D3748", lw=2, linestyle="--"
    ))

    ax_map.scatter(0, 0, color="#E53E3E", s=130, marker="P", label="Center Target", zorder=5)

    signs = {
        "top_left": (-1, 1, "Top-Left (FL)"),
        "top_right": (1, 1, "Top-Right (FR)"),
        "bottom_right": (1, -1, "Bottom-Right (BR)"),
        "bottom_left": (-1, -1, "Bottom-Left (BL)")
    }

    err_magnitudes = []
    for pos, (sx, sy, label_str) in signs.items():
        d_target = corner_shifts_cm.get(pos, 0.0)
        tx = sx * d_target / np.sqrt(2)
        ty = sy * d_target / np.sqrt(2)

        ax_map.scatter(tx, ty, color="#2B6CB0", s=90, marker="o", zorder=5)
        ax_map.plot([0, tx], [0, ty], color="#CBD5E0", linestyle=":", lw=1.2)

        v_shift = measured_shifts_cm.get(pos, np.nan)
        if not np.isnan(v_shift):
            scale = v_shift / d_target if d_target > 0 else 1.0
            vx, vy = tx * scale, ty * scale
            err_cm = abs(v_shift - d_target)
            err_magnitudes.append(err_cm)

            ax_map.scatter(vx, vy, color="#D69E2E" if err_cm < 2.5 else "#E53E3E", s=70, marker="x", lw=2, zorder=6)
            ax_map.annotate("", xy=(vx, vy), xytext=(tx, ty),
                            arrowprops=dict(arrowstyle="->", color="#C53030", lw=1.5, mutation_scale=12))

            badge_color = "#38A169" if err_cm <= 1.5 else "#D69E2E" if err_cm <= 3.0 else "#E53E3E"
            info_text = f"{label_str}\nTarget: {d_target:.1f}cm\nMeas: {v_shift:.1f}cm\nΔ: {err_cm:.2f}cm"
        else:
            badge_color = "#A0AEC0"
            info_text = f"{label_str}\nTarget: {d_target:.1f}cm\nMeas: FAILED"

        ax_map.annotate(
            info_text, (tx, ty),
            textcoords="offset points", xytext=(15 * sx, 15 * sy),
            ha="left" if sx > 0 else "right", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=badge_color, lw=1.8)
        )

    rig_cams = [
        ("▲ Frontal (Wide)", 0, tbl_h / 2.0 + 8, "#2C7A7B"),
        ("◤ 45 Left", -tbl_w / 2.0 - 6, tbl_h / 2.0 + 6, "#4A5568"),
        ("◥ 45 Right", tbl_w / 2.0 + 6, tbl_h / 2.0 + 6, "#4A5568"),
        ("◀ Lateral Left", -tbl_w / 2.0 - 12, 0, "#2B6CB0"),
        ("▶ Lateral Right", tbl_w / 2.0 + 12, 0, "#2B6CB0"),
        ("⦿ Overhead (Nadir)", 0, 0, "#805AD5")
    ]
    for label, cx, cy, col in rig_cams:
        ax_map.text(cx, cy, label, ha="center", va="center", fontsize=8, fontweight="bold", color=col)

    margin = 40
    ax_map.set_xlim(-tbl_w / 2.0 - margin, tbl_w / 2.0 + margin)
    ax_map.set_ylim(-tbl_h / 2.0 - margin, tbl_h / 2.0 + margin + 10)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_title("6-Camera Table Plane Residual Vector Map", fontsize=11, fontweight="bold", color="#1A202C")
    ax_map.set_xlabel("X Length (cm)", fontsize=9)
    ax_map.set_ylabel("Y Width (cm)", fontsize=9)
    ax_map.grid(True, linestyle="--", alpha=0.4)

    ax_diag.axis("off")
    cams = list(calib_status_dict.keys())
    reproj_errors = [calib_status_dict[c].get("reproj_error_px", np.nan) for c in cams]

    bar_ax = fig.add_axes([0.58, 0.40, 0.38, 0.50])
    clean_errs = [e if not np.isnan(e) else 0.0 for e in reproj_errors]
    bar_colors = ["#38A169" if e < 0.30 else "#D69E2E" if e < 0.60 else "#E53E3E" if not np.isnan(e) else "#A0AEC0" for e in reproj_errors]

    bars = bar_ax.barh(cams, clean_errs, color=bar_colors, height=0.55, edgecolor="#4A5568", lw=0.8)
    bar_ax.axvline(0.30, color="#38A169", linestyle="--", lw=1.2, label="Target (<0.30px)")
    bar_ax.axvline(0.60, color="#E53E3E", linestyle=":", lw=1.2, label="Acceptable (0.60px)")

    bar_ax.set_xlabel("Reprojection Error (pixels)", fontsize=9, fontweight="bold")
    bar_ax.set_title("6-Camera Intrinsic Health Check", fontsize=11, fontweight="bold", color="#1A202C")
    bar_ax.set_xlim(0, max(max(clean_errs + [0.8]) * 1.25, 0.8))
    bar_ax.grid(axis="x", linestyle="--", alpha=0.4)
    bar_ax.legend(loc="lower right", fontsize=8)

    for bar, val in zip(bars, reproj_errors):
        txt = f"{val:.3f} px" if not np.isnan(val) else "FAILED"
        bar_ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2.0, txt,
                    va="center", fontsize=8, fontweight="bold", color="#2D3748")

    avg_reproj = float(np.nanmean(reproj_errors)) if np.any(~np.isnan(reproj_errors)) else np.nan
    mean_linear_err = float(np.mean(err_magnitudes)) if err_magnitudes else np.nan

    status_str = "OPTIMAL" if (avg_reproj < 0.35 and mean_linear_err < 2.0) else "DEGRADED" if (avg_reproj < 0.65) else "POOR"
    status_color = "#22543D" if status_str == "OPTIMAL" else "#744210" if status_str == "DEGRADED" else "#742A2A"
    box_color = "#C6F6D5" if status_str == "OPTIMAL" else "#FEFCBF" if status_str == "DEGRADED" else "#FED7D7"

    max_radial = max(err_magnitudes) if err_magnitudes else 0.0
    summary_text = (
        f"CALIBRATION QUALITY VERDICT: {status_str}\n"
        f"------------------------------------------------\n"
        f"* Active Channels               : {len(cams)}/6\n"
        f"* Mean Reprojection Error       : {avg_reproj:.3f} px\n"
        f"* Mean Table Shift Error        : {mean_linear_err:.2f} cm\n"
        f"* Max Radial Corner Drift       : {max_radial:.2f} cm\n"
        f"* Dedicated In-Out Radial Take  : Take 4 Active\n"
        f"* Center Multi-Angle Swivel     : Takes 5 & 10 Active"
    )

    fig.text(
        0.58, 0.10, summary_text,
        fontsize=8.5, fontfamily="monospace", color=status_color,
        bbox=dict(boxstyle="round,pad=0.6", fc=box_color, ec=status_color, lw=1.2)
    )

    plt.savefig(output_png, bbox_inches="tight")
    plt.close()


# ==============================================================================
# SECTION 4: PIPELINE EXECUTION
# ==============================================================================
def run_pipeline(inputs, table_dim_cm, corner_shifts_cm, dynamic_take_map, status_cb=None):
    def update_status(text):
        print(text)
        if status_cb:
            status_cb(text)

    update_status("Verifying hardware stream synchronization across 6 channels...")
    is_synced, sync_report = validate_sync_checkpoint(inputs)
    print(sync_report)

    if not is_synced:
        should_continue = [False]
        ev = threading.Event()
        def ask():
            should_continue[0] = messagebox.askyesno("Sync Discrepancy", f"{sync_report}\n\nProceed with analysis anyway?")
            ev.set()
        root.after(0, ask)
        ev.wait()
        if not should_continue[0]:
            update_status("Aborted due to sync discrepancy.")
            return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), f"analysis_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    update_status(f"Output folder: {output_dir}")

    table_height_m = table_dim_cm["height_cm"] / 100.0
    d_m = {pos: val / 100.0 for pos, val in corner_shifts_cm.items()}

    metric_target_positions_m = {
        "center": np.array([0.0, 0.0, table_height_m]),
        "top_left": np.array([-d_m.get("top_left", 0.4) / np.sqrt(2), d_m.get("top_left", 0.4) / np.sqrt(2), table_height_m]),
        "top_right": np.array([d_m.get("top_right", 0.65) / np.sqrt(2), d_m.get("top_right", 0.65) / np.sqrt(2), table_height_m]),
        "bottom_right": np.array([d_m.get("bottom_right", 0.65) / np.sqrt(2), -d_m.get("bottom_right", 0.65) / np.sqrt(2), table_height_m]),
        "bottom_left": np.array([-d_m.get("bottom_left", 0.4) / np.sqrt(2), -d_m.get("bottom_left", 0.4) / np.sqrt(2), table_height_m]),
    }

    camera_matrices = {}
    virtual_corner_positions = {
        cam_id: {pos: [] for pos in list(corner_shifts_cm.keys()) + ["center"]}
        for cam_id in inputs.keys()
    }
    calibration_status = {}

    for cam_id, cfg in inputs.items():
        board_clips = cfg["board_paths"]
        if not board_clips:
            calibration_status[cam_id] = {
                "success": False, "views": 0, "error": "No board clips selected in UI."
            }
            continue

        is_wide = cfg["wide"]
        all_corners, all_ids = [], []
        cached_spatial_corners = []
        image_size = None

        update_status(f"[{cam_id}] Decoding clips & extracting ChArUco corners (Takes 1-10)...")

        for clip_path in board_clips:
            if not os.path.exists(clip_path):
                continue

            take_id, spot_id = resolve_take_identity_dynamic(clip_path, dynamic_take_map)
            cap = cv2.VideoCapture(clip_path)
            f_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                f_idx += 1
                if f_idx % 4 != 0:
                    continue

                if image_size is None:
                    image_size = (frame.shape[1], frame.shape[0])

                proc_gray = preprocess_for_detection(frame, is_wide)
                corners, ids = detect_charuco(proc_gray, is_upscaled=is_wide)

                if corners is not None and ids is not None and len(ids) >= 6:
                    all_corners.append(corners)
                    all_ids.append(ids)

                    if spot_id not in [None, "dynamic_sweep", "planar_translation", "radial_in_out", "ignore"]:
                        cached_spatial_corners.append((spot_id, corners, ids))

            cap.release()

        views_count = len(all_corners)
        update_status(f"[{cam_id}] Accumulated {views_count} valid views.")

        if views_count < 5:
            err_text = f"Insufficient views ({views_count} found, min 5 required)."
            calibration_status[cam_id] = {"success": False, "views": views_count, "error": err_text}
            continue

        try:
            err, K, dist = run_charuco_calibration(all_corners, all_ids, board, image_size, is_wide)
            out_calib = os.path.join(output_dir, f"calib_{cam_id}.npz")
            np.savez(out_calib, camera_matrix=K, dist_coeffs=dist, image_size=image_size)
            camera_matrices[cam_id] = {"K": K, "dist": dist, "img_size": image_size}
            calibration_status[cam_id] = {
                "success": True, "views": views_count, "reproj_error_px": round(err, 4), "error": "None"
            }
            update_status(f"[{cam_id}] Calibrated successfully: Reproj Error = {err:.4f} px")
        except Exception as e:
            err_text = f"Math solver exception: {str(e)}"
            calibration_status[cam_id] = {"success": False, "views": views_count, "error": err_text}
            continue

        flags_pnp = cv2.SOLVEPNP_SQPNP if hasattr(cv2, "SOLVEPNP_SQPNP") else cv2.SOLVEPNP_ITERATIVE
        spot_tvecs_grouped = {pos: [] for pos in virtual_corner_positions[cam_id].keys()}

        for spot_id, corners, ids in cached_spatial_corners:
            obj_pts, img_pts = board.matchImagePoints(corners, ids)
            if obj_pts is not None and len(obj_pts) >= 6:
                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=flags_pnp)
                if ok and spot_id in spot_tvecs_grouped:
                    spot_tvecs_grouped[spot_id].append(tvec.flatten())

        for spot_id, tvecs in spot_tvecs_grouped.items():
            if tvecs:
                virtual_corner_positions[cam_id][spot_id].append(np.median(tvecs, axis=0))

    # Phase 2: Pose Extraction (Take 11 A-Pose)
    update_status("Starting MediaPipe pose extraction on Take 11 (Complexity 1)...")
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
    all_camera_dataframes = {}

    for cam_id, cfg in inputs.items():
        human_video = cfg["human_path"]
        calib_file = os.path.join(output_dir, f"calib_{cam_id}.npz")
        if not human_video or not os.path.exists(human_video):
            continue

        has_calib = os.path.exists(calib_file)
        map_x, map_y = None, None
        if has_calib:
            calib = np.load(calib_file)
            K, dist = calib["camera_matrix"], calib["dist_coeffs"]
            w, h = calib["image_size"]
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0, newImgSize=(w, h))
            map_x, map_y = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_32FC1)

        cap = cv2.VideoCapture(human_video)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_records = []
        current_frame = 0

        update_status(f"[{cam_id}] Tracking pose across {total_frames} frames...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR) if (has_calib and map_x is not None) else frame
            rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    frame_records.append({
                        "frame_number": current_frame,
                        "landmark_id": idx,
                        "landmark_name": LANDMARK_NAMES[idx],
                        "x_px": lm.x * w,
                        "y_px": lm.y * h,
                        "z_relative_depth": lm.z * w,
                        "visibility": lm.visibility
                    })
            else:
                for idx in range(33):
                    frame_records.append({
                        "frame_number": current_frame,
                        "landmark_id": idx,
                        "landmark_name": LANDMARK_NAMES[idx],
                        "x_px": np.nan,
                        "y_px": np.nan,
                        "z_relative_depth": np.nan,
                        "visibility": 0.0
                    })
            current_frame += 1

        cap.release()
        all_camera_dataframes[cam_id] = pd.DataFrame(frame_records)
        update_status(f"[{cam_id}] Pose extracted for {current_frame} frames.")

    pose.close()

    # Phase 3: Export & Schematic
    update_status("Generating 6-camera analysis workbook and schematic...")
    excel_filename = os.path.join(output_dir, f"multiview_pose_analysis_{timestamp}.xlsx")
    diagram_png = os.path.join(output_dir, f"table_schematic_{timestamp}.png")

    correlation_rows = []
    physical_displacements = []
    virtual_displacements = []
    measured_shifts_map = {}

    ref_cam = "Frontal" if "Frontal" in inputs and calibration_status.get("Frontal", {}).get("success") else next(
        (c for c, s in calibration_status.items() if s["success"]), None
    )

    ref_spots = virtual_corner_positions.get(ref_cam, {}) if ref_cam else {}
    center_tvecs = ref_spots.get("center", [])
    center_virtual = np.mean(center_tvecs, axis=0) if center_tvecs else None

    for pos in ["top_left", "top_right", "bottom_right", "bottom_left"]:
        phys_shift_cm = corner_shifts_cm.get(pos, 0.0)
        physical_displacements.append(phys_shift_cm)

        pos_tvecs = ref_spots.get(pos, [])
        if center_virtual is not None and pos_tvecs:
            pos_virtual = np.mean(pos_tvecs, axis=0)
            virt_shift_cm = float(np.linalg.norm(pos_virtual - center_virtual) * 100.0)
        else:
            virt_shift_cm = np.nan

        virtual_displacements.append(virt_shift_cm)
        measured_shifts_map[pos] = virt_shift_cm

        scaling_ratio = (virt_shift_cm / phys_shift_cm) if (phys_shift_cm > 0 and not np.isnan(virt_shift_cm)) else np.nan

        correlation_rows.append({
            "Corner_Target": pos.replace("_", " ").title(),
            "Physical_Shift_cm": phys_shift_cm,
            "Virtual_Measured_Shift_cm": virt_shift_cm,
            "Scale_Ratio_Virtual_to_Physical": scaling_ratio,
            "Metric_X_m": metric_target_positions_m[pos][0],
            "Metric_Y_m": metric_target_positions_m[pos][1],
            "Metric_Z_m": metric_target_positions_m[pos][2]
        })

    valid_mask = ~np.isnan(virtual_displacements)
    if np.sum(valid_mask) >= 2:
        p_arr = np.array(physical_displacements)[valid_mask]
        v_arr = np.array(virtual_displacements)[valid_mask]
        std_p = np.std(p_arr)
        std_v = np.std(v_arr)

        if std_p > 1e-6 and std_v > 1e-6:
            pearson_r = float(np.corrcoef(p_arr, v_arr)[0, 1])
            corr_status_msg = f"Valid Pearson r = {pearson_r:.4f}"
        elif std_p <= 1e-6:
            pearson_r = np.nan
            corr_status_msg = "Undefined: Physical shifts have zero variance."
        else:
            pearson_r = np.nan
            corr_status_msg = "Undefined: Virtual measurements had zero variance."
    else:
        pearson_r = np.nan
        corr_status_msg = "Insufficient valid target measurements across corners."

    corr_df = pd.DataFrame(correlation_rows)
    summary_df = pd.DataFrame([
        {"Metric": "Table_Length_cm", "Value": table_dim_cm["length_cm"]},
        {"Metric": "Table_Width_cm", "Value": table_dim_cm["width_cm"]},
        {"Metric": "Table_Height_cm", "Value": table_dim_cm["height_cm"]},
        {"Metric": "Reference_Camera_Used", "Value": str(ref_cam)},
        {"Metric": "Pearson_Correlation_Physical_vs_Virtual", "Value": pearson_r},
        {"Metric": "Correlation_Notes", "Value": corr_status_msg}
    ])

    calib_summary_rows = []
    for cid, stat in calibration_status.items():
        calib_summary_rows.append({
            "Camera": cid,
            "Calibrated": "YES" if stat["success"] else "NO",
            "Valid_Views_Detected": stat["views"],
            "Reprojection_Error_px": stat.get("reproj_error_px", "N/A"),
            "Error_Reason": stat["error"]
        })
    calib_df = pd.DataFrame(calib_summary_rows)

    with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
        calib_df.to_excel(writer, sheet_name="Calibration_Status", index=False)
        corr_df.to_excel(writer, sheet_name="Target_Shift_Correlation", index=False)
        summary_df.to_excel(writer, sheet_name="Dimension_Summary", index=False)
        for cam_id, df in all_camera_dataframes.items():
            df.to_excel(writer, sheet_name=f"{cam_id}_frames", index=False)

    generate_calibration_diagram(table_dim_cm, corner_shifts_cm, measured_shifts_map, calibration_status, diagram_png)

    wb = openpyxl.load_workbook(excel_filename)
    if "Target_Shift_Correlation" in wb.sheetnames:
        ws = wb["Target_Shift_Correlation"]
        img = OpenPyXLImage(diagram_png)
        ws.add_image(img, "I2")
        wb.save(excel_filename)

    calibrated_cams = [cid for cid, s in calibration_status.items() if s["success"]]
    failed_cams = [f"{cid} ({s['error']})" for cid, s in calibration_status.items() if not s["success"]]

    result_dialog = (
        f"Processing Complete!\n\n"
        f"CALIBRATED CAMERAS ({len(calibrated_cams)}/6):\n"
        f"  {', '.join(calibrated_cams) if calibrated_cams else 'None'}\n\n"
        f"FAILED / SKIPPED CAMERAS ({len(failed_cams)}):\n  "
        + "\n  ".join(failed_cams)
        + f"\n\nCorrelation: {corr_status_msg}\n"
        f"Diagnostic Diagram embedded into Target_Shift_Correlation.\n"
        f"Saved to: {os.path.basename(excel_filename)}"
    )

    update_status("Pipeline finished successfully.")

    def finish_gui():
        if failed_cams:
            messagebox.showwarning("Calibration Diagnostics", result_dialog)
        else:
            messagebox.showinfo("Done", result_dialog)
        start_btn.config(state="normal", text="Run Calibration, Dynamic Mapping & Pose Tracking")
        status_label.config(text="Status: Completed", fg="green")

    root.after(0, finish_gui)


# ==============================================================================
# SECTION 5: 6-CAMERA GUI LAUNCHER WITH TAKES 1-11 MAPPING
# ==============================================================================
root = tk.Tk()
root.title("6-Camera Kinematic Rig Calibration & Dynamic Pose Tracking Engine")
root.geometry("980x920")

camera_inputs = CAMERA_CONFIGS
board_labels = {}
human_labels = {}

def pick_files(cam_id, mode):
    display_name = camera_inputs[cam_id]["label"]
    if mode == "board":
        paths = filedialog.askopenfilenames(
            title=f"Select Calibration Takes for {display_name} (Takes 1-10)",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")]
        )
        if paths:
            camera_inputs[cam_id]["board_paths"] = list(paths)
            board_labels[cam_id].config(text=f"{len(paths)} clip(s)", fg="green")
    else:
        path = filedialog.askopenfilename(
            title=f"Select Synchronized Human Video for {display_name} (Take 11 A-Pose)",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")]
        )
        if path:
            camera_inputs[cam_id]["human_path"] = path
            human_labels[cam_id].config(text=os.path.basename(path), fg="green")

file_frame = tk.LabelFrame(root, text=" 1. 6-Camera Input Streams ", font=("Arial", 10, "bold"), padx=10, pady=8)
file_frame.pack(fill="x", padx=14, pady=8)

tk.Label(file_frame, text="View Perspective", font=("Arial", 9, "bold")).grid(row=0, column=0, padx=8, pady=4, sticky="w")
tk.Label(file_frame, text="ChArUco Takes (Takes 1-10)", font=("Arial", 9, "bold")).grid(row=0, column=1, columnspan=2, padx=6, pady=4)
tk.Label(file_frame, text="Human A-Pose Video (Take 11)", font=("Arial", 9, "bold")).grid(row=0, column=3, columnspan=2, padx=6, pady=4)

row = 1
for cid, info in camera_inputs.items():
    tk.Label(file_frame, text=info["label"], font=("Arial", 9, "bold"), anchor="w", width=26).grid(row=row, column=0, padx=8, pady=4, sticky="w")

    btn_b = tk.Button(file_frame, text="Select Clips", width=11, command=lambda c=cid: pick_files(c, "board"))
    btn_b.grid(row=row, column=1, padx=4, pady=4)
    lbl_b = tk.Label(file_frame, text="No clips", fg="red", width=12, anchor="w")
    lbl_b.grid(row=row, column=2, padx=4, pady=4)
    board_labels[cid] = lbl_b

    btn_h = tk.Button(file_frame, text="Select Video", width=11, command=lambda c=cid: pick_files(c, "human"))
    btn_h.grid(row=row, column=3, padx=4, pady=4)
    lbl_h = tk.Label(file_frame, text="No file", fg="red", width=18, anchor="w")
    lbl_h.grid(row=row, column=4, padx=4, pady=4)
    human_labels[cid] = lbl_h
    row += 1

dim_frame = tk.LabelFrame(root, text=" 2. Physical Table Dimensions (cm) ", font=("Arial", 10, "bold"), padx=10, pady=8)
dim_frame.pack(fill="x", padx=14, pady=6)

tk.Label(dim_frame, text="Length (X) cm:", font=("Arial", 9)).grid(row=0, column=0, padx=6, pady=4, sticky="e")
entry_length = tk.Entry(dim_frame, width=8)
entry_length.insert(0, "120")
entry_length.grid(row=0, column=1, padx=6, pady=4)

tk.Label(dim_frame, text="Width (Y) cm:", font=("Arial", 9)).grid(row=0, column=2, padx=6, pady=4, sticky="e")
entry_width = tk.Entry(dim_frame, width=8)
entry_width.insert(0, "80")
entry_width.grid(row=0, column=3, padx=6, pady=4)

tk.Label(dim_frame, text="Height (Z / Floor to Table) cm:", font=("Arial", 9)).grid(row=0, column=4, padx=6, pady=4, sticky="e")
entry_height = tk.Entry(dim_frame, width=8)
entry_height.insert(0, "85")
entry_height.grid(row=0, column=5, padx=6, pady=4)

map_frame = tk.LabelFrame(root, text=" 3. Spatial Target & Swivel Mapping (Takes 2, 4-10) ", font=("Arial", 10, "bold"), padx=10, pady=8)
map_frame.pack(fill="x", padx=14, pady=6)

POS_OPTIONS = ["top_left", "top_right", "bottom_right", "bottom_left", "center", "ignore"]

active_takes = [
    ("take2", "Take 2 (Flat Center)", "center", "0.0"),
    ("take4", "Take 4 (Radial In/Out)", "ignore", "0.0"),
    ("take5", "Take 5 (Center Swivels)", "center", "0.0"),
    ("take6", "Take 6 (Front-Right Swivel)", "top_right", "65.0"),
    ("take7", "Take 7 (Front-Left Swivel)", "top_left", "40.0"),
    ("take8", "Take 8 (Back-Right Swivel)", "bottom_right", "65.0"),
    ("take9", "Take 9 (Back-Left Swivel)", "bottom_left", "40.0"),
    ("take10", "Take 10 (Center Continuous)", "center", "0.0"),
]

take_ui_controls = {}

for i, (tkey, tlabel, def_pos, def_dist) in enumerate(active_takes):
    grid_r = i // 2
    grid_c = (i % 2) * 3

    tk.Label(map_frame, text=tlabel, font=("Arial", 8, "bold"), width=24, anchor="w").grid(row=grid_r, column=grid_c, padx=6, pady=3, sticky="w")

    combo = ttk.Combobox(map_frame, values=POS_OPTIONS, width=12, state="readonly", font=("Arial", 8))
    combo.set(def_pos)
    combo.grid(row=grid_r, column=grid_c + 1, padx=4, pady=3)

    ent_dist = tk.Entry(map_frame, width=6, font=("Arial", 8))
    ent_dist.insert(0, def_dist)
    ent_dist.grid(row=grid_r, column=grid_c + 2, padx=4, pady=3)

    take_ui_controls[tkey] = {"spot_combo": combo, "dist_entry": ent_dist}


def on_run():
    table_dims = {
        "length_cm": float(entry_length.get().strip() or 120.0),
        "width_cm": float(entry_width.get().strip() or 80.0),
        "height_cm": float(entry_height.get().strip() or 85.0),
    }

    dynamic_take_map = {}
    corner_shifts_cm = {"top_left": 40.0, "top_right": 65.0, "bottom_right": 65.0, "bottom_left": 40.0}

    for tkey, ctrl in take_ui_controls.items():
        chosen_spot = ctrl["spot_combo"].get()
        chosen_dist = float(ctrl["dist_entry"].get().strip() or 0.0)
        dynamic_take_map[tkey] = {"spot": chosen_spot, "dist": chosen_dist}

        if chosen_spot in corner_shifts_cm and chosen_dist > 0.0:
            corner_shifts_cm[chosen_spot] = chosen_dist

    start_btn.config(state="disabled", text="Processing... (Check console & status below)")
    status_label.config(text="Status: Starting 6-camera worker pipeline...", fg="#D97706")

    def update_ui_status(msg):
        root.after(0, lambda: status_label.config(text=f"Status: {msg}", fg="#2563EB"))

    worker = threading.Thread(
        target=run_pipeline,
        args=(camera_inputs, table_dims, corner_shifts_cm, dynamic_take_map, update_ui_status),
        daemon=True
    )
    worker.start()


start_btn = tk.Button(
    root,
    text="Run Calibration, Dynamic Mapping & Pose Tracking",
    bg="#1B5E20",
    fg="white",
    font=("Arial", 11, "bold"),
    height=2,
    command=on_run
)
start_btn.pack(pady=8)

status_label = tk.Label(root, text="Status: Ready", font=("Arial", 9, "italic"), fg="#4B5563")
status_label.pack(pady=4)

root.mainloop()