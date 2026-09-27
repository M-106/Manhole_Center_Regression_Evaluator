import os
import shutil
import json
import glob
import pickle

import h5py
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import auc

import open3d as o3d
# FIXME Maybe add a more interactive Visualizer:
# import polyscope as ps



# ---------------------
# > Point Cloud Loader <
# ---------------------
def load_point_cloud_h5(path):
    """Loads an HDF5 point cloud file and returns an Open3D PointCloud object."""
    with h5py.File(path, "r") as file_:
        coordinates = file_["coords"][:]
        instances = file_["instances"][:] if "instances" in file_ else None
        intensities = file_["intensity"][:]
        semantics = file_["semantics"][:]

    point_cloud = o3d.t.geometry.PointCloud()
    point_cloud.point["positions"] = o3d.core.Tensor(coordinates, dtype=o3d.core.Dtype.Float32)
    point_cloud.point["classes"] = o3d.core.Tensor(semantics.reshape(-1, 1), dtype=o3d.core.Dtype.Int32)
    if instances is not None:
        point_cloud.point["instances"] = o3d.core.Tensor(instances.reshape(-1, 1), dtype=o3d.core.Dtype.Int32)
    point_cloud.point["intensity"] = o3d.core.Tensor(intensities.reshape(-1, 1), dtype=o3d.core.Dtype.Float32)

    return point_cloud



# ------------------------------------
# > Debug Plotter (Matplotlib 2D/3D) <
# ------------------------------------
def save_crop_debug_plot(
    coords,
    intensities,
    center_ref,
    gt_center=None,
    pred_center=None,
    error_m=None,
    title_info="",
    save_path="debug.png",
    crop_size=2.0,  # 2x2 Meter
):
    """
    Creates a 2D crop plot for debugging purposes.
    """
    cx, cy, cz = center_ref["x"], center_ref["y"], center_ref["z"]
    half_s = crop_size / 2.0

    # Mask for crop-area around the reference center
    mask = (
        (coords[:, 0] >= cx - half_s) & (coords[:, 0] <= cx + half_s) &
        (coords[:, 1] >= cy - half_s) & (coords[:, 1] <= cy + half_s)
    )
    crop_coords = coords[mask]
    crop_intens = intensities[mask] if intensities is not None else None

    if len(crop_coords) == 0:
        return  # No Points in the crop area

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bird's Eye View (X/Y Plane)
    ax_bev = axes[0]
    if crop_intens is not None and len(crop_intens) > 0:
        sc = ax_bev.scatter(crop_coords[:, 0], crop_coords[:, 1], c=crop_intens, cmap="gray", s=3, alpha=0.8)
        fig.colorbar(sc, ax=ax_bev, label="Intensity")
    else:
        ax_bev.scatter(crop_coords[:, 0], crop_coords[:, 1], c="gray", s=3, alpha=0.5)

    if gt_center is not None:
        ax_bev.scatter(gt_center["x"], gt_center["y"], c="lime", marker="o", s=100, edgecolors="black", label="GT Center")
    if pred_center is not None:
        ax_bev.scatter(pred_center["x"], pred_center["y"], c="red", marker="x", s=100, linewidths=2, label="Pred Center")

    if gt_center is not None and pred_center is not None:
        ax_bev.plot([gt_center["x"], pred_center["x"]], [gt_center["y"], pred_center["y"]], "b--", label=f"Err: {error_m:.3f}m")

    ax_bev.set_xlim(cx - half_s, cx + half_s)
    ax_bev.set_ylim(cy - half_s, cy + half_s)
    ax_bev.set_aspect("equal")
    ax_bev.set_title("Bird's Eye View (X-Y)")
    ax_bev.legend(loc="upper right")
    ax_bev.grid(True, linestyle=":", alpha=0.6)

    # --- 2. Side View / Height Profile (X/Z Plane) ---
    ax_side = axes[1]
    ax_side.scatter(crop_coords[:, 0], crop_coords[:, 2], c="tab:blue", s=3, alpha=0.6)

    if gt_center is not None:
        ax_side.scatter(gt_center["x"], gt_center["z"], c="lime", marker="o", s=100, edgecolors="black", label="GT Z")
    if pred_center is not None:
        ax_side.scatter(pred_center["x"], pred_center["z"], c="red", marker="x", s=100, linewidths=2, label="Pred Z")

    ax_side.set_xlim(cx - half_s, cx + half_s)
    ax_side.set_title("Side Profile (X-Z)")
    ax_side.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(f"{title_info}", fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)



def find_matching_patch_file(patch_dir, pc_id, target_x, target_y):
    """
    Searches for the appropriate .pkl patch file based on pc_id and world coordinates.
    File format: preprocessed_patch_bev_{pc_id}_{x_start}_{y_start}.pkl
    """
    pattern = os.path.join(patch_dir, f"preprocessed_patch_bev_{pc_id}_*.pkl")
    matching_files = glob.glob(pattern)

    for pkl_path in matching_files:
        try:
            with open(pkl_path, "rb") as f:
                tile, meta = pickle.load(f)

            # Checks if the target coordinates fall within the patch's bounding box
            pkl_file_name = ".".join(os.path.basename(pkl_path).split(".")[:-1])
            x_min = meta.get("origin_x", None)
            if x_min is None:
                x_min = pkl_file_name.split("_")[-1]
            y_min = meta.get("origin_y", None)
            if y_min is None:
                y_min = pkl_file_name.split("_")[-2]
            
            patch_size = meta.get("tile_size", 5.0) 
            # resolution = meta.get("resolution")

            x_max = x_min + patch_size
            y_max = y_min + patch_size

            if x_min <= target_x <= x_max and y_min <= target_y <= y_max:
                return pkl_path, tile, meta
        except Exception as e:
            continue

    return None, None, None


def save_crop_with_patch_debug_plot(
    coords,
    intensities,
    center_ref,
    gt_center=None,
    pred_center=None,
    error_m=None,
    patch_tile=None,
    patch_meta=None,
    title_info="",
    save_path="debug.png",
    crop_size=3.0,
):
    """
    Creates a 3-panel debug image:
    1. 3D Point Cloud Crop (BEV)
    2. Full 2D Patch Image (shows if the manhole is at the patch edge)
    3. Zoomed 2D Patch Crop
    """
    cx, cy = center_ref["x"], center_ref["y"]
    half_s = crop_size / 2.0

    # If patch_tile and patch_meta are provided, we will create a 3-panel figure; otherwise, only 2 panels.
    has_patch = patch_tile is not None and patch_meta is not None
    num_cols = 3 if has_patch else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))

    # 1. Point Cloud Crop (X-Y Plane)
    ax_pcd = axes[0]
    mask = (
        (coords[:, 0] >= cx - half_s) & (coords[:, 0] <= cx + half_s) &
        (coords[:, 1] >= cy - half_s) & (coords[:, 1] <= cy + half_s)
    )
    crop_coords = coords[mask]
    crop_intens = intensities[mask] if intensities is not None else None

    if crop_intens is not None and len(crop_intens) > 0:
        ax_pcd.scatter(crop_coords[:, 0], crop_coords[:, 1], c=crop_intens, cmap="plasma", s=4, alpha=0.8)  # plasma
    else:
        ax_pcd.scatter(crop_coords[:, 0], crop_coords[:, 1], c="gray", s=4, alpha=0.5)

    if gt_center is not None:
        ax_pcd.scatter(gt_center["x"], gt_center["y"], c="lime", marker="o", s=100, edgecolors="black", label="GT")
    if pred_center is not None:
        ax_pcd.scatter(pred_center["x"], pred_center["y"], c="red", marker="x", s=120, linewidths=2, label="Pred")

    if gt_center is not None and pred_center is not None:
        ax_pcd.plot([gt_center["x"], pred_center["x"]], [gt_center["y"], pred_center["y"]], "b--", label=f"Err: {error_m:.3f}m")

    ax_pcd.set_xlim(cx - half_s, cx + half_s)
    ax_pcd.set_ylim(cy - half_s, cy + half_s)
    ax_pcd.set_aspect("equal")
    ax_pcd.set_title("3D PCD Local Crop")
    ax_pcd.legend(loc="upper right", fontsize="small")
    ax_pcd.grid(True, linestyle=":", alpha=0.6)
    ax_pcd.tick_params(axis="x", labelrotation=45)

    # 2. Full 2D BEV Patch Plot
    if has_patch:
        ax_patch = axes[1]
        
        bev_img = patch_tile[2] if patch_tile.ndim == 3 else patch_tile
        bev_img = (bev_img - bev_img.min()) / (bev_img.max()-bev_img.min())

        x_start = patch_meta.get("origin_x")
        y_start = patch_meta.get("origin_y")
        res = patch_meta.get("resolution", 0.01)  # m/pixel
        h, w = bev_img.shape

        x_end = x_start + w * res
        y_end = y_start + h * res

        # Extend the image to match the world coordinates
        ax_patch.imshow(
            bev_img,
            cmap="plasma",
            extent=[x_start, x_end, y_start, y_end],
            origin="lower"
        )

        if gt_center is not None:
            ax_patch.scatter(gt_center["x"], gt_center["y"], c="lime", marker="o", s=80, edgecolors="black")
        if pred_center is not None:
            ax_patch.scatter(pred_center["x"], pred_center["y"], c="red", marker="x", s=100, linewidths=2)

        # mark if the manhole is located near the image border
        margin = 1.0
        is_edge = (cx - x_start < margin) or (x_end - cx < margin) or (cy - y_start < margin) or (y_end - cy < margin)
        edge_str = " [EDGE CASE]" if is_edge else ""

        ax_patch.set_title(f"Full BEV Patch{edge_str}")
        ax_patch.set_aspect("equal")
        ax_patch.grid(True, linestyle=":", alpha=0.4)
        ax_patch.tick_params(axis="x", labelrotation=45)

        # 3. Zoomed 2D Patch Crop
        ax_zoom = axes[2]
        ax_zoom.imshow(
            bev_img,
            cmap="plasma",
            extent=[x_start, x_end, y_start, y_end],
            origin="lower"
        )
        if gt_center is not None:
            ax_zoom.scatter(gt_center["x"], gt_center["y"], c="lime", marker="o", s=100, edgecolors="black")
        if pred_center is not None:
            ax_zoom.scatter(pred_center["x"], pred_center["y"], c="red", marker="x", s=120, linewidths=2)

        half_s /= 2.0
        ax_zoom.set_xlim(cx - half_s, cx + half_s)
        ax_zoom.set_ylim(cy - half_s, cy + half_s)
        ax_zoom.set_aspect("equal")
        ax_zoom.set_title("Zoomed BEV Patch Crop")
        ax_zoom.grid(True, linestyle=":", alpha=0.6)
        ax_zoom.tick_params(axis="x", labelrotation=45)

        # Enforce the same phsyical grid system
        grid_step = 1.0 # meters

        xmin = min(ax.get_xlim()[0] for ax in [ax_pcd, ax_patch, ax_zoom])
        xmax = max(ax.get_xlim()[1] for ax in [ax_pcd, ax_patch, ax_zoom])

        ymin = min(ax.get_ylim()[0] for ax in [ax_pcd, ax_patch, ax_zoom])
        ymax = max(ax.get_ylim()[1] for ax in [ax_pcd, ax_patch, ax_zoom])

        xticks = np.arange(
            np.floor(xmin / grid_step) * grid_step,
            xmax + grid_step,
            grid_step,
        )

        yticks = np.arange(
            np.floor(ymin / grid_step) * grid_step,
            ymax + grid_step,
            grid_step,
        )

        for ax in [ax_pcd, ax_patch, ax_zoom]:
            ax.xaxis.set_major_locator(FixedLocator(xticks))
            ax.yaxis.set_major_locator(FixedLocator(yticks))
            ax.ticklabel_format(style="plain", axis="both", useOffset=False)

    fig.suptitle(f"{title_info}", fontsize=11)
    # plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)



def debug_plots(evaluator, data_root_dir, debug_dir, patch_dir):
    summary = evaluator.get_summary()
    p50_err = summary["p50_error_m"]
    p95_err = summary["p95_error_m"]
    
    print("Generating Debug Plots...")
    pcd_cache = {}

    for idx, rec in enumerate(evaluator.matches_record):
        dataset, pc_id = rec["dataset"], rec["pc_id"]
        # pcd_file = os.path.join(data_root_dir, f"preprocessed_patch_{pc_id:04}.h5")
        if "sud" in dataset.lower():
            pcd_file = os.path.join(data_root_dir, f"{pc_id}.h5")
        else:
            pcd_file = os.path.join(data_root_dir, f"{pc_id:04}.h5")

        if not os.path.exists(pcd_file):
            continue

        # load Pointcloud
        if pc_id not in pcd_cache:
            t_pcd = load_point_cloud_h5(pcd_file)
            coords = t_pcd.point["positions"].numpy()
            intens = t_pcd.point["intensity"].numpy().flatten() if "intensity" in t_pcd.point else None
            pcd_cache[pc_id] = (coords, intens)
        else:
            coords, intens = pcd_cache[pc_id]

        ref_center = rec["gt"] if rec["gt"] is not None else rec["pred"]
        m_type = rec["type"]

        # Search the BEV-Patch file
        _, patch_tile, patch_meta = find_matching_patch_file(
            patch_dir=patch_dir,
            pc_id=pc_id,
            target_x=ref_center["x"],
            target_y=ref_center["y"]
        )

        # CASE 1: Bad Localization (P95)
        # FIXME: Should be via a given threshold or is it ok that "bad" is the worse from (only) this given model
        if m_type == "TP" and rec["dist"] >= p95_err:
            folder = os.path.join(debug_dir, "localization_bad_p95")
            title = f"Bad Localization (P95) - {dataset}_{pc_id} (Err: {rec['dist']:.3f}m)"
            save_path_img = os.path.join(folder, f"{dataset}_{pc_id}_tp_bad_{idx}.png")
            # save_crop_debug_plot(coords, intens, center_ref=rec["gt"], gt_center=rec["gt"], pred_center=rec["pred"], error_m=rec["dist"], title_info=title, save_path=save_path_img)
            save_crop_with_patch_debug_plot(
                coords, intens, center_ref=ref_center, gt_center=rec["gt"], pred_center=rec["pred"],
                error_m=rec["dist"], patch_tile=patch_tile, patch_meta=patch_meta, title_info=title, save_path=save_path_img
            )

        # CASE 2: Good Localization (P50) - as comparison to P95, but still a TP
        elif m_type == "TP" and rec["dist"] <= p50_err:
            folder = os.path.join(debug_dir, "localization_good_p50")
            title = f"Good Localization (P50) - {dataset}_{pc_id} (Err: {rec['dist']:.3f}m)"
            save_path_img = os.path.join(folder, f"{dataset}_{pc_id}_tp_good_{idx}.png")
            # save_crop_debug_plot(coords, intens, center_ref=rec["gt"], gt_center=rec["gt"], pred_center=rec["pred"], error_m=rec["dist"], title_info=title, save_path=save_path_img)
            save_crop_with_patch_debug_plot(
                coords, intens, center_ref=ref_center, gt_center=rec["gt"], pred_center=rec["pred"],
                error_m=rec["dist"], patch_tile=patch_tile, patch_meta=patch_meta, title_info=title, save_path=save_path_img
            )

        # CASE 3: False Positive (Detection Error - Manholes predicted but not present)
        elif m_type == "FP":
            folder = os.path.join(debug_dir, "detection_false_positives")
            title = f"False Positive (Manholes predicted but not present) - {dataset}_{pc_id}"
            save_path_img = os.path.join(folder, f"{dataset}_{pc_id}_fp_{idx}.png")
            # save_crop_debug_plot(coords, intens, center_ref=rec["pred"], gt_center=None, pred_center=rec["pred"], error_m=None, title_info=title, save_path=save_path_img)
            save_crop_with_patch_debug_plot(
                coords, intens, center_ref=ref_center, gt_center=None, pred_center=rec["pred"],
                error_m=None, patch_tile=patch_tile, patch_meta=patch_meta, title_info=title, save_path=save_path_img
            )

        # CASE 4: False Negative (Detection Error - Manholes not detected)
        elif m_type == "FN":
            folder = os.path.join(debug_dir, "detection_false_negatives")
            title = f"False Negative (Manholes not detected) - {dataset}_{pc_id}"
            save_path_img = os.path.join(folder, f"{dataset}_{pc_id}_fn_{idx}.png")
            # save_crop_debug_plot(coords, intens, center_ref=rec["gt"], gt_center=rec["gt"], pred_center=None, error_m=None, title_info=title, save_path=save_path_img)
            save_crop_with_patch_debug_plot(
                coords, intens, center_ref=ref_center, gt_center=rec["gt"], pred_center=None,
                error_m=None, patch_tile=patch_tile, patch_meta=patch_meta, title_info=title, save_path=save_path_img
            )

    print(f"Finish! Plots got saved in '{debug_dir}'.")



# --------------------------
# > Interactive Visualizer <
# --------------------------

def visualize_scene_interactive(
    pcd_path,
    gt_centers,
    pred_centers,
    match_threshold=0.2,
    sphere_radius=0.15,
    is_preprocessed=False,
):
    """
    Alternative interactive visualizer rendering:
      - Point Cloud: Grayscale based on intensity
      - Manhole Points: Highlighted in bright Magenta/Pink (semantic label matching)
      - Green Spheres: All Ground Truth centers
      - Red Spheres: All Predicted centers
    """
    geometries = []

    # Determine semantic label value for manholes based on dataset type
    if "sud-road" in pcd_path.lower():
        target_labels = [1, 255] if is_preprocessed else [3]
    else:
        target_labels = [1, 255] if is_preprocessed else [104002]

    # --- DEBUG PRINT SECTION START ---
    print("\n=================== COORDINATE DEBUG INFO ===================")
    print(f"File Path: {pcd_path}")

    coords = None
    if os.path.exists(pcd_path):
        t_pcd = load_point_cloud_h5(pcd_path)
        coords = t_pcd.point["positions"].numpy()
        min_pcd, max_pcd = coords.min(axis=0), coords.max(axis=0)
        mean_pcd = coords.mean(axis=0)
        print(f"[Point Cloud] Count: {len(coords):,}")
        print(f"  ├─ Min (X, Y, Z):  [{min_pcd[0]:12.3f}, {min_pcd[1]:12.3f}, {min_pcd[2]:12.3f}]")
        print(f"  ├─ Max (X, Y, Z):  [{max_pcd[0]:12.3f}, {max_pcd[1]:12.3f}, {max_pcd[2]:12.3f}]")
        print(f"  └─ Mean (X, Y, Z): [{mean_pcd[0]:12.3f}, {mean_pcd[1]:12.3f}, {mean_pcd[2]:12.3f}]")
    else:
        print(f"[Point Cloud] FILE NOT FOUND: {pcd_path}")

    gt_pts = np.array([(x["x"], x["y"], x["z"]) for x in gt_centers]) if gt_centers else np.empty((0, 3))
    if len(gt_pts) > 0:
        min_gt, max_gt = gt_pts.min(axis=0), gt_pts.max(axis=0)
        mean_gt = gt_pts.mean(axis=0)
        print(f"[Ground Truth] Count: {len(gt_pts)}")
        print(f"  ├─ Min (X, Y, Z):  [{min_gt[0]:12.3f}, {min_gt[1]:12.3f}, {min_gt[2]:12.3f}]")
        print(f"  ├─ Max (X, Y, Z):  [{max_gt[0]:12.3f}, {max_gt[1]:12.3f}, {max_gt[2]:12.3f}]")
        print(f"  └─ Mean (X, Y, Z): [{mean_gt[0]:12.3f}, {mean_gt[1]:12.3f}, {mean_gt[2]:12.3f}]")
    else:
        print("[Ground Truth] Count: 0 (EMPTY)")

    pred_pts = np.array([(x["x"], x["y"], x["z"]) for x in pred_centers]) if pred_centers else np.empty((0, 3))
    if len(pred_pts) > 0:
        min_pred, max_pred = pred_pts.min(axis=0), pred_pts.max(axis=0)
        mean_pred = pred_pts.mean(axis=0)
        print(f"[Predictions] Count: {len(pred_pts)}")
        print(f"  ├─ Min (X, Y, Z):  [{min_pred[0]:12.3f}, {min_pred[1]:12.3f}, {min_pred[2]:12.3f}]")
        print(f"  ├─ Max (X, Y, Z):  [{max_pred[0]:12.3f}, {max_pred[1]:12.3f}, {max_pred[2]:12.3f}]")
        print(f"  └─ Mean (X, Y, Z): [{mean_pred[0]:12.3f}, {mean_pred[1]:12.3f}, {mean_pred[2]:12.3f}]")
    else:
        print("[Predictions] Count: 0 (EMPTY)")

    # Coordinate Offset Check
    if coords is not None and len(gt_pts) > 0:
        gt_dist_to_pcd_mean = np.linalg.norm(mean_gt - mean_pcd)
        print(f"\n[Coordinate Alignment Check]")
        print(f"  └─ Distance between PCD Mean and GT Mean: {gt_dist_to_pcd_mean:.3f} meters")
        if gt_dist_to_pcd_mean > 100.0:
            print("WARNING: Extreme offset detected! Point cloud and GT are maybe in different coordinate systems!")
    print("============================================================\n")
    # --- DEBUG PRINT SECTION END ---

    # 1. Load and process Point Cloud colors
    if coords is not None:
        # Base color: Intensity-based grayscale
        if "intensity" in t_pcd.point:
            intensities = t_pcd.point["intensity"].numpy().flatten()
            intensity_min = intensities.min()
            intensity_range = intensities.max() - intensity_min
            
            if intensity_range > 0:
                norm_intensities = (intensities - intensity_min) / (intensity_range + 1e-8)
            else:
                norm_intensities = np.zeros_like(intensities)

            colors = np.tile(norm_intensities[:, None], (1, 3)) * 0.7  # Dim gray
        else:
            colors = np.full((len(coords), 3), 0.6)

        # Highlight Manhole Points using semantic classes
        if "classes" in t_pcd.point:
            semantics = t_pcd.point["classes"].numpy().flatten()
            manhole_mask = np.isin(semantics, target_labels)
            manhole_count = np.sum(manhole_mask)
            print(f"[Semantic Check] Manhole points found in PCD: {manhole_count:,} / {len(coords):,}")

            # Paint manhole points bright Magenta/Pink [R=1.0, G=0.0, B=1.0]
            colors[manhole_mask] = [1.0, 0.0, 1.0]

        # Explicit legacy creation from NumPy to ensure colors and coords are properly bound
        legacy_pcd = o3d.geometry.PointCloud()
        legacy_pcd.points = o3d.utility.Vector3dVector(coords)
        legacy_pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(legacy_pcd)

    # 2. Add Ground Truth Centers as Green Spheres
    if len(gt_pts) > 0:
        for pt in gt_pts:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sphere.translate(pt)
            sphere.paint_uniform_color([0.0, 1.0, 0.0])  # Pure Green
            sphere.compute_vertex_normals()
            geometries.append(sphere)

    # 3. Add Prediction Centers as Red Spheres
    if len(pred_pts) > 0:
        for pt in pred_pts:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sphere.translate(pt)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Pure Red
            sphere.compute_vertex_normals()
            geometries.append(sphere)

    # Launch Interactive Viewer using Visualizer class to control point rendering size
    print("Visualization Opened...")
    # print("\n--- Visualizer Controls ---")
    # print(" - Left Click + Drag : Rotate")
    # print(" - Shift + Left Click: Pan")
    # print(" - Scroll Wheel       : Zoom")
    # print(" - Press 'Q' or 'ESC' : Next Scene")
    # print("---------------------------\n")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"Direct GT vs Pred Viewer (Green: GT, Red: Pred, Pink: Manhole PCD) - {os.path.basename(pcd_path)}",
        width=1280,
        height=720,
    )

    for geom in geometries:
        vis.add_geometry(geom)

    # Adjust point size & dark background for contrast
    render_option = vis.get_render_option()
    render_option.point_size = 3.0
    render_option.background_color = np.array([0.1, 0.1, 0.1])

    vis.run()
    vis.destroy_window()

    # pc = ps.register_point_cloud("my_cloud", points)
    # pc.add_scalar_quantity("Class Labels", intensity, cmap="jet")
    # pc.add_scalar_quantity("Semantics", semantics, cmap="viridis")

    # # 4. Look at the data
    # ps.show()


# --------------
# > Evaluation <
# --------------
class CenterEvaluator:
    def __init__(self, thresholds=[0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]):
        self.thresholds = thresholds
        self.tp_distances = []
        self.all_distances = []
        self.results = {t: {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}
        self.matches_record = []  # for detailed debugging

    def update(self, gt_centers, pred_centers, dataset="", pc_id=""):
        
        if len(gt_centers) == 0 and len(pred_centers) == 0:
            return
        elif len(gt_centers) == 0:
            # only FPs
            for t in self.thresholds: 
                self.results[t]["fp"] += len(pred_centers)
            for j, p in enumerate(pred_centers):
                self.matches_record.append({
                    "type": "FP", "dataset": dataset, "pc_id": pc_id,
                    "gt": None, "pred": p, "dist": None
                })
            return
        elif len(pred_centers) == 0:
            # only FNs
            for t in self.thresholds:
                self.results[t]["fn"] += len(gt_centers)
            for i, g in enumerate(gt_centers):
                self.matches_record.append({
                    "type": "FN", "dataset": dataset, "pc_id": pc_id,
                    "gt": g, "pred": None, "dist": None
                })
            return
        
        gt_arr = np.array([(x['x'], x['y'], x['z']) for x in gt_centers])
        pred_arr = np.array([(x['x'], x['y'], x['z']) for x in pred_centers])
        
        dist_matrix = cdist(gt_arr, pred_arr, metric='euclidean')
        row_indeces, col_indeces = linear_sum_assignment(dist_matrix)
        # matches = dist_matrix[row_indeces, col_indeces]


        # max_t = max(self.thresholds)
        # valid_tp_distances = matches[matches <= max_t]
        # self.tp_distances.extend(valid_tp_distances.tolist())
        # self.all_distances.extend(matches.tolist())

        # for t in self.thresholds:
        #     tp_mask = matches <= t
        #     tp = np.sum(tp_mask)
        #     self.results[t]["tp"] += tp
        #     self.results[t]["fp"] += len(pred_centers) - tp
        #     self.results[t]["fn"] += len(gt_centers) - tp

        matched_gt_idx = set()
        matched_pred_idx = set()

        max_match_t = max(self.thresholds)  # e.g., 0.8m als Obergrenze für TP
        for r, c in zip(row_indeces, col_indeces):
            d = dist_matrix[r, c]
            if d <= max_match_t:
                matched_gt_idx.add(r)
                matched_pred_idx.add(c)
                self.tp_distances.append(d)
                self.matches_record.append({
                    "type": "TP", "dataset": dataset, "pc_id": pc_id,
                    "gt": gt_centers[r], "pred": pred_centers[c], "dist": d
                })

        # Unmatched GT -> FN
        for i, g in enumerate(gt_centers):
            if i not in matched_gt_idx:
                self.matches_record.append({
                    "type": "FN", "dataset": dataset, "pc_id": pc_id,
                    "gt": g, "pred": None, "dist": None
                })

        # Unmatched Pred -> FP
        for j, p in enumerate(pred_centers):
            if j not in matched_pred_idx:
                self.matches_record.append({
                    "type": "FP", "dataset": dataset, "pc_id": pc_id,
                    "gt": None, "pred": p, "dist": None
                })

        for t in self.thresholds:
            tp_mask = dist_matrix[row_indeces, col_indeces] <= t
            tp = np.sum(tp_mask)
            self.results[t]["tp"] += tp
            self.results[t]["fp"] += len(pred_centers) - tp
            self.results[t]["fn"] += len(gt_centers) - tp

    # FIXME : compare with G* here

    def get_summary(self):
        summary = {}
        if self.tp_distances:
            summary["mean_localization_error_m"] = float(np.mean(self.tp_distances))
            summary["std_localization_error_m"] = float(np.std(self.tp_distances))
            summary["p50_error_m"] = float(np.percentile(self.tp_distances, 50))
            summary["p95_error_m"] = float(np.percentile(self.tp_distances, 95))
        else:
            summary.update({"mean_localization_error_m": 0.0, "std_localization_error_m": 0.0, "p50_error_m": 0.0, "p95_error_m": 0.0})

        if self.all_distances:
            summary["all_mean_localization_error_m"] = float(np.mean(self.all_distances))
            summary["all_std_localization_error_m"] = float(np.std(self.all_distances))
            summary["all_p50_error_m"] = float(np.percentile(self.all_distances, 50))
            summary["all_p95_error_m"] = float(np.percentile(self.all_distances, 95))
        else:
            summary.update({"all_mean_localization_error_m": 0.0, "all_std_localization_error_m": 0.0, "all_p50_error_m": 0.0, "all_p95_error_m": 0.0})

        precisions, recalls, f1s = [], [], []
        for t in self.thresholds:
            cur = self.results[t]
            tp, fp, fn = cur["tp"], cur["fp"], cur["fn"]

            precision = tp / max(tp + fp, 1e-8)
            recall = tp / max(tp + fn, 1e-8)
            f1 = (2 * precision * recall) / max(precision + recall, 1e-8)

            summary[f"precision_at_{t}m"] = float(precision)
            summary[f"recall_at_{t}m"] = float(recall)
            summary[f"f1_at_{t}m"] = float(f1)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        recalls_arr, precisions_arr = np.array(recalls), np.array(precisions)
        sort_idx = np.argsort(recalls_arr)
        map_auc = auc(recalls_arr[sort_idx], precisions_arr[sort_idx]) if len(self.thresholds) > 1 else precisions[0]

        summary["mAP"] = float(map_auc)
        summary["mean_precision"] = float(np.mean(precisions))
        summary["mAR"] = float(np.mean(recalls))
        summary["mAF1"] = float(np.mean(f1s))

        return summary

    def print_and_save_benchmark(self, exp_name, should_print=True, save_path=None):
        summary = self.get_summary()
        output = f"--- Benchmark Summary: {exp_name} ---\n"
        output += f"Mean Localization Error (L2): {summary['mean_localization_error_m']:.4f} m\n"
        output += f"Median (P50) Error:     {summary['p50_error_m']:.4f} m\n"
        output += f"P95 Error Bound:        {summary['p95_error_m']:.4f} m\n"
        output += f"-------------------------\n"
        output += f"mAP (PR-AUC):           {summary['mAP']:.4f}\n"
        output += f"mAR (Mean Recall):      {summary['mAR']:.4f}\n"
        output += f"mAF1 (Mean F1):         {summary['mAF1']:.4f}\n"
        output += f"-------------------------\n"

        for t in self.thresholds:
            p = summary[f"precision_at_{t}m"]
            r = summary[f"recall_at_{t}m"]
            f1 = summary[f"f1_at_{t}m"]
            output += f"Threshold {t:4.2f}m -> P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f}\n"

        if should_print:
            print(output)

        if save_path:
            full_save_path = os.path.join(save_path, f"mcreval_{exp_name}.txt")
            os.makedirs(save_path, exist_ok=True)
            with open(full_save_path, "w") as file_:
                file_.write(output)

        return output


# ----------
# > Helper <
# ----------
def count_double_entry(predictions):
    found_entries = set()
    counter = 0
    for cur_entry in predictions:
        key = (cur_entry["dataset"], cur_entry["pointcloud-id"])
        if key in found_entries:
            counter += 1
        found_entries.add(key)
    return counter



# --------
# > Main <
# --------
def main():
    with open("./config.json") as f:
        cfg = json.load(f)

    save_path = cfg.get("save_path", "./results")
    exp_name = cfg.get("exp_name", "experiment")
    thresholds = cfg.get("distance_threshold", [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8])
    
    # Configuration flags for Interactive Mode
    interactive_mode = cfg.get("interactive_mode", False)
    create_debug_plots = cfg.get("create_debug_plots", False)
    data_root_dir = cfg.get("data_root_dir", "./data")  # Root folder containing .h5 files
    patch_dir = cfg.get("patch_dir", "./data/preprocessed")

    evaluator = CenterEvaluator(thresholds=thresholds)

    with open(cfg["ground_truth_path"], "r", encoding="utf-8") as json_file:
        gt = json.load(json_file)

    with open(cfg["prediction_path"], "r", encoding="utf-8") as json_file:
        pred = json.load(json_file)

    if count_double_entry(gt) > 0 or count_double_entry(pred) > 0:
        raise ValueError("Duplicate pointcloud entries found in ground truth or predictions!")

    # Evaluation Loop
    matched_pred_keys = set()
    for cur_entry in gt:
        cur_gt_pc_id = cur_entry["pointcloud-id"]
        cur_gt_dataset = cur_entry["dataset"]
        cur_gt_centers = cur_entry.get("centers", [])

        # Match predictions
        cur_pred_centers = []
        for cur_pred_entry in pred:
            if cur_gt_dataset == cur_pred_entry["dataset"] and cur_gt_pc_id == cur_pred_entry["pointcloud-id"]:
                cur_pred_centers = cur_pred_entry.get("centers", [])
                matched_pred_keys.add((cur_gt_dataset, cur_gt_pc_id))
                break

        evaluator.update(
            gt_centers=cur_gt_centers, 
            pred_centers=cur_pred_centers,
            dataset=cur_gt_dataset,
            pc_id=cur_gt_pc_id
        )

        # Interactive Mode Trigger
        if interactive_mode:
            pcd_path = os.path.join(data_root_dir, f"{cur_gt_pc_id:04}.h5")
            print(f"Displaying Scene: {cur_gt_dataset} / {cur_gt_pc_id}")
            visualize_scene_interactive(
                pcd_path=pcd_path,
                gt_centers=cur_gt_centers,
                pred_centers=cur_pred_centers,
                match_threshold=0.2  # Threshold used to color TPs in blue
            )

    evaluator.print_and_save_benchmark(exp_name=exp_name, should_print=True, save_path=save_path)

    if create_debug_plots:
        debug_dir = os.path.join(save_path, f"debug_plots_{exp_name}")
        os.makedirs(debug_dir, exist_ok=True)
        shutil.rmtree(debug_dir)
        os.makedirs(debug_dir)
        debug_plots(evaluator, data_root_dir=data_root_dir, debug_dir=debug_dir, patch_dir=patch_dir)



if __name__ == "__main__":
    main()





    