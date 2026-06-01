# -----------
# > Imports <
# -----------
import json

import numpy as np




# ----------
# > Helper <
# ----------
def compute_distance(p1, p2, mode="xy"):
    assert len(mode.replace("x", "").replace("y", "").replace("z", "")) == 0
    
    distances = 0.0

    for cur_dim in mode:
        distances += (p1[cur_dim] - p2[cur_dim])**2

    return np.sqrt(distances)



def compare_centers(gt_centers, pred_centers, threshold=0.5, distance_mode="xy"):
    matched_gt = set()

    distances = []
    total_error = 0
    tp = 0
    fp = 0

    for pred_center in pred_centers:

        best_gt_idx = None
        best_dist = threshold

        for gt_idx, gt_center in enumerate(gt_centers):
            if gt_idx in matched_gt:
                continue

            dist = compute_distance(pred_center, gt_center, mode=distance_mode)

            if dist < best_dist:
                best_dist = dist
                best_gt_idx = gt_idx

        if best_gt_idx is not None:
            matched_gt.add(best_gt_idx)
            tp += 1
            total_error += best_dist
            distances.append(best_dist)
        else:
            fp += 1

    fn = len(gt_centers) - len(matched_gt)

    return tp, fp, fn, total_error, distances


# --------
# > Main <
# --------
def main():
    with open("./config.json") as f:
        cfg = json.load(f)

    gt_json_path = cfg["ground_truth_path"]
    pred_json_path = cfg["prediction_path"]
    thresholds = cfg["distance_threshold"]
    distance_mode = cfg["distance_mode"]

    # load gt and pred
    with open(gt_json_path, "r", encoding="utf-8") as json_file:
        gt = json.load(json_file)

    with open(pred_json_path, "r", encoding="utf-8") as json_file:
        pred = json.load(json_file)

    # go through every file
    for gt_dataset in gt:
        # init accumulators
        total_tp = [0] * len(thresholds)
        total_fp = [0] * len(thresholds)
        total_fn = [0] * len(thresholds)
        total_error = [0] * len(thresholds)
        all_distances = []

        dataset_name = gt_dataset["dataset"]

        # get the same dataset from prediction
        pred_dataset = None
        for cur_pred_dataset in pred:
            if cur_pred_dataset["dataset"] == dataset_name:
                pred_dataset = cur_pred_dataset
                break

        if pred_dataset is None:
            # or maybe just skip and print?
            raise ValueError(f"No prediction for '{dataset_name}'-dataset found.")
        
        # go through every point cloud
        for gt_point_cloud in gt_dataset["data"]:
            pc_id = gt_point_cloud["pointcloud-id"]
            gt_centers = gt_point_cloud["centers"]

            # search for the same pc in predictions
            pred_point_cloud = None
            for cur_pred_point_cloud in pred_dataset["data"]:
                if cur_pred_point_cloud["pointcloud-id"] == pc_id:
                    pred_point_cloud = cur_pred_point_cloud
                    break

            if pred_point_cloud is None:
                # raise ValueError(f"No prediction for '{pc_id}'-point cloud found.")
                print(f"[WARNING] no prediction for {pc_id}")

                for i in range(len(thresholds)):
                    total_fn[i] += len(gt_centers)
                continue

            pred_centers = pred_point_cloud["centers"]

            # compare the predictions
            for idx, threshold in enumerate(thresholds):
                tp, fp, fn, error, distances = compare_centers(gt_centers=gt_centers,
                                                               pred_centers=pred_centers,
                                                               threshold=threshold,
                                                               distance_mode=distance_mode)
                total_tp[idx] += tp
                total_fp[idx] += fp
                total_fn[idx] += fn
                total_error[idx] += error
                mean_error = total_error[idx] / max(total_tp[idx], 1)
                all_distances += distances


        # compute additional errors and result string
        result = "\n\n--- Evaluation Results ---\n"
        result += f"\n DATASET: {dataset_name}"
        for i, threshold in enumerate(thresholds):

            precision = (
                total_tp[i] /
                max(total_tp[i] + total_fp[i], 1)
            )

            recall = (
                total_tp[i] /
                max(total_tp[i] + total_fn[i], 1)
            )

            f1 = (
                2 * precision * recall /
                max(precision + recall, 1e-8)
            )

            mean_error = (
                total_error[i] /
                max(total_tp[i], 1)
            )

            all_distances = np.array(all_distances)

            result += f"\n -> Threshold: {threshold}\n"
            result += f"\nF1: {f1}"
            result += f"\nPrecision: {precision}"
            result += f"\nRecall: {recall}"
            result += f"\nMean Error: {mean_error}"
            result += f"\nMean Distances: {all_distances.mean()}"
            result += f"\nMedian Distances: {all_distances.median()}"
            result += f"\nMax Distances: {all_distances.max()}"
            result += f"\nMin Distances: {all_distances.min()}"
            result += f"\nStd Distances: {all_distances.std()}"
            result += f"\nOthers:"
            result += f"\n    - TP: {total_tp[i]}"
            result += f"\n    - FP: {total_fp[i]}"
            result += f"\n    - FN: {total_fn[i]}"
            result += f"\n    - Total Error: {total_error[i]}\n\n{'-'*36}\n\n"

    # FIXME -> maybe add stats for every dataset seperate?
    # any other measurements?
    # -> I want to evaluate different methods, I should have very good evaluation metrics!

    print(result)



if __name__ == "__main__":
    main()
























