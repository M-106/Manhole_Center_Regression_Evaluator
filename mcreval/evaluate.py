# -----------
# > Imports <
# -----------
import os
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import auc



# ----------
# > Helper <
# ----------



# --------------
# > Evaluation <
# --------------
class CenterEvaluator:
    def __init__(self, thresholds=[0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]):
        self.thresholds = thresholds
        self.tp_distances = []
        self.results = {
            t: {"tp":0, "fp":0, "fn":0} for t in thresholds
        }

    def update(self, gt_centers, pred_centers):
        if len(gt_centers) == 0:
            for t in self.thresholds: 
                self.results[t]["fp"] += len(pred_centers)
            return
        if len(pred_centers) == 0:
            for t in self.thresholds:
                self.results[t]["fn"] += len(gt_centers)
            return
        
        # optimal matching
        dist_matrix = cdist(np.array(gt_centers), np.array(pred_centers), metric='euclidean')
        row_indeces, col_indeces = linear_sum_assignment(dist_matrix)
        matches = dist_matrix[row_indeces, col_indeces]

        self.tp_distances.extend(matches.tolist())

        for t in self.thresholds:
            tp_mask = matches <= t
            tp = np.sum(tp_mask)
            self.results[t]["tp"] += tp
            self.results[t]["fp"] += len(pred_centers) - tp
            self.results[t]["fn"] += len(gt_centers) - tp
            # self.results[t]["dist"].extend(matches[tp_mask].tolist())



    def get_summary(self):
        """
        Mean Localization Error (MLE): 
        Mean euclidean distance 
        (in meters/centimeters) for only true positives, 
        how precise is the predictions if foudn the manhole.

        Distance Error Distribution (Quantile):
        95% of all correct found manholes are lying in X m/cm precision

        mAP at varying thresholds:
        How precise and how many manholes got detected with which distance is still counting as a match. 
        """
        summary = {}
        # MLE => Mean Localization Error
        mle = np.mean(self.tp_distances) if self.tp_distances else 0
        summary["mean_localization_error_m"] = float(mle)
        summary["std_localization_error_m"] = float(np.std(self.tp_distances)) if self.tp_distances else 0
        summary["p95_error_m"] = float(np.percentile(self.tp_distances, 95)) if self.tp_distances else 0
        summary["p50_error_m"] = float(np.percentile(self.tp_distances, 50)) if self.tp_distances else 0
        # typical performance -> mean error -> median

        # ap_list = []
        # ar_list = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for t in self.thresholds:
            cur_result = self.results[t]
            precision = cur_result["tp"] / max(cur_result["tp"] + cur_result["fp"], 1)
            recall = cur_result["tp"] / max(cur_result["tp"] + cur_result["fn"], 1)
            summary[f"precision_at_{t}m"] = precision
            summary[f"recall_at_{t}m"] = recall
            summary[f"f1_at_{t}m"] = (2 * precision * recall) / (precision + recall)
            # ap_list.append(precision)
            # ar_list.append(recall)
            total_tp += cur_result["tp"]
            total_fp += cur_result["fp"]
            total_fn += cur_result["fn"]

        # ar_list = np.array(ar_list)
        # ap_list = np.array(ap_list)
        # idx = np.argsort(ar_list)
        # mAP = auc(np.array(ar_list)[idx], np.array(ap_list)[idx])
        # summary["mAP"] = mAP    # np.mean(ap_list)
        # summary["mAR"] = np.mean(ar_list)
        total_precision = total_tp / max(total_tp + total_fp, 1)
        total_recall = total_tp / max(total_tp + total_fn, 1)
        summary["mAP"] = total_precision
        summary["mAR"] = total_recall
        summary[f"mAF1"] = (2 * total_precision * total_recall) / (total_precision + total_recall)

        return summary



    def print_and_save_benchmark(self, exp_name, should_print=True, save_path=None):
        distances = np.array(self.tp_distances)

        output = "--- Benchmark Summary ---"
        output += f"\n  From: {exp_name}"
        output += f"\n\nMean Localization Error: {distances.mean():.4f} m"
        output += f"\nMedian Localization Error: {np.median(distances):.4f} m"
        output += f"\nP95 Error Bound: {np.percentile(distances, 95):.4f} m"
        
        recalls = [res["tp"] / max(res["tp"] + res["fn"], 1) for res in self.results.values()]
        output += f"\nmAR: {np.mean(recalls):.4f}"

        precisions = [res["tp"] / max(res["tp"] + res["fp"], 1) for res in self.results.values()]
        # mAP = np.mean(precisions)
        idx = np.argsort(recalls)
        mAP = auc(np.array(recalls)[idx], np.array(precisions)[idx])
        output += f"\nmAP: {mAP:.4f}"

        # F1
        best_idx = np.argmin(np.abs(np.array(self.thresholds) - 0.05))
        f1_at_005m = 2*(precisions[best_idx] * recalls[best_idx]) / (precisions[best_idx] + recalls[best_idx] + 1e-8)
        output += f"\nF1 {self.thresholds[best_idx]}: {f1_at_005m:.4f}"

        output += "\n-------------------------"

        if should_print:
            print(output)

        if save_path:
            full_save_path = os.path.join(save_path, f"mcreval_{exp_name}.txt")
            os.makedirs(save_path, exist_ok=True)
            with open(full_save_path, "w") as file_:
                file_.write(output)
            print(f"Benchmark results saved under: '{full_save_path}'")

        return output

    
    def plot_benchmark(self, show_plot=True, save_path=None, exp_name=""):
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

        # Plot 1: PR Curve
        # FIXME -> add F1 scores?
        precisions = [res["tp"] / max(res["tp"]+res["fp"], 1) for res in self.results.values()]
        recalls = [res["tp"] / max(res["tp"]+res["fn"], 1) for res in self.results.values()]

        # AUC = Area Under Curve
        ax[0].plot(recalls, precisions, marker='o', linestyle="-", linewidth=2, color='#2c3e50', label=f"PR-AUC: {auc(recalls, precisions):.3f}")
        # add thresholds as description
        for i, t in enumerate(self.thresholds):
            if i % 2 == 0:  # do not add text for every point
                ax[0].annotate(f"{t:.2f}m", (recalls[i], precisions[i]),
                               xytext=(5,5), textcoords='offset points',
                               fontsize=8)
        ax[0].set_title(f"Precision-Recall Curve (PR-AUC/mAP: {auc(recalls, precisions):.3f})")
        ax[0].set_xlabel("Recall")
        ax[0].set_ylabel("Precision")
        # ax[0].legend()
        ax[0].grid()

        # Plot 2: Error Histogram
        bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]
        ax[1].hist(self.tp_distances, bins=bins, color="#3498db", edgecolor="black", alpha=0.7)
        ax[1].set_title("Error Histogram/Distribution (TP Distances)")
        ax[1].set_xlabel("Distance to the center (m)")
        ax[1].set_ylabel("Amount")
        ax[1].grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()

        if save_path:
            full_save_path = os.path.join(save_path, f"mcreval_pr_curve_and_error_hist_{exp_name}.png")
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(full_save_path, dpi=300)
            print(f"Plot saved under: '{full_save_path}'")

        if show_plot:
            plt.show()

        plt.close(fig)



# ----------
# > Helper <
# ----------
def count_double_entry(predictions):
    found_entries = set()
    counter = 0
    for cur_entry in predictions:
        cur_pc_id = cur_entry["pointcloud-id"]
        cur_dataset = cur_entry["dataset"]

        if (cur_dataset, cur_pc_id) in found_entries:
            counter += 1 

        found_entries.add((cur_dataset, cur_pc_id))
    return counter



# --------
# > Main <
# --------
def main():
    # load config
    with open("./config.json") as f:
        cfg = json.load(f)

    save_path = cfg["save_path"]
    exp_name = cfg["exp_name"]

    # maybe thresholds via
    thresholds = cfg["distance_threshold"]
    evaluator = CenterEvaluator(thresholds=thresholds)

    # load gt and preds
    gt_json_path = cfg["ground_truth_path"]
    pred_json_path = cfg["prediction_path"]

    with open(gt_json_path, "r", encoding="utf-8") as json_file:
        gt = json.load(json_file)

    with open(pred_json_path, "r", encoding="utf-8") as json_file:
        pred = json.load(json_file)

    # check if there is only one entry for one dataset x point cloud scene
    gt_double_counts = count_double_entry(gt)
    if gt_double_counts > 0:
        raise ValueError(f"Ground Truth have {gt_double_counts} double entries!")
    pred_double_counts = count_double_entry(gt)
    if pred_double_counts > 0:
        raise ValueError(f"Predictions have {pred_double_counts} double entries!")

    # make loop
    matched_pred_keys = set()
    for cur_entry in gt:
        cur_gt_pc_id = cur_entry["pointcloud-id"]
        cur_gt_dataset = cur_entry["dataset"]
        cur_gt_centers = cur_entry["centers"] if hasattr(cur_entry, "centers") else []

        # find prediction entry
        found_entry_match = False
        for cur_pred_entry in pred:
            cur_pred_pc_id = cur_pred_entry["pointcloud-id"]
            cur_pred_dataset = cur_pred_entry["dataset"]
            cur_pred_centers = cur_pred_entry["centers"] if hasattr(cur_pred_entry, "centers") else []

            if cur_gt_dataset == cur_pred_dataset and \
               cur_gt_pc_id == cur_pred_pc_id:
                evaluator.update(
                    gt_centers=cur_gt_centers,
                    pred_centers=cur_pred_centers
                )
                matched_pred_keys.add((cur_pred_dataset, cur_pred_pc_id))
                found_entry_match = True
                break
        
        if not found_entry_match:
            evaluator.update(gt_centers=cur_gt_centers, pred_centers=[])

    # finding entries from prediction which are not in gt?
    for cur_pred_entry in pred:
        pred_key = (cur_pred_entry["dataset"], cur_pred_entry["pointcloud-id"])
        if pred_key not in matched_pred_keys:
            evaluator.update(gt_centers=[], pred_centers=cur_pred_entry.get("centers", []))

    # eval results / summary
    evaluator.print_and_save_benchmark(exp_name=exp_name, should_print=True, save_path=save_path)
    evaluator.plot_benchmark(show_plot=True, save_path=save_path, exp_name=exp_name)



if __name__ == "__main__":
    main()


# ----------
# > Helper <
# ----------
# def compute_distance(p1, p2, mode="xy"):
#     assert len(mode.replace("x", "").replace("y", "").replace("z", "")) == 0
    
#     distances = 0.0

#     for cur_dim in mode:
#         distances += (p1[cur_dim] - p2[cur_dim])**2

#     return np.sqrt(distances)



# def compare_centers(gt_centers, pred_centers, threshold=0.5, distance_mode="xy"):
#     matched_gt = set()

#     distances = []
#     total_error = 0
#     tp = 0
#     fp = 0

#     for pred_center in pred_centers:

#         best_gt_idx = None
#         best_dist = threshold

#         for gt_idx, gt_center in enumerate(gt_centers):
#             if gt_idx in matched_gt:
#                 continue

#             dist = compute_distance(pred_center, gt_center, mode=distance_mode)

#             if dist < best_dist:
#                 best_dist = dist
#                 best_gt_idx = gt_idx

#         if best_gt_idx is not None:
#             matched_gt.add(best_gt_idx)
#             tp += 1
#             total_error += best_dist
#             distances.append(best_dist)
#         else:
#             fp += 1

#     fn = len(gt_centers) - len(matched_gt)

#     return tp, fp, fn, total_error, distances


# # --------
# # > Main <
# # --------
# def main():
#     with open("./config.json") as f:
#         cfg = json.load(f)

#     gt_json_path = cfg["ground_truth_path"]
#     pred_json_path = cfg["prediction_path"]
#     thresholds = cfg["distance_threshold"]
#     distance_mode = cfg["distance_mode"]

#     # load gt and pred
#     with open(gt_json_path, "r", encoding="utf-8") as json_file:
#         gt = json.load(json_file)

#     with open(pred_json_path, "r", encoding="utf-8") as json_file:
#         pred = json.load(json_file)

#     # go through every file
#     for gt_dataset in gt:
#         # init accumulators
#         total_tp = [0] * len(thresholds)
#         total_fp = [0] * len(thresholds)
#         total_fn = [0] * len(thresholds)
#         total_error = [0] * len(thresholds)
#         all_distances = []

#         dataset_name = gt_dataset["dataset"]

#         # get the same dataset from prediction
#         pred_dataset = None
#         for cur_pred_dataset in pred:
#             if cur_pred_dataset["dataset"] == dataset_name:
#                 pred_dataset = cur_pred_dataset
#                 break

#         if pred_dataset is None:
#             # or maybe just skip and print?
#             raise ValueError(f"No prediction for '{dataset_name}'-dataset found.")
        
#         # go through every point cloud
#         for gt_point_cloud in gt_dataset["data"]:
#             pc_id = gt_point_cloud["pointcloud-id"]
#             gt_centers = gt_point_cloud["centers"]

#             # search for the same pc in predictions
#             pred_point_cloud = None
#             for cur_pred_point_cloud in pred_dataset["data"]:
#                 if cur_pred_point_cloud["pointcloud-id"] == pc_id:
#                     pred_point_cloud = cur_pred_point_cloud
#                     break

#             if pred_point_cloud is None:
#                 # raise ValueError(f"No prediction for '{pc_id}'-point cloud found.")
#                 print(f"[WARNING] no prediction for {pc_id}")

#                 for i in range(len(thresholds)):
#                     total_fn[i] += len(gt_centers)
#                 continue

#             pred_centers = pred_point_cloud["centers"]

#             # compare the predictions
#             for idx, threshold in enumerate(thresholds):
#                 tp, fp, fn, error, distances = compare_centers(gt_centers=gt_centers,
#                                                                pred_centers=pred_centers,
#                                                                threshold=threshold,
#                                                                distance_mode=distance_mode)
#                 total_tp[idx] += tp
#                 total_fp[idx] += fp
#                 total_fn[idx] += fn
#                 total_error[idx] += error
#                 mean_error = total_error[idx] / max(total_tp[idx], 1)
#                 all_distances += distances


#         # compute additional errors and result string
#         result = "\n\n--- Evaluation Results ---\n"
#         result += f"\n DATASET: {dataset_name}"
#         for i, threshold in enumerate(thresholds):

#             precision = (
#                 total_tp[i] /
#                 max(total_tp[i] + total_fp[i], 1)
#             )

#             recall = (
#                 total_tp[i] /
#                 max(total_tp[i] + total_fn[i], 1)
#             )

#             f1 = (
#                 2 * precision * recall /
#                 max(precision + recall, 1e-8)
#             )

#             mean_error = (
#                 total_error[i] /
#                 max(total_tp[i], 1)
#             )

#             all_distances = np.array(all_distances)

#             result += f"\n -> Threshold: {threshold}\n"
#             result += f"\nF1: {f1}"
#             result += f"\nPrecision: {precision}"
#             result += f"\nRecall: {recall}"
#             result += f"\nMean Error: {mean_error}"
#             result += f"\nMean Distances: {all_distances.mean()}"
#             result += f"\nMedian Distances: {all_distances.median()}"
#             result += f"\nMax Distances: {all_distances.max()}"
#             result += f"\nMin Distances: {all_distances.min()}"
#             result += f"\nStd Distances: {all_distances.std()}"
#             result += f"\nOthers:"
#             result += f"\n    - TP: {total_tp[i]}"
#             result += f"\n    - FP: {total_fp[i]}"
#             result += f"\n    - FN: {total_fn[i]}"
#             result += f"\n    - Total Error: {total_error[i]}\n\n{'-'*36}\n\n"

#     # FIXME -> maybe add stats for every dataset seperate?
#     # any other measurements?
#     # -> I want to evaluate different methods, I should have very good evaluation metrics!

#     print(result)


























