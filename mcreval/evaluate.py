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
        if len(gt_centers) == 0 and len(pred_centers) == 0:
            # for t in self.thresholds: 
            #     self.results[t]["tn"] += len(pred_centers)
            return
        elif len(gt_centers) == 0:
            for t in self.thresholds: 
                self.results[t]["fp"] += len(pred_centers)
            return
        elif len(pred_centers) == 0:
            for t in self.thresholds:
                self.results[t]["fn"] += len(gt_centers)
            return
        
        # optimal matching
        # array([[7210.93457031, 3649.18530273,   13.93712902]])
        gt_centers = np.array([(x['x'], x['y'], x['z']) for x in gt_centers])  # [gt_centers['x'], gt_centers['y'], gt_centers['z']]
        # array([[7178.71355081, 3739.8159719 ,   14.07421112]])
        pred_centers = np.array([(x['x'], x['y'], x['z']) for x in pred_centers])  # [pred_centers['x'], pred_centers['y'], pred_centers['z']]
        dist_matrix = cdist(gt_centers, pred_centers, metric='euclidean')
        row_indeces, col_indeces = linear_sum_assignment(dist_matrix)
        matches = dist_matrix[row_indeces, col_indeces]

        max_t = max(self.thresholds)
        valid_tp_distances = matches[matches <= max_t]
        self.tp_distances.extend(valid_tp_distances.tolist())
        # # add to tp distances no matter how far they are 
        # self.tp_distances.extend(matches.tolist())

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
        # localization error (only on valid TPs)
        if self.tp_distances:
            summary["mean_localization_error_m"] = float(np.mean(self.tp_distances))
            summary["std_localization_error_m"] = float(np.std(self.tp_distances))
            summary["p50_error_m"] = float(np.percentile(self.tp_distances, 50))
            summary["p95_error_m"] = float(np.percentile(self.tp_distances, 95))
        else:
            summary["mean_localization_error_m"] = 0.0
            summary["std_localization_error_m"] = 0.0
            summary["p50_error_m"] = 0.0
            summary["p95_error_m"] = 0.0
        # typical performance -> mean error -> median

        # ap_list = []
        # ar_list = []
        precisions = []
        recalls = []
        f1s = []

        # total_tp = 0
        # total_fp = 0
        # total_fn = 0
        for t in self.thresholds:
            cur_result = self.results[t]
            tp = cur_result["tp"]
            fp = cur_result["fp"]
            fn = cur_result["fn"]

            # Calculate metrics for current threshold with zero-division safety
            precision = tp / max(tp + fp, 1e-8)
            recall = tp / max(tp + fn, 1e-8)
            f1 = (2 * precision * recall) / max(precision + recall, 1e-8)

            summary[f"precision_at_{t}m"] = float(precision)
            summary[f"recall_at_{t}m"] = float(recall)
            summary[f"f1_at_{t}m"] = float(f1)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            # total_tp += cur_result["tp"]
            # total_fp += cur_result["fp"]
            # total_fn += cur_result["fn"]

        recalls_arr = np.array(recalls)
        precisions_arr = np.array(precisions)

        sort_idx = np.argsort(recalls_arr)
        map_auc = auc(recalls_arr[sort_idx], precisions_arr[sort_idx]) if len(self.thresholds) > 1 else precisions[0]

        summary["mAP"] = float(map_auc)                  # PR-AUC across distance thresholds
        summary["mean_precision"] = float(np.mean(precisions)) # Average Precision across thresholds
        summary["mAR"] = float(np.mean(recalls))          # Mean Recall across thresholds
        summary["mAF1"] = float(np.mean(f1s))             # Mean F1 across thresholds
            
        # total_precision = total_tp / max(total_tp + total_fp, 1)
        # total_recall = total_tp / max(total_tp + total_fn, 1)
        # summary["mAP"] = total_precision
        # summary["mAR"] = total_recall
        # summary[f"mAF1"] = (2 * total_precision * total_recall) / (total_precision + total_recall)

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
            
        output += "-------------------------"

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
    pred_double_counts = count_double_entry(pred)
    if pred_double_counts > 0:
        raise ValueError(f"Predictions have {pred_double_counts} double entries!")

    # make loop
    matched_pred_keys = set()
    for cur_entry in gt:
        cur_gt_pc_id = cur_entry["pointcloud-id"]
        cur_gt_dataset = cur_entry["dataset"]
        cur_gt_centers = cur_entry["centers"] if "centers" in cur_entry else []

        # find prediction entry
        found_entry_match = False
        for cur_pred_entry in pred:
            cur_pred_pc_id = cur_pred_entry["pointcloud-id"]
            cur_pred_dataset = cur_pred_entry["dataset"]
            cur_pred_centers = cur_pred_entry["centers"] if "centers" in cur_pred_entry else []

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



























