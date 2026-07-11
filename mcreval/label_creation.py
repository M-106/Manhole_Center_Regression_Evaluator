"""
Legacy Code

Due to changes in my json label format, this is not longer needed.

"""

# -----------
# > Imports <
# -----------
from __future__ import annotations

import os
import json
import time
from typing import Any



# ------------------
# > Label Creation <
# ------------------

def create_labels(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert a flat list of predictions into the evaluation JSON format.

    Parameters
    ----------
    predictions : list of dict
        Each prediction must contain:
            - dataset (str)
            - pointcloud_id (str or int)
            - centers (list of {"x", "y", "z"})

    Returns
    -------
    list
        Label structure expected by the evaluation.
    """

    grouped = {}

    for pred in predictions:
        dataset = pred["dataset"]

        if dataset not in grouped:
            grouped[dataset] = []

        grouped[dataset].append(
            {
                "pointcloud-id": str(pred["pointcloud_id"]),
                "centers": pred.get("centers", []),
            }
        )

    return [
        {
            "dataset": dataset,
            "data": data,
        }
        for dataset, data in grouped.items()
    ]


def save_labels(
        predictions: list[dict[str, Any]],
        output_dir: str,
        exp_name: str,
        indent: int = 4,
    ) -> None:
    """
    Create the evaluation JSON and save it.

    Parameters
    ----------
    predictions
        Input predictions.
    output_file
        Path to the output json file.
    indent
        JSON indentation.
    """

    labels = create_labels(predictions)

    os.path.mkdir(output_dir, exist_ok=True)
    output_file = f"MCR-Eval_{time.strftime("%Y-%m-%d_%H-%M-%S")}_{exp_name}.json"

    output_path = os.path.join(output_dir, output_file)

    with open(output_path, "w", encoding="utf-8") as file_:
        json.dump(labels, file_, indent=indent)

    print(f"Saved labels to {output_file}")







