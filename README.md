# Manhole_Prediction_Evaluator

Evaluation Repo to eval manhole center prediction.


<br><br>

---
### Installation & Usage

1. Prepare Env
    1. Install Anaconda
    2. Open Anaconda Prompt
    3. Create env (you might to install another pytorch version):
        ```bash
        conda create -n mcreval python=3.12 pip -y
        ```
2. Download Dataset via https://challenge.isic-archive.com/data/#2019
    - Download `Training Data` Task 1 -> 9,1 GB
    - Download `Training Ground Truth` -> 1 MB
    - Download `Test Data` Task 1 -> 3,6 GB
    - Download `Test Ground Truth` -> 454 KB
    - Unzip everything into one folder
3. Adjust the config file as you need
4. Run it
    1. Activate the env in VS Code (as current used interpreter)
    2. Open any python file + Click on the arrow next to the Run button and choose "Debug using launch.json" and choose mcreval then


> If you want to use the label format helper, [see here](#use-label-format-helper).


<br><br>

---
### Used Data

In this benchmark we use the [WHU Urban 3D Dataset](https://whu3d.com/dataset/docs/#whu-3d/en/dataset/WHU-Urban-3D-dataset) and the [SUD-ROAD Dataset](https://github.com/msqiu/SUD-Road). Both already include segmentation of manholes and we apply a simple pre-processing step, to extract the center points using the geometric mean.

FIXME: make preprocessed data, 3D and 2D? Upload on Kaggle? Speak with your supervisor if that would be ok, for transparency this would be important. -> with licensing of the 2 datasets ok?

> Note that these labels might not be the exact true center of the manholes. It would be very difficult to find the exact center and also human labeling would include a bias to the data.<br>To reduce the bias we decided to do not use the geometric mean as center-finding approach for segmentation models and FIXME (maybe use mean of RANSAC, Mean and Least Square -> still have bias, just more hidden bias).

<bR><br>

The used datasplits are described below:

<br>

WHU Urban 3D Datasplit:
```python
self.train_ids = ['8018', '4938', '0414', '2002', '0444', '1046', '5642', '4333', '4629', '0424', '2421', '0947', '0434', '2022', '2719', '2810', '8048', '2423', '2522', '8008', '0502', '6017', '3918', '2422', '2322', '3405', '2323', '8038']
self.val_ids = ['0404', '6027', '3648']
self.test_ids = ['0940', '2447', '6037', '2321', '8028', '5627', '2521']
```

<br>

SUD-ROAD Datasplit:
```python
self.train_ids = ['8', '10', '1', '11', '3', '4', '6', '5', '7']
self.val_ids = ['0']
self.test_ids = ['2', '9']
```


<br><br>

---
### Label Format

Predictions have to be in following json format:

```python
preds = [
  {
    "dataset": "whu"/"sud",
    "pointcloud_id": "4040",  # id of the point cloud -> not from the patch, from the scene
    "centers": [
        {
            "x": 12.30,
            "y": 5.70,
            "z": 0.10
        },
        {
            "x": 34.22,
            "y": 0.11,
            "z": 1.45
        },
        ...
    ] 
  },

  {...},
  ...
]
```

<!--
```json
[{
  "dataset": "whu",
  "data": [
    {
      "pointcloud-id": "0404",
      "ground_truth": [ 
        {
            "x": 12.30,
            "y": 5.70,
            "z": 0.10
        },
        {
            "x": 34.22,
            "y": 0.11,
            "z": 1.45
        }
      ]
    },
    {
      "pointcloud-id": "0405",
      "ground_truth": [...]
    }
  ]
},
...
]
```
-->

<!--
<br><br>

You can use this repo to convert a list of dicts to this format and directly save it. [See here for it.](#use-label-format-helper)
-->

<!--
<br><br>

---
### Use Label Format Helper

The label format helper can help you to get your predictions in the right json format. Just prepare your predictions in following format:
```python
preds = [
  {
    "dataset": "whu"/"sud",
    "pointcloud_id": "4040",  # id of the point cloud -> not from the patch, from the scene
    "centers": [
        {
            "x": 12.30,
            "y": 5.70,
            "z": 0.10
        },
        {
            "x": 34.22,
            "y": 0.11,
            "z": 1.45
        },
        ...
    ] 
  },

  {...},
  ...
]
```

<br><br>

In order to use the label format helper you need to install this eval lib in your env (we tried to keep dependencies small).

```bash
pip install -e PATH/TO/THIS/LIB/FOLDER
```

And in your code add:
```python
from mcreval.label_creation import save_labels

# ...

save_labels(
        predictions=YOUR_PREDS,                  # list[dict[str, Any]]
        output_dir="./YOUR_RESULT_FILE_NAME",    # str
        exp_name="EXPERIMENT NAME",              # str
        indent=4                                 # int
    )
```
-->



