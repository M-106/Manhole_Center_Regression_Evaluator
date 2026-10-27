# Manhole Center Regression **Evaluator**

Evaluation Repo to eval manhole center predictions.


<br><br>

---
### Installation & Usage

1. Prepare Env
    1. Install Anaconda
    2. Open Anaconda Prompt
    3. Create env (you might to install another pytorch version):
        ```bash
        conda create -n mcreval python=3.12 pip -y
        conda activate mcreval
        ```
    4. Install Libs
        ```bash
        pip install numpy scipy matplotlib open3d h5py scikit-learn scipy
        ```
2. Download Dataset + Preprocessing (FIXME -> cooming soon)
    1. Download Datasets
        1. Visit [WHU Urban 3D Dataset](https://whu3d.com/download.html) and download the MLS dataset
        2. [Download the SUD-ROAD dataset](https://drive.google.com/file/d/17rQKcSWQhFsOPyQK3QrYZ7cSf0pjuTQr/view?usp=sharing)
    2. Preprocess Datasets 
        1. Install MCR-Lab
            - Download the [MCR-Lab](https://github.com/M-106/MCR-Lab)  *(with git clone or download as zip an unzip it)*
            - Download [Anaconda](https://www.anaconda.com/download/success) -> `Anaconda Distribution`, recommendation: one person installation (not all)
            - Install Anaconda via donwloaded installer -> default settings are most likely fine
            - Run Anaconda Prompt
                ```bash
                conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
                conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
                ```
            - Install env
                ```bash
                conda create -n mcrlab python=3.12 pip -y
                conda activate mcrlab
                cd "C:\Users\tippolito\workspace\MCR-Lab" # with your path to the downloaded repo
                # you might want to install a specific PyTroch version before
                # https://pytorch.org/get-started/locally/
                pip install -e .
                ```
          2. Preprocess data and create patches
              - Change these seetings in the **MCR-Lab/configs/config.yaml**
                  ```yaml
                  mode: preprocessing
                  # ...
                  data: 
                    name: "whu"
                    path: "Patch/to/whu3d-dataset-wo-test-label"
                  ```
              - Start env (`conda env list`) + run python -> in anconda prompt
                  ```bash
                  conda activate mcrlab
                  cd "C:\Users\tippolito\workspace\MCR-Lab"  # with your path to the downloaded repo
                  mcrlab --config "./configs/config.yaml"
                  ```
              - Change the settings for SUD and run again if you want to also preprocess SUD dataset
              - In some cases you must change the rights of the new created files/folder which is inside the old dataset path a new folder called `preprocessed` 
          3. Create Center Ground Truths (if wanted) 
              - Change these seetings in the **MCR-Lab/configs/config.yaml**
                  ```yaml
                  mode: gt_extraction
                  
                  gt_extraction:
                    generate_2d_gt_maps: False  # must be False
                    names: ["whu", "sud"]
                    data_paths: ["F:/Fraunhofer_Backup/data/whu3d-dataset-wo-test-label", "F:/Fraunhofer_Backup/data/SUD-ROAD"]
                    save_path: "./output"  # where to save the gt, will crate a new folder inside of here
                    center_algorithm: "squares"    # choose algorithm to extract the centers: "squares", "mean", "ransac", "mesqra"
                    
                  data: 
                    type: "test"
                  ```
              - Start env (`conda env list`) + run python -> in anconda prompt
                  ```bash
                  conda activate mcrlab
                  cd "C:\Users\tippolito\workspace\MCR-Lab"  # with your path to the downloaded repo
                  mcrlab --config "./configs/config.yaml"
                  ```
              - Centers will be at `your/save/path/center_gt` and some debugging plots at: `your/save/path/center_gt/plots`
          4. Create Heatmap Ground Truths (if wanted) 
              - Change these seetings in the **MCR-Lab/configs/config.yaml**
                ```yaml
                mode: gt_extraction
                
                gt_extraction:
                  generate_2d_gt_maps: True  # must be True
                  names: ["whu", "sud"]
                  data_paths: ["F:/Fraunhofer_Backup/data/whu3d-dataset-wo-test-label", "F:/Fraunhofer_Backup/data/SUD-ROAD"]
                  save_path: "./data"  # where to save the gt, will crate a new folder inside of here
                  center_algorithm: "squares"    # choose algorithm to extract the centers: "squares", "mean", "ransac", "mesqra"
                  debugging_path: "./output"  # if None, will not create debugging plots (only for heatmaps)

                data: 
                  type: "all"
                ```
              - Start env (`conda env list`) + run python -> in anconda prompt
                  ```bash
                  conda activate mcrlab
                  cd "C:\Users\tippolito\workspace\MCR-Lab"  # with your path to the downloaded repo
                  mcrlab --config "./configs/config.yaml"
                  ```
              - It is sometimes needed to change the rights of the new created files, depending on your specific system: your heatmaps are created under: 'your/save/path/heatmaps' and your debuggings under 'your/debugging/path/heatmaps_debugging'
3. Adjust the [config file](./config.json) as you need -> set there you target ground truth, prediction and debugging options and data-paths (needed for debugging) and also the name of the eval run.
4. Run it
    1. Open any python file
    2. Activate the env in VS Code (as current used interpreter in the right bottom corner)
    3. Click on the arrow next to the Run button and choose "Debug using launch.json" and choose mcreval then


> If you want to know the needed prediction format [see here](#label-format).


<br><br>

---
### Used Data

In this benchmark we use the [WHU Urban 3D Dataset](https://whu3d.com/dataset/docs/#whu-3d/en/dataset/WHU-Urban-3D-dataset) and the [SUD-ROAD Dataset](https://github.com/msqiu/SUD-Road). Both already include segmentation of manholes and we apply a simple pre-processing step, to extract the center points using the geometric mean.

> **See there licensing, using the links on top, before publishing anything connected with there data.**

<bR><br>

The used datasplits are described below:

<br>

WHU Urban 3D Datasplit: (official split)
```python
self.train_ids = ['8018', '4938', '0414', '2002', '0444', '1046', '5642', '4333', '4629', '0424', '2421', '0947', '0434', '2022', '2719', '2810', '8048', '2423', '2522', '8008', '0502', '6017', '3918', '2422', '2322', '3405', '2323', '8038']
self.val_ids = ['0404', '6027', '3648']
self.test_ids = ['0940', '2447', '6037', '2321', '8028', '5627', '2521']
```

<br>

SUD-ROAD Datasplit: (own splitting: 64 - 16 - 20)
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
            "z": 0.10,
            "time": 0.01  # in seconds
        },
        {
            "x": 34.22,
            "y": 0.11,
            "z": 1.45,
            "time": 0.00023
        },
        ...
      "confidence": 0.5
    ] 
  },

  {...},
  ...
]
```



