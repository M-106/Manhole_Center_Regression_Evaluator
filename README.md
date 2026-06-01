# Manhole_Prediction_Evaluator

Evaluation Repo to eval manhole center prediction.


Predictions have to make in following json format:
```json
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

<br><br>

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

This evaluator uses extracted center points from MCR-Lab repo in following format:
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




