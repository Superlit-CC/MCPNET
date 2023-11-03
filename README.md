# MCPNET

## Overview

PyTorch implementation of MCPNET.

<img src=".\doc\figs\Abstract.jpg" alt="Abstract" style="zoom:45%;" />

## Installation

Make sure PyTorch 1.11.0 and RDKIT are installed,  should resolve all dependencies. The program is run and tested using python3.

## How to Use

A complete example of running the program is provided in the `attention_pred.py`, including molecular point cloud feature generation, model training and evaluation. You can put your own data in the /data to train model. You should modify the parameters in the following command line to your data path.

```shell
python3 attention_pred.py --source_path [your data path]
```

For visualizing MCPNET, you should provide your trained model, data, and scaler of data. Then call the `get_important_points` method in `model_utils.py`. The returned data can be rendered by calling the `plot_points` method in `plot_utils.py`. The following figure shows the results:

<img src=".\doc\figs\example.jpg" alt="example" width="300" height="300" align=left />

