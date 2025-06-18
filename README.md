# PSD_code
The public code for the paper "A Novel Representation of Periodic Pattern and Its Application to Untrained Anomaly Detection"

## Some Results in Paper
Please note that due to file size limitations, some of the experimental result files have been stored in Google Drive. These files include:
`Grid_Result_final_0729.pkl` : output by `Experiment_Grid.py`, can be read using `read_result.py`
`Grid_Result_LR_AS.pkl` : output by `Experiment_Grid_LR.py`
`Grid_Result_PA.pkl` : output by `Experiment_Grid_PA.py`, can be read using `read_result.py`
You can access these files via the following link:
[Google Drive Link](https://drive.google.com/drive/folders/1r3AexLGMbtgz_17OgnT0eViy8FH-vX1I?usp=sharing)
If you need to download these files, please click on the above link and follow the instructions provided. Thank you for your understanding!

## Experiment of the Grid dataset

Run `Experiment_Grid.py` to apply the anomaly detection method to the Grid dataset from the MVTec AD dataset:

    $ python Experiment_Grid.py --dataset_path  GRID_DATA_SET_PATH

    optional arguments:
      -h, --help            show this help message and exit
          
      --dataset_path DATASET_PATH
                            Path to the Grid dataset
      --result_output_path RESULT_OUTPUT_PATH
                            Path to the save results, .pkl
      --lr_l LR_L           Learning rate for adam optimizer for periodic pattern vectors lambda_1 and lambda_2
      --rou_l1_lambda ROU_L1_LAMBDA
                            Tuning parameter to control the sparsity of pattern vectors lambda_1 and lambda_2
      --rou_l1_A ROU_L1_A   Tuning parameter to control the sparsity of Anomaly components
      --rou_row_sum ROU_ROW_SUM
                            Tuning parameter to control the sum of each row of W matrix
      --max_iteration MAX_ITERATION
                            max iteration time
      --patch_width PATCH_WIDTH
                            patch width
      --kernel_std KERNEL_STD
                            standard deviation for gaussian kernel average
      --device DEVICE       torch device, cuda(default) or cpu

The results are saved to `result_output_path` (default: `Grid_Result.pkl`). 

### Read Experiment Result

Run `read_result.py` to read result file: 

`result_path` default: `Grid_Result.pkl`

    $ python read_results.py

    optional arguments:
      -h, --help            show this help message and exit
      --result_path RESULT_PATH
                            Path to the result file, .pkl

## Test an image

Run `Anomaly_Detection.py` to test an image (image size: 256x256):

`image_path` default: `test_grid_image.png`

    $ python Anomaly_Detection.py
    
    optional arguments:
      -h, --help            show this help message and exit
      --image_path IMAGE_PATH
                            Path to the image
      --lr_l LR_L           Learning rate for adam optimizer for periodic pattern vectors lambda_1 and lambda_2
      --lr_A LR_A           learning rate for adam optimizer for Anomaly components
      --rou_l1_lambda ROU_L1_LAMBDA
                            Tuning parameter to control the sparsity of pattern vectors lambda_1 and lambda_2
      --rou_l1_A ROU_L1_A   Tuning parameter to control the sparsity of Anomaly components
      --rou_row_sum ROU_ROW_SUM
                            Tuning parameter to control the sum of each row of W matrix
      --max_iteration MAX_ITERATION
                            max iteration time
      --patch_width PATCH_WIDTH
                            patch width
      --kernel_std KERNEL_STD
                            standard deviation for gaussian kernel average
      --device DEVICE       torch device, cuda(default) or cpu

## Requirements

    # Please use Python 3.8 or newer.
    numpy
    Pillow
    scipy
    tabulate
    tifffile
    tqdm
    matplotlib
    opencv-python
    scikit-learn
    matplotlib
    hnswlib
    numba
    torch

## Proximal_Adam
    In the code files with the "_PA" suffix, the algorithm in the pattern learning section adopts Proximal_Adam.
    Proximal_Adam: 
    Melchior, P., Joseph, R., & Moolekamp, F. (2019). Proximal Adam: robust adaptive update scheme for constrained optimization. arXiv preprint arXiv:1910.10094.
