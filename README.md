# FocalNet-AE-DualPipeline-LUS

## Download Processed Datasets
1. Download from https://drive.google.com/file/d/1gBcONDFEbUcpqdNAOZq0OCtOHrgWhpOh/view
2. Unzip and place all files in the `data/` directory.

## Install Dependencies
1. All required dependencies are listed in the `environment.yml` file.
2. You can use `conda env create --file environment.yml` to create the environment.

## FocalNet-AE Finetuning
1. Execute step1.sh to start the finetuning process.
   ```bash
    bash step1.sh
    ```
2. The finetuned model will be saved in the `models/` directory.

## Dual-Pipeline Execution
1. You have to finetune the FocalNet-AE model before running this step. (We already provided a finetuned model `models/focalnetae` for convenience.)
2. Execute step2.sh to start the dual-pipeline process.
   ```bash
    bash step2.sh
    ```
3. This step will run unsupervised task, unsupervised independent task, supervised task, and supervised independent task sequentially.

## Calculate Metrics
1. The result files will be saved in the `output/` directory.
2. To get final evaluation metrics, run the following command:
   ```bash
    python postp.py RESULT_PATH
    ```

## Note
1. Do not change the directory structure.
2. Please run the scripts in root directory of repository.