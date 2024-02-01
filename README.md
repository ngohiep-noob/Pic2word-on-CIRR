# Composed Image Retrieval on CIRR Dataset

## Introduction

This repository utilizes the open-source implementation of the zero-shot image retrieval approach [Pic2word](https://github.com/google-research/composed_image_retrieval).

Experiments are conducted on the train, val, and test sets of the [CIRR](https://github.com/Cuberick-Orion/CIRR) dataset.

## Installation

1. Follow the instructions in [Pic2word](https://github.com/google-research/composed_image_retrieval), especially for [Test Data](https://github.com/google-research/composed_image_retrieval#test-data) and [Pre-trained Models](https://github.com/google-research/composed_image_retrieval?tab=readme-ov-file#pre-trained-model).

2. Change the value of `CKPT_PATH` in `app/params.py` to the path of the downloaded checkpoint file.

3. Download the index file from [here](https://drive.google.com/file/d/1-12Tt4e-qnc_TmpQujUO5UgNL9qx-9cG/view?usp=sharing). This file stores feature vectors of images from the training set.

4. Change the value of `INDEX_PATH` in `app/params.py` to the path of the downloaded index file.

5. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   _Some dependencies may not be installed correctly. Please install them manually if needed._

6. Run the Streamlit app:
   ```bash
   streamlit run streamlit.py
   ```

## Experiment Results

| Data Split | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
| :--------: | :------: | :------: | :-------: | :-------: | :-------: |
|   Train    |   0.07   |   0.26   |   0.36    |   0.48    |   0.65    |
|    Val     |   0.15   |   0.45   |   0.58    |    0.7    |   0.82    |
|    Test    |   0.14   |   0.47   |   0.62    |    \_     |   0.86    |

_Results on the test set are evaluated by submitting retrieved answers to the [evaluation server](https://cirr.cecs.anu.edu.au/)._

## Demo

Click [here](https://drive.google.com/file/d/1fi1I035gnc4u0f-gqu2lAC-anLod5Uba/view?usp=drive_link) to watch demo video.
