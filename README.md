# Composed image retrieval on CIRR dataset

## Introduction

This repository utilizes the open-source implementation of zero-shot image retrieval approach [Pic2word](https://github.com/google-research/composed_image_retrieval)

Experiments are conducted on train, val and test set of the [CIRR](https://github.com/Cuberick-Orion/CIRR) dataset.

## Installation

1. Follow the instructions in [Pic2word](https://github.com/google-research/composed_image_retrieval), especially [Test Data](https://github.com/google-research/composed_image_retrieval#test-data) and [Pre-trained Models](https://github.com/google-research/composed_image_retrieval?tab=readme-ov-file#pre-trained-model).

2. Download index file from [here](https://drive.google.com/file/d/1-12Tt4e-qnc_TmpQujUO5UgNL9qx-9cG/view?usp=sharing), this file stored feature vectors of images of training set.

3. Change value of `INDEX_PATH` in `app/params.py` to the path of downloaded index file.

4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run streamlit app:

   ```bash
   streamlit run streamlit.py
   ```

## Experiment result

| Data split | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@50 |
| :----: | :-: | :-: | :--: | :--: | :--: 
|  Train   |  0.07  |  0.26  |  0.36   |  0.48   |  0.65   
|  Val   |  0.15  |  0.45  |  0.58   |  0.7   |  0.82   
|  Test   |  0.14  |  0.47  |  0.62   |  _   | 0.86    

*Results on test set are evaluated by submitting retrieved answers to [evaluation server](https://cirr.cecs.anu.edu.au/)*