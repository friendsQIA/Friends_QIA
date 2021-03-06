# Friends_QIA

This repository was created for a study on understanding indirect answers in NLP. We release a new dataset, `Friends-QIA` to enable progress on this task.

The repository contains the code and datasets used in the study focused on the task of classification of indirect answers to polar questions.

The two datasets (Circa and Friends) used for the study can be found in the `Data` folder: `Circa_data` and `Friends_data`, respectively. `Friends_data` is further divided into `Original_QA_category_files` with the original files containing the data in the original format divided by category (used for Exploratory Data Analysis) and `Final_QA_datasets`, with the dataset split into train, tune, dev and test sets, used for the models. The Circa dataset was originally created by researchers at [Google](https://github.com/google-research-datasets/circa), the version presented here is split into train, tune, dev and test sets and is structured differently.

The `data_preparation` folder contains Python files with functions used for creating the datasets from the category files as well as the preprocessing functions used for preparing the input for the models.

Files: `model_MISC.py` and `model_SISC.py` contain the two CNN models used in the study: multi input single channel and single input single channel, respectively.

`Predictions` folder contains predictions generated by three runs of `CNN with BERT` and `CNN with BERT + CrowdLayer` as well as the annotator-specific weights estimated by one of the models with CrowdLayer. All of them are used in `performance_bert.ipynb` and `performance_bert_crowds.ipynb` to obtain necessary performance scores.

All scripts were created with Python3 and the following packages:
- jupyter-core==4.7.1
- Keras==2.4.3
- Keras-Preprocessing==1.1.2
- matplotlib==3.4.2
- notebook==6.3.0
- numpy==1.19.5
- pandas==1.1.1
- scikit-learn==0.24.2
- seaborn==0.11.1
- tensorflow==2.4.1
- transformers==4.6.1
- npy-append-array==0.9.6

The results for the base CNN models without using BERT embeddings were obtained with the use of GloVe embeddings. They are available [here](https://nlp.stanford.edu/projects/glove/).

The crowds study was inspired by the paper by [Filipe Rodrigues and Francisco C. Pereira](https://arxiv.org/pdf/1709.01779.pdf) and the code is strongly based on their implementation which can be found in this [GitHub repository](https://github.com/fmpr/CrowdLayer).

The generated BERT embeddings are not included in the repository. They were created by following this [article](https://huggingface.co/bert-base-cased) and used as input to the CNN models. Due to preprocessing issues, the instance number `4678` was removed from the dataset (train+tune set), before creating the embeddings.
The code used for generating the embeddings for this study can be found in `generate_bert_embeddings.py`.


## References

If you use Friends-QIA please cite:
```
@inproceedings{friendsqia,
    title = "{``}{I}{'}ll be there for you{''}: The One with Understanding Indirect Answers",
    author = "Damgaard, Cathrine  and
      Toborek, Paulina  and
      Eriksen, Trine  and
      Plank, Barbara",
    booktitle = "Proceedings of the 2nd Workshop on Computational Approaches to Discourse (CODI) at EMNLP",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
```


For Circa please cite:
```
@inproceedings{louis-etal-2020-id,
    title = "{``}{I}{'}d rather just go to bed{''}: Understanding Indirect Answers",
    author = "Louis, Annie  and
      Roth, Dan  and
      Radlinski, Filip",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.601",
    doi = "10.18653/v1/2020.emnlp-main.601",
    pages = "7411--7425"
}
```

