# Why My Code Summarization Model Does Not Work: Code Comment Improvement with Category Prediction. 
## Published on TOSEM (CCF Rank A) 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4133474.svg)](https://doi.org/10.5281/zenodo.4133474)

This repository is the accompanying repository for our paper "Why My Code Summarization Model Does Not Work: Code Comment Improvement with Category Prediction"

https://dl.acm.org/doi/abs/10.1145/3434280

The paper is accessed in [pdf](http://chenqiuyuan.com/TOSEM_Qiuyuan_Chen_2021_Why%20My%20Code%20Summarization%20Model%20Does%20Not%20Work.pdf)

## Requirements
```
* Python3.7+
* Java 8+
```

* For Python requirements, we list Python package dependency in `requirements.txt`. Note that the environment is based on [Anaconda](https://www.anaconda.com/).


* For Java requirements, we adopt Maven to build the projects. The core component is [JavaParser](https://github.com/javaparser/javaparser). Just add the following to your maven configuration.

```
<dependency>
    <groupId>com.github.javaparser</groupId>
    <artifactId>javaparser-symbol-solver-core</artifactId>
    <version>3.16.1</version>
</dependency>
```

## Dataset

* We use the dataset provided by [DeepCom](https://github.com/tech-srl/code2seq) for constructing code summarization models.

* We provide the annotated dataset that our work produced in `dataset` folder.

## Reproducing

### Train the code summarization models
When training models, we use the code provided by [CodeNN](https://github.com/sriniiyer/codenn), [Code2Seq](https://github.com/tech-srl/code2seq), [DeepCom](https://github.com/xing-hu/EMSE-DeepCom), and [NNGenn](https://github.com/Tbabm/nngen).

These projects are all open-sourced.

Please note that we re-design the pre-processing phase of Code2Seq to perform code summarization experiments. We provide the code in the same GoogleDrive url with the dataset. As training model is not feasible for every machine, we also provide our trained model.


### Comment category prediction
We provide source code of comment category prediction. Please locate `ccpm.py` and run as instructions.
1. Leverage the trained model to generate comments.
2. Set the path of the generated comments of each model.
3. Run the commands.
```
python ccpm.py classify --input_file --cross_validation
python ccpm.py evaluate --input_file --output_file
```

## Evaluation
We evaluate the BLEU and ROUGE using the package [nmt-eval](https://github.com/Maluuba/nlg-eval).

Running
```
nlg-eval --hypothesis=examples/generated_comments.txt --references=examples/reference_comments.txt
```


## Citation
If you use our dataset or you are inspired by our work, please consider citing our paper:

```bib
@article{10.1145/3434280,
author = {Chen, Qiuyuan and Xia, Xin and Hu, Han and Lo, David and Li, Shanping},
title = {Why My Code Summarization Model Does Not Work: Code Comment Improvement with Category Prediction},
year = {2021},
issue_date = {February 2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {30},
number = {2},
issn = {1049-331X},
url = {https://doi.org/10.1145/3434280},
doi = {10.1145/3434280},
month = feb,
articleno = {25},
numpages = {29},
keywords = {comment classification, code comment, Code summarization}
}
```

Thanks!

## Reference

[1] Qiuyuan Chen, Xin Xia, Han Hu, David Lo, and Shanping Li, “Why My Code Summarization Model Does Not Work: Code Comment Improvement with Category Prediction,” ACM Trans. Softw. Eng. Methodol. (TOSEM), vol. 30, no. 2, pp. 1–29, 2021, DOI: https://doi.org/10.1145/3434280.

