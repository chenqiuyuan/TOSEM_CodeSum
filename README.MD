# TOSEM_CodeSum
This repository is the accompanying repository for our paper "Why My Code Summarization Approach Does Not Work."

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
