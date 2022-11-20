# Triplet Contrastive Learning For Aspect Level Sentiment Classification

code and datasets of "Triplet Contrastive Learning For Aspect Level Sentiment Classification"

## Requirements

- torch==1.7.0
- scikit-learn==1.0.2
- transformers==3.5.0



To install requirements, run `pip install -r requirements.txt`.

## Preparation

1. Download and unzip GloVe vectors(`glove.840B.300d.zip`) from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) and put it into  `./glove` directory.

2. Prepare vocabulary with:

   `sh ./build_vocab.sh`

3. Training:

​		 `sh /run.sh`
