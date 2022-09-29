# EnglishCheckBert
A fine tuned Bert model API for answering this question "Is the sentence grammatically correct?"

## Installation

1. ``` docker build -t <container-name> .```
2. ``` docker run -p 80:80 <container-name> ```

## Usage gateways

1. ``` GET: /  -> { "health_check": "OK", "model_version": model_version }```
2. ``` POST: /predict/"text"  -> {"veredict": ("acceptable" or "unacceptable")}```

## ML model creation process
### You can find the Notebook [here](https://colab.research.google.com/drive/1qFnrOaCxfip0GiDEv_zct1x8UnW93vFQ?usp=sharing)