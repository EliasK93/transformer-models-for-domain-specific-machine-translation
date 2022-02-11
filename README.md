## Transformer Models for Domain-Specific Machine Translation

Examplary application for the task of fine-tuning pretrained machine translation models on highly domain-specific 
translated sentences. 

For this, likely translation pairs are first extracted from the original versions and the German book translations of the 
_Harry Potter_ fantasy novel series using a _Translated Sentence Mining_ approach. The extracted sentence translations are then used to fine-tune two baseline 
machine translation models (pre-trained model **MarianMT** for translation from English to German and Google's **T5**).

Afterwards, some metrics are calculated to evaluate the performance boost from fine-tuning the models. 

<br>

### Overview of the procedure

##### I. Parallel Sentence Extraction (Bitext Mining)

1. Split the unaligned txt files for each book and its translation file into sentences using [Lingtrain Aligners](https://github.com/averkij/lingtrain-aligner) splitter and preprocessor
2. Calculate language-independent sentence level embeddings for the split sentences using GoogleAI's [Language-Agnostic BERT Sentence Embeddings](https://ai.googleblog.com/2020/08/language-agnostic-bert-sentence.html) (LaBSE) in the [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) framework
3. Match the best fitting translation pairs for all sentences using K-Nearest Neighbors search, mostly following Sentence Transformers [example application for Translated Sentence Mining](https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/parallel-sentence-mining)
4. Filter the sentence pairs by a minimum similarity score
5. Remove sentence pairs containing sentences shorter than 20 or longer than 200 characters 
5. Split the resulting corpus of ~54.000 likely parallel sentences randomly into a train, validation and test set (80%, 10%, 10%)

##### II. Machine Translation Engine Training on Domain-Specific Corpus

1. Load the pre-trained models [`Helsinki-NLP/opus-mt-en-de`](https://github.com/Helsinki-NLP/Opus-MT) (MarianMTModel) and [`t5-base`](https://github.com/google-research/text-to-text-transfer-transformer) (T5ForConditionalGeneration) in [huggingface](https://huggingface.co/)
2. Fine-tune the models on the extracted parallel sentences using the train and evaluation set for 10 epochs each (training time: 03h-04m-45s for _MarianMT_ and 09h-20m-10s for _T5_ on NVIDIA GeForce GTX 1660 Ti)

##### III. Machine Translation Quality Evaluation

1. Use the non-fine-tuned _MarianMT_ and _T5_ models to get machine translations for the test set
2. Use the fine-tuned models to get machine translations for the test set
3. Calculate [BLEU](https://github.com/mjpost/sacrebleu), [METEOR](https://github.com/nltk/nltk/blob/develop/nltk/translate/meteor_score.py) and [BertScore](https://github.com/Tiiiger/bert_score) between references and the target language translations for each the non-fine-tuned and the fine-tuned models

<br>

### Results

|               Model               |  BLEU  | METEOR | BertScore<sup>1</sup> |
|:---------------------------------:|:------:|:------:|:---------------------:|
|  MarianMT (baseline)              | 0.256  | 0.433  |  0.597                |
|  MarianMT (fine-tuned)            | 0.388  | 0.552  |  0.717                |
|  T5-base (baseline)               | 0.166  | 0.307  |  0.309                |
|  T5-base (fine-tuned)             | 0.340  | 0.492  |  0.662                |

<sup>1</sup>: setting the parameter `rescale_with_baseline` to `True`

<br>

### Requirements

##### - Python >= 3.8

##### - Conda
  - `pytorch==1.7.1`
  - `cudatoolkit=10.1`
  - `pywin32`

##### - pip
  - `transformers`
  - `sentence_transformers`
  - `faiss-gpu`
  - `sacrebleu`
  - `datasets`
  - `bert-score`
  - `lingtrain-aligner`
  - `razdel`
  - `dateparser`
  - `python-dateutil`
  - `numpy`
  - `openpyxl`

<br>

### Note

All files in this repository which contain text from the books are cut off after the first 50 rows.
