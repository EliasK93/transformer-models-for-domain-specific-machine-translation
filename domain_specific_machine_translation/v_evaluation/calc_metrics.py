from statistics import mean
import bert_score
import pandas
import nltk
import datasets
from domain_specific_machine_translation import file_utils


def calc_meteor_avg(predictions_: list[str], references_: list[list[str]]) -> float:
    """
    Calculate the average METEOR score between model translations and reference translations.

    :param predictions_: list of model translations
    :param references_: list of reference translations, each sentence additionally wrapped as a list
    :return: METEOR score average (between 0 and 1)
    """
    scores = [nltk.translate.meteor_score.single_meteor_score(reference=ref[0].split(), hypothesis=hyp.split())
              for hyp, ref in zip(predictions_, references_)]
    return mean(scores)


def calc_bleu(predictions_: list[str], references_: list[list[str]]) -> float:
    """
    Calculate the corpus level BLEU score between model translations and reference translations.

    :param predictions_: list of model translations
    :param references_: list of reference translations, each sentence additionally wrapped as a list
    :return: corpus level BLEU score (between 0 and 1)
    """
    return metric.compute(predictions=predictions_, references=references_)["score"] / 100


def calc_bert_score(predictions_: list[str], references_: list[list[str]], lang: str) -> float:
    """
    Calculate the average BertScore between model translations and reference translations.

    :param predictions_: list of model translations
    :param references_: list of reference translations, each sentence additionally wrapped as a list
    :param lang: language code (ISO-639-1)
    :return: corpus level BLEU score (between 0 and 1)
    """
    return float(bert_score.score(predictions_, references_, lang=lang, rescale_with_baseline=True)[2].mean())


if __name__ == '__main__':

    # load BLEU metric
    metric = datasets.load_metric("sacrebleu")

    # ensure wordnet is downloaded for METEOR metric
    nltk.download('wordnet')

    # create dictionaries to put scores in
    scores_dict = {}

    # load reference translations, finetuned model translations and untrained model translations
    references = [[r] for r in file_utils.read_txt("../iv_prediction/txt/targets_reference.txt")]
    sources = [s for s in file_utils.read_txt("../iv_prediction/txt/sources.txt")]

    # calculate scores and put result in nested dict
    for model in ["opus-mt-en-de", "opus-mt-en-de_finetuned", "t5-large", "t5-large_finetuned",
                  "facebook/nllb-200-distilled-600M", "facebook/nllb-200-distilled-600M_finetuned"]:
        preds = file_utils.read_txt(f"../iv_prediction/txt/targets_" + model.split("/")[-1] + ".txt")
        scores_dict[model] = {}
        scores_dict[model]["sacrebleu"] = calc_bleu(preds, references)
        scores_dict[model]["meteor"] = calc_meteor_avg(preds, references)
        scores_dict[model]["bert_score"] = calc_bert_score(preds, references, lang='de')

    # create DataFrame from nested dict using pandas, print and save to Excel
    scores_df = pandas.DataFrame.from_dict(scores_dict).transpose().round(3)
    print(scores_df.to_string())
    scores_df.to_excel("results.xlsx")
