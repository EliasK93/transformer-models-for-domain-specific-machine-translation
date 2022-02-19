import os
import re
from typing import List, Tuple
from transformers import MarianTokenizer, T5Tokenizer, MarianMTModel, T5ForConditionalGeneration, \
    PreTrainedTokenizer, PreTrainedModel


model_dirs = {
    model_type: "../iii_model_training/" + re.sub("[-/]", "_", model_type)
    for model_type in ["Helsinki-NLP/opus-mt-en-de", "t5-base"]
}

prefixes = {
    "Helsinki-NLP/opus-mt-en-de": "",
    "t5-base": "translate English to German: "
}

batch_sizes = {
    "Helsinki-NLP/opus-mt-en-de": 16,
    "t5-base": 4
}


def load_model(model_type: str, local: bool) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Loads tokenizer and model either by downloading a pretrained model or from a local directory.

    :param model_type: either the a pretrained model (model_id in huggingface) or path to a local directory
    :param local: True if model should be loaded from local path, False otherwise
    :return: tuple containing tokenizer and model
    """
    if local:
        model_path = model_dirs[model_type]
        model_path += f"\\{find_max_checkpoint(path=model_path)}"
    else:
        model_path = model_type
    print("Loading model " + model_path, end="")

    if model_type == "Helsinki-NLP/opus-mt-en-de":
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        model = MarianMTModel.from_pretrained(model_path)
    elif model_type == "t5-base":
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    else:
        raise ValueError
    print(" -- done")
    return tokenizer, model


def find_max_checkpoint(path: str) -> str:
    """
    Helper method to find highest checkpoint model folder name in the given directory.

    :param path: model directory containing the models checkpoints
    :return: name of the highest checkpoint folder
    """
    all_checkpoints = [c for c in os.listdir(path) if c.startswith("checkpoint")]
    return sorted(all_checkpoints)[-1]


def translate(text_list: List[str], model_type: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> List[str]:
    """
    Creates translations for a list of strings using the provided model and tokenizer.

    :param text_list: list of source texts
    :param model: loaded model object
    :param tokenizer: loaded tokenizer object
    :param model_type: name of the (baseline) model (model_id in huggingface)
    :return: list of strings containing the translations
    """
    tokenized = tokenizer([t + prefixes[model_type] for t in text_list],
                          return_tensors="pt", padding="max_length", max_length=512)
    translated = model.generate(tokenized.input_ids, max_length=512)
    return list([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
