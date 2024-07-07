import os
import re
from tqdm import tqdm
from transformers import MarianTokenizer, T5Tokenizer, MarianMTModel, T5ForConditionalGeneration, \
    PreTrainedTokenizer, PreTrainedModel, AutoModelForSeq2SeqLM, NllbTokenizer


model_dirs = {
    model_type: "../iii_model_training/" + re.sub("[-/]", "_", model_type)
    for model_type in ["Helsinki-NLP/opus-mt-en-de", "t5-large", "facebook/nllb-200-distilled-600M"]
}

prefixes = {
    "Helsinki-NLP/opus-mt-en-de": "",
    "t5-large": "translate English to German: ",
    "facebook/nllb-200-distilled-600M": ""
}

batch_sizes = {
    "Helsinki-NLP/opus-mt-en-de": 32,
    "t5-large": 8,
    "facebook/nllb-200-distilled-600M": 16
}


def load_model(model_type: str, local: bool) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
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
        model = MarianMTModel.from_pretrained(model_path).to("cuda:0")
    elif model_type == "t5-large":
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path).to("cuda:0")
    elif model_type == "facebook/nllb-200-distilled-600M":
        tokenizer = NllbTokenizer.from_pretrained(model_path, src_lang="eng_Latn", tgt_lang="deu_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda:0")
    else:
        raise ValueError(model_type)
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


def translate(text_list: list[str], model_type: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, chunk_size: int = 64) -> list[str]:
    """
    Creates translations for a list of strings using the provided model and tokenizer.

    :param text_list: list of source texts
    :param model_type: name of the (baseline) model (model_id in huggingface)
    :param model: loaded model object
    :param tokenizer: loaded tokenizer object
    :param chunk_size: number of sentences to put through tokenization, generation and decoding at once
    :return: list of strings containing the translations
    """
    translated_texts = []
    text_list_chunks = [text_list[i:i + chunk_size] for i in range(0, len(text_list), chunk_size)]
    for text_list_chunk in tqdm(text_list_chunks, desc=f"{model_type}: generating translations"):
        tokenized_chunk = tokenizer([t + prefixes[model_type] for t in text_list_chunk],
                                    return_tensors="pt", padding="max_length", max_length=512).to("cuda:0")
        if model_type == "facebook/nllb-200-distilled-600M":
            translated_chunk = model.generate(tokenized_chunk.input_ids, forced_bos_token_id=tokenizer.encode("deu_Latn")[1], max_length=512)
        else:
            translated_chunk = model.generate(tokenized_chunk.input_ids, max_length=512)
        decoded_chunk = tokenizer.batch_decode(translated_chunk, skip_special_tokens=True)
        translated_texts.extend(decoded_chunk)
    return translated_texts
