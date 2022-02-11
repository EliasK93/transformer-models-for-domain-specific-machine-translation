import random
from typing import List, Tuple
import model_utils


def sample_sentence_pairs(sample_size: int, sample_seed: int) -> Tuple[List[str], List[str]]:
    """
    Samples sentence pairs (source sentences, reference translations).

    :param sample_size: number of sentence pairs to sample
    :param sample_seed: random seed used for sampling
    :return: tuple (list of source sentences, list of reference translations)
    """
    random.seed(sample_seed)
    with open("../iii_model_training/training_data/txt/test_src.txt", encoding="utf-8") as f_:
        src_text_list_ = [line.rstrip() for line in f_.readlines()]
    with open("../iii_model_training/training_data/txt/test_trg.txt", encoding="utf-8") as f_:
        trg_text_list_ = [line.rstrip() for line in f_.readlines()]
    random_indices = random.sample(range(len(src_text_list_)), sample_size)
    return [src for index, src in enumerate(src_text_list_) if index in random_indices], \
           [trg for index, trg in enumerate(trg_text_list_) if index in random_indices]


if __name__ == '__main__':

    # sample sentence pairs (source sentences, reference translations)
    # optimally this would sample the entire test set (~5.400 sentence pairs),
    # limiting to 500 here since model inference is quite slow
    src_text_list, trg_text_list = sample_sentence_pairs(sample_size=500, sample_seed=1)

    # write source sentences to file
    with open("sources.txt", "w", encoding="utf-8") as f:
        for r in src_text_list:
            f.write(r + "\n")

    # write reference translations to file
    with open("targets_reference.txt", "w", encoding="utf-8") as f:
        for r in trg_text_list:
            f.write(r + "\n")

    # load pretrained (non-finetuned) models, generate translations and write to file
    for model_type in ["Helsinki-NLP/opus-mt-en-de", "t5-base"]:
        tokenizer, model = model_utils.load_model(model_type, local=False)
        results = model_utils.translate(src_text_list, model_type, model, tokenizer)
        with open("targets_" + model_type.split("/")[-1] + ".txt", "w", encoding="utf-8") as f:
            for r in results:
                f.write(r + "\n")

    # load finetuned models, generate translations and write to file
    for model_type in ["Helsinki-NLP/opus-mt-en-de", "t5-base"]:
        tokenizer, model = model_utils.load_model(model_type, local=True)
        results = model_utils.translate(src_text_list, model_type, model, tokenizer)
        with open("targets_" + model_type.split("/")[-1] + "_finetuned.txt", "w", encoding="utf-8") as f:
            for r in results:
                f.write(r + "\n")
