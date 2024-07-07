import random
from domain_specific_machine_translation import model_utils


def load_test_set(sample_size: int | None = None, sample_seed: int = 1) -> tuple[list[str], list[str]]:
    """
    Samples sentence pairs (source sentences, reference translations).

    :param sample_size: number of sentence pairs to sample, or None to include all
    :param sample_seed: random seed used for sampling
    :return: tuple (list of source sentences, list of reference translations)
    """
    with open("../iii_model_training/training_data/txt/test_src.txt", encoding="utf-8") as f_:
        src_text_list_ = [line.rstrip() for line in f_.readlines()]
    with open("../iii_model_training/training_data/txt/test_trg.txt", encoding="utf-8") as f_:
        trg_text_list_ = [line.rstrip() for line in f_.readlines()]
    if sample_size is None:
        return src_text_list_, trg_text_list_
    random.seed(sample_seed)
    random_indices = random.sample(range(len(src_text_list_)), sample_size)
    return [src for index, src in enumerate(src_text_list_) if index in random_indices], \
           [trg for index, trg in enumerate(trg_text_list_) if index in random_indices]


if __name__ == '__main__':

    # load sentence pairs from test set (source sentences, reference translations)
    src_text_list, trg_text_list = load_test_set()

    # write source sentences to file
    with open("sources.txt", "w", encoding="utf-8") as f:
        for line in src_text_list:
            f.write(line + "\n")

    # write reference translations to file
    with open("targets_reference.txt", "w", encoding="utf-8") as f:
        for line in trg_text_list:
            f.write(line + "\n")

    # load pretrained (non-finetuned) models, generate translations and write to file
    for model_type in ["Helsinki-NLP/opus-mt-en-de", "t5-large", "facebook/nllb-200-distilled-600M"]:
        for fine_tuned in [False, True]:
            tokenizer, model = model_utils.load_model(model_type, local=fine_tuned)
            results = model_utils.translate(src_text_list, model_type, model, tokenizer)
            out_file_name = "targets_" + model_type.split("/")[-1] + ("_finetuned" if fine_tuned else "") + ".txt"
            with open(out_file_name, "w", encoding="utf-8") as f:
                for line in results:
                    f.write(line + "\n")
