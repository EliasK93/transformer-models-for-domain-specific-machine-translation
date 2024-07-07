import random
from domain_specific_machine_translation import file_utils


def split_dict(dict_: dict, splits: tuple[float, float, float], seed: int = 1):
    """
    Helper method to split dictionary to train, val and test set.

    :param dict_: dictionary to split
    :param splits: tuple for train_share, val_share and test_share where each is between 0 and 1 and the sum is 1
    :param seed: seed for splitting
    :return: tuple (train dict, val dict, test dict)
    """
    if sum(splits) != 1.0:
        raise ValueError
    random.seed(seed)
    val_size = int(len(dict_) * splits[1])
    test_size = int(len(dict_) * splits[2])
    train_keys = list(dict_.keys())
    val_keys = []
    test_keys = []
    random.shuffle(train_keys)
    for i in range(test_size):
        test_keys.append(train_keys.pop(0))
    for i in range(val_size):
        val_keys.append(train_keys.pop(0))
    return {k: v for k, v in dict_.items() if k in train_keys}, \
           {k: v for k, v in dict_.items() if k in val_keys}, \
           {k: v for k, v in dict_.items() if k in test_keys}


def create_pairs_dict_from_txt(src_lang: str, trg_lang: str):
    """
    Reads the aligned source and target txt files and returns them in a dict.

    :param src_lang: source language (ISO-639-1)
    :param trg_lang: target language (ISO-639-1)
    :return: nested dictionary containing index as keys and dicts {src_lang: ..., trg_lang: ...} as values
    """
    src_sentences = file_utils.read_txt('../ii_sentence_mining/txt/src.txt')
    trg_sentences = file_utils.read_txt('../ii_sentence_mining/txt/trg.txt')
    return {i: {src_lang: src, trg_lang: trg} for i, (src, trg) in enumerate(zip(src_sentences, trg_sentences))}


if __name__ == '__main__':

    pairs_dict = create_pairs_dict_from_txt(src_lang="en", trg_lang="de")

    train_dict, val_dict, test_dict = split_dict(pairs_dict, splits=(0.8, 0.1, 0.1))
    print(f"Splitting sentence pairs to {len(train_dict)} train, {len(val_dict)} val and {len(test_dict)} test pairs.")

    print(f"Writing sets to JSONL", end="")
    file_utils.write_jsonl({k: {"translation": v} for k, v in train_dict.items()}, path="training_data/jsonl/train.json")
    file_utils.write_jsonl({k: {"translation": v} for k, v in val_dict.items()}, path="training_data/jsonl/val.json")
    file_utils.write_jsonl({k: {"translation": v} for k, v in test_dict.items()}, path="training_data/jsonl/test.json")
    print(" -- done")

    print(f"Writing sets to TXT", end="")
    file_utils.write_txt([v["en"] for v in train_dict.values()], "training_data/txt/train_src.txt")
    file_utils.write_txt([v["en"] for v in val_dict.values()], "training_data/txt/val_src.txt")
    file_utils.write_txt([v["en"] for v in test_dict.values()], "training_data/txt/test_src.txt")
    file_utils.write_txt([v["de"] for v in train_dict.values()], "training_data/txt/train_trg.txt")
    file_utils.write_txt([v["de"] for v in val_dict.values()], "training_data/txt/val_trg.txt")
    file_utils.write_txt([v["de"] for v in test_dict.values()], "training_data/txt/test_trg.txt")
    print(" -- done")
