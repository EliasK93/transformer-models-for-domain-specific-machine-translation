"""
Largely based on:
https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/parallel-sentence-mining
and:
https://habr.com/en/post/586574/
"""
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from lingtrain_aligner import preprocessor, splitter
from domain_specific_machine_translation.file_utils import read_txt, write_txt


def find_best_knn_pair_indices(x: np.array, y: np.array, k: int = 4, min_score: float = 1.1) -> np.array:
    """
    See https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/parallel-sentence-mining
    """

    # for each point in x, calculate the k nearest neighbours from y and vice versa
    x2y_sim, x2y_ind = k_nearest_neighbours(x, y, k=k)
    y2x_sim, y2x_ind = k_nearest_neighbours(y, x, k=k)

    # for each point in x and in y, calculate the mean of these k neighbours
    x2y_mean = x2y_sim.mean(axis=1)
    y2x_mean = y2x_sim.mean(axis=1)

    # compute normalized dot-product scores based on these neighbors
    x2y_scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean)
    y2x_scores = score_candidates(y, x, y2x_ind, y2x_mean, x2y_mean)

    # find the best translation for each sentence based on the score
    x2y_best = x2y_ind[np.arange(x.shape[0]), x2y_scores.argmax(axis=1)]
    y2x_best = y2x_ind[np.arange(y.shape[0]), y2x_scores.argmax(axis=1)]

    # create pairs of indices from the best x2y and y2x matches
    indices = np.stack([np.concatenate([np.arange(x.shape[0]), y2x_best]),
                        np.concatenate([x2y_best, np.arange(y.shape[0])])], axis=1)

    # get the respective scores from both directions
    scores = np.concatenate([x2y_scores.max(axis=1), y2x_scores.max(axis=1)])

    # keep only those pairs where both indices are the first occurrence of that index in this language
    first_seen_indices = []
    seen_src, seen_trg = set(), set()
    for i in np.argsort(-scores):
        src_ind, trg_ind = indices[i]
        if src_ind not in seen_src and trg_ind not in seen_trg:
            seen_src.add(src_ind)
            seen_trg.add(trg_ind)
            first_seen_indices.append(i)
    indices = indices[first_seen_indices]
    scores = scores[first_seen_indices]

    # filter by min_score
    indices = indices[scores >= min_score]

    print(f"{len(indices)} sentence pairs extracted.")

    return indices


def k_nearest_neighbours(x, y, k):
    """
    See https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/parallel-sentence-mining

    Find the k nearest neighbors using FAISS.
    """
    index = faiss.IndexFlatIP(y.shape[1])
    index.add(y)
    return index.search(x, k)


def score_candidates(x, y, candidates, x2y_mean, y2x_mean):
    """
    See https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/parallel-sentence-mining

    1) Calculate dot product as a proxy for the cosine similarity (proportional for unit vectors)
    2) Select similarities between each value and its k nearest neighbours candidates
    3) Normalize by mean of the k nearest neighbours similarity values
    """
    return (x @ y.T)[np.arange(x.shape[0])[:, None], candidates] / ((x2y_mean[:, None] + y2x_mean[candidates]) / 2)


def clean_pairs(pairs: list[dict[str: str]], min_length: int = 20, max_length: int = 200) -> list[dict[str: str]]:
    """
    Clean unwanted characters and placeholders from the sentence pairs and filter by length.

    :param pairs: list of dictionaries containing the aligned source and target sentences
    :param min_length: minimum length (characters) for inclusion
    :param max_length: maximum length (characters) for inclusion
    """

    # remove newlines, tabs and placeholders
    pairs = [{k: v.replace("\n", " ").replace("\t", " ").replace("%%%%%", "") for k, v in d.items()} for d in pairs]

    # filter by length requirements
    pairs = [{src_lang: v[src_lang], trg_lang: v[trg_lang]} for v in pairs
             if min_length <= len(v[src_lang]) <= max_length and min_length <= len(v[trg_lang]) <= max_length]

    print(f"{len(pairs)} sentence pairs matched length requirements ({min_length} to {max_length} chars).")

    return pairs


if __name__ == '__main__':

    src_lang = "en"
    trg_lang = "de"
    model = SentenceTransformer("LaBSE")

    all_pairs = []
    for i in range(1, 8):
        print(f"\nFile {i}: ")
        # read and sentence split raw files
        sents_src = read_txt(f"../i_raw_text_files/hp_{i}_{src_lang}.txt")
        sents_src = splitter.split_by_sentences_wrapper(preprocessor.mark_paragraphs(sents_src), src_lang)
        sents_trg = read_txt(f"../i_raw_text_files/hp_{i}_{trg_lang}.txt")
        sents_trg = splitter.split_by_sentences_wrapper(preprocessor.mark_paragraphs(sents_trg), trg_lang)
        # encode sentences
        encoded_src = model.encode(sentences=list(sents_src), show_progress_bar=True, normalize_embeddings=True)
        encoded_trg = model.encode(sentences=list(sents_trg), show_progress_bar=True, normalize_embeddings=True)
        # use kNN to find indices of pairs above score threshold
        indices = find_best_knn_pair_indices(x=encoded_src, y=encoded_trg)
        # get the actual sentences
        pairs = [{src_lang: sents_src[src_ind], trg_lang: sents_trg[trg_ind]} for src_ind, trg_ind in indices]
        # clean and filter by length
        pairs = clean_pairs(pairs)
        # add to list of pairs
        all_pairs.extend(pairs)

    # write to file
    print(f"\nWriting {len(all_pairs)} sentence pairs to aligned TXT files.")
    write_txt([d[src_lang] for d in all_pairs], "txt/src.txt")
    write_txt([d[trg_lang] for d in all_pairs], "txt/trg.txt")
