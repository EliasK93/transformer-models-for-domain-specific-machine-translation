"""
Largely based on:
https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/parallel-sentence-mining
and:
https://habr.com/en/post/586574/
"""
import faiss
from sentence_transformers import SentenceTransformer
import numpy
from lingtrain_aligner import preprocessor, splitter


class ParallelSentenceMiner:

    def __init__(self, src_lang: str, trg_lang: str, model: str = "LaBSE"):
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.model = SentenceTransformer(model)
        self.pairs = {}

    def mine_and_add_pairs(self, file_src: str, file_trg: str):
        """
        Extracts parallel sentence pairs from two txt documents where one is a translation of
        the other document, then adds the pairs to the self.pairs dict.

        :param file_src: path to source corpus txt file
        :param file_trg: path to source corpus txt file
        """
        with open(file_src, encoding="utf-8") as f:
            sents_de = self.split_by_sentences_(f.readlines(), self.src_lang)
        with open(file_trg, encoding="utf-8") as f:
            sents_en = self.split_by_sentences_(f.readlines(), self.trg_lang)
        self.add_pairs_knn(sents_src=sents_de, sents_trg=sents_en)

    def add_pairs_knn(self, sents_src: list[str], sents_trg: list[str]):
        """
        Finds bilingual sentence pairs which maximize their similarity by first encoding them,
        using k-nearest neighbors to match and score them and then filtering pairs by a minimum score.

        :param sents_src: source corpus split by sentences
        :param sents_trg: target corpus split by sentences
        """
        encoded_de = self.model.encode(sentences=list(sents_src), show_progress_bar=True,
                                       convert_to_numpy=True, normalize_embeddings=True)
        encoded_en = self.model.encode(sentences=list(sents_trg), show_progress_bar=True,
                                       convert_to_numpy=True, normalize_embeddings=True)
        indices, scores = self.get_knn_indices_and_scores(x=encoded_de, y=encoded_en)
        self.add_to_pairs_dict(indices, scores, sents_src, sents_trg)

    def add_to_pairs_dict(self, indices: numpy.ndarray, scores: numpy.ndarray,
                          sents_src: list[str], sents_trg: list[str],
                          min_score: float = 1.1, min_length: int = 20, max_length: int = 200):
        """
        Uses the indices and scores arrays to build a dictionary of the sentence pairs which have a score of
        at least min_score.

        :param indices: numpy.ndarray of shape [2 x num of pairs] containing the indices of best matches
        :param scores: numpy.ndarray of shape [1 x num of pairs] containing the scores of best matches
        :param sents_src: source corpus split by sentences
        :param sents_trg: target corpus split by sentences
        :param min_score: minimum score for inclusion of sentence pairs
        :param min_length: minimum length (characters) for inclusion
        :param max_length: maximum length (characters) for inclusion
        """
        seen_src, seen_trg = set(), set()
        extracted_parallel_sents = {}
        for i in numpy.argsort(-scores):
            src_ind, trg_ind = indices[i]
            src_ind = int(src_ind)
            trg_ind = int(trg_ind)
            if src_ind not in seen_src and trg_ind not in seen_trg:
                seen_src.add(src_ind)
                seen_trg.add(trg_ind)
                extracted_parallel_sents[len(extracted_parallel_sents) + 1] = {
                    "score": scores[i],
                    self.src_lang: sents_src[src_ind].replace("\t", " ").replace("%%%%%", ""),
                    self.trg_lang: sents_trg[trg_ind].replace("\t", " ").replace("%%%%%", "")
                }
        print(f"{len(extracted_parallel_sents)} sentences extracted.")
        extracted_parallel_sents = [{self.src_lang: v[self.src_lang], self.trg_lang: v[self.trg_lang]}
                                    for k, v in extracted_parallel_sents.items() if v["score"] >= min_score]
        print(f"{len(extracted_parallel_sents)} sentences matched min score ({min_score}).")
        extracted_parallel_sents = [{self.src_lang: v[self.src_lang], self.trg_lang: v[self.trg_lang]}
                                    for v in extracted_parallel_sents
                                    if min_length <= len(v[self.src_lang]) <= max_length
                                    and min_length <= len(v[self.trg_lang]) <= max_length]
        print(f"{len(extracted_parallel_sents)} sentences matched length requirements "
              f"({min_length} to {max_length} chars).")

        for pair in extracted_parallel_sents:
            self.pairs[len(self.pairs) + 1] = pair

    def write_output_txt_files(self):
        """
        Writes all sentences from self.pairs to source txt file and target txt file, aligned by sentences.
        """
        with open("txt/trg.txt", "w", encoding="utf-8") as f:
            for sent in [d[self.src_lang].replace("\n", " ") for d in self.pairs.values()]:
                f.write(sent+"\n")

        with open("txt/src.txt", "w", encoding="utf-8") as f:
            for sent in [d[self.trg_lang].replace("\n", " ") for d in self.pairs.values()]:
                f.write(sent+"\n")

        print(f"\nWrote {len(self.pairs)} sentence pairs to aligned TXT files.")

    @staticmethod
    def split_by_sentences_(text, lang):
        """
        See https://habr.com/en/post/586574/
        """
        text_prepared = preprocessor.mark_paragraphs(text)
        return splitter.split_by_sentences_wrapper(text_prepared, lang)

    def encode(self, sents):
        """
        See https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/parallel-sentence-mining
        """
        return self.model.encode(sents, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    @staticmethod
    def get_knn_indices_and_scores(x, y):
        """
        See https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/parallel-sentence-mining
        """
        # Perform kNN in both directions
        x2y_sim, x2y_ind = ParallelSentenceMiner.kNN(x, y, k=4)
        x2y_mean = x2y_sim.mean(axis=1)
        y2x_sim, y2x_ind = ParallelSentenceMiner.kNN(y, x, k=4)
        y2x_mean = y2x_sim.mean(axis=1)

        # Compute forward and backward scores
        fwd_scores = ParallelSentenceMiner.score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, lambda a, b: a / b)
        bwd_scores = ParallelSentenceMiner.score_candidates(y, x, y2x_ind, y2x_mean, x2y_mean, lambda a, b: a / b)
        fwd_best = x2y_ind[numpy.arange(x.shape[0]), fwd_scores.argmax(axis=1)]
        bwd_best = y2x_ind[numpy.arange(y.shape[0]), bwd_scores.argmax(axis=1)]

        indices = numpy.stack([numpy.concatenate([numpy.arange(x.shape[0]), bwd_best]),
                               numpy.concatenate([fwd_best, numpy.arange(y.shape[0])])], axis=1)

        scores = numpy.concatenate([fwd_scores.max(axis=1), bwd_scores.max(axis=1)])
        return indices, scores

    @staticmethod
    def score(x, y, fwd_mean, bwd_mean, margin):
        """
        See https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/parallel-sentence-mining
        """
        return margin(x.dot(y), (fwd_mean + bwd_mean) / 2)

    @staticmethod
    def score_candidates(x, y, candidate_inds, fwd_mean, bwd_mean, margin):
        """
        See https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/parallel-sentence-mining
        """
        scores = numpy.zeros(candidate_inds.shape)
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                k = candidate_inds[i, j]
                scores[i, j] = ParallelSentenceMiner.score(x[i], y[k], fwd_mean[i], bwd_mean[k], margin)
        return scores

    @staticmethod
    def kNN(x, y, k):
        """
        See https://github.com/UKPLab/sentence-transformers/tree/master/examples/applications/parallel-sentence-mining
        """
        idx = faiss.IndexFlatIP(y.shape[1])
        idx.add(y)
        sim, ind = idx.search(x, k)
        return sim, ind
