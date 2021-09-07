import codecs
import numpy as np
import re

from sklearn.metrics.pairwise import cosine_similarity


class VectorBuilder:
    def __init__(self):
        self.stopwords = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
             "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
             "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
             "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
             "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
             "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
             "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
             "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
             "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
             "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}

    @staticmethod
    def corp_to_vec(embedding: dict, corpus: list):
        count = 0
        mean_word2vec = None

        for _word in corpus:
            if _word in embedding.keys():
                count += 1
                if mean_word2vec is None:
                    mean_word2vec = embedding[_word]
                else:
                    mean_word2vec += embedding[_word]

        if mean_word2vec is not None:
            mean_word2vec = mean_word2vec / count
            return mean_word2vec
        else:
            return np.zeros(300)

    def _clean(self, raw: str):
        punc_stripped = [re.sub(r'[^\w\s]', '', j.lower().strip("\n").replace("\n", " ")) for j in raw]

        return [word for word in punc_stripped if word not in self.stopwords and word]

    def build_corpus(self, card: dict):
        _name = card['name']

        string = " ".join((card['name'], card['type_line']))
        _corp = self._clean(string.split(" "))
        print(_corp)
        return _name, _corp


class Engine:
    def __init__(self,  embedding: dict, ids, comma_seperated_words):
        self.stopwords = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
                          "yourself",
                          "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
                          "itself",
                          "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
                          "that", "these",
                          "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                          "having", "do",
                          "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
                          "while",
                          "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during",
                          "before",
                          "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
                          "under", "again",
                          "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
                          "both", "each",
                          "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
                          "so", "than",
                          "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}

        self.embedding_file_name_w_o_suffix = ""

        self.ids = ids
        self.csw = comma_seperated_words

        self.embedding = self.load_word_emb_binary()
        vectors = self.build_vectors()
        self.similarity_matrix = cosine_similarity(vectors, vectors)

    def load_word_emb_binary(self):
        print("Loading binary word embedding from {0}.vocab and {0}.npy".format(self.embedding_file_name_w_o_suffix))

        with codecs.open(self.embedding_file_name_w_o_suffix + '.vocab', 'r', 'utf-8') as f_in:
            index2word = [line.strip() for line in f_in]

        wv = np.load(self.embedding_file_name_w_o_suffix + '.npy')
        word_embedding_map = {}
        for i, w in enumerate(index2word):
            word_embedding_map[w] = wv[i]

        return word_embedding_map

    def build_vectors(self):
        vb = VectorBuilder()
        vectors = []
        for id, text in zip(self.ids, self.csw):
            vectors.append(vb.build_corpus(text))
        return np.array(vectors)

    @staticmethod
    def create_similarity_matrix( vectors: np.array):
        return cosine_similarity(vectors, vectors)

    @staticmethod
    def generate_suggestions( ids, similarity_matrix: np.matrix):
        sorted_sim_scores = [b[0] for b in sorted(enumerate(similarity_matrix), reverse=True, key=lambda j: j[1])]
        return [(ids[y], similarity_matrix[y]) for y in sorted_sim_scores]

    def _clean(self, raw):
        return [re.sub(r'[^\w\s]', '', j.lower().strip("\n").replace("\n", " ")) for j in raw if
                j not in self.stopwords]

    def build_corpus(self, card: dict):
        _name = card['name']

        string = " ".join((card['name'], card['type_line']))
        _corp = self._clean(string.split(" "))
        print(_corp)
        return _name, _corp

    def corp_to_vec(self, corpus: list):
        count = 0
        mean_word2vec = None

        for _word in corpus:
            if _word in self.embedding.keys():
                count += 1
                if mean_word2vec is None:
                    mean_word2vec = self.embedding[_word]
                else:
                    mean_word2vec += self.embedding[_word]

        if mean_word2vec is not None:
            mean_word2vec = mean_word2vec / count
            return mean_word2vec
        else:
            return np.zeros(300)

