"""
File for reading a scryfall json file.
"""
import codecs
import json
import re
import time

import gensim.models.keyedvectors
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib import gridspec

from collections import OrderedDict
import operator
from flask import Flask, render_template

# def cosine_similarity(v0, v1):
#     return np.inner(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))

# https://www.kdnuggets.com/2020/08/content-based-recommendation-system-word-embeddings.html

input_json = json.load(open("data/oracle-cards-20210429090441.json", encoding='utf-8'))
print(input_json[0].keys())
print(input_json[0]['image_uris']['normal'])

# exit()
print(len(input_json))

stopwords = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
             "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
             "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
             "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
             "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
             "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
             "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
             "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
             "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
             "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}


stripped_stopwords = {x.rstrip("s") for x in stopwords}
colors = {"R": "red",
          "U": "blue",
          "G": "green",
          "B": "black",
          "W": "white"}


def load_word_emb_binary(embedding_file_name_w_o_suffix):
    print("Loading binary word embedding from {0}.vocab and {0}.npy".format(embedding_file_name_w_o_suffix))

    with codecs.open(embedding_file_name_w_o_suffix + '.vocab', 'r', 'utf-8') as f_in:
        index2word = [line.strip() for line in f_in]

    wv = np.load(embedding_file_name_w_o_suffix + '.npy')
    word_embedding_map = {}
    for i, w in enumerate(index2word):
        word_embedding_map[w] = wv[i]

    return word_embedding_map


# embedding_map = load_word_emb_binary('data/glove.6B.300d')

# wv = gensim.models.KeyedVectors.load('./data/epochs_1000_window_4_vectors.kv')
# DIMENSION = 100
# embedding_map = {w: wv[w] for w in wv.index_to_key}
# #
#
# wv = gensim.models.KeyedVectors.load('./data/epochs_1500_window_5_vectors.kv')
# DIMENSION = 100
# embedding_map = {w: wv[w] for w in wv.index_to_key}
#
#
# wv = gensim.models.KeyedVectors.load('./data/epochs_300_window_4_vectors.kv')
# DIMENSION = 100
# embedding_map = {w: wv[w] for w in wv.index_to_key}
# wv = gensim.models.KeyedVectors.load('./data/epochs_2000_window_3_vectors.kv')
# DIMENSION = 100
#
# embedding_map = {w: wv[w] for w in wv.index_to_key if w != 'file'}
# wv = gensim.models.KeyedVectors.load('./data/epochs_2000_window_3_vectors.kv')
# DIMENSION = 50
#
# embedding_map = {w: wv[w] for w in wv.index_to_key if w != 'file'}
wv = gensim.models.KeyedVectors.load('./data/epochs_6000_window_3_32_vectors.kv')
DIMENSION = 32

embedding_map = {w: wv[w] for w in wv.index_to_key if w != 'file'}

np.savez(r'data/word2vec_dict_dim_32.npz', **embedding_map)
# print(embedding_map.keys())
# np.savez(r'data/word2vec_dict.npz', **embedding_map)
#
# var = np.load(r'data/word2vec_dict.npz')
# print([x for x in var.keys()])
# print(var['target'])
# exit()

print('Loaded %s word vectors.' % len(embedding_map))


def corp_2_vec(corpus: list):
    count = 0
    mean_word2vec = np.zeros(DIMENSION)

    for _word in corpus:
        if _word in embedding_map.keys():
            count += 1
            mean_word2vec += embedding_map[_word]

    if count > 0:
        mean_word2vec = mean_word2vec / count
    return mean_word2vec


def preprocess_sentence(sentence):
    punc_stripped = [re.sub(r'[^a-zA-Z]+', ' ', j.lower().strip("\n").replace("\n", " ").rstrip('s')) for j in sentence]

    return [word for word in punc_stripped if word not in stripped_stopwords and len(word) > 2 and ' ' not in word]


def build_bag_of_words(card: dict):
    fields = ('name', 'oracle_text', 'type_line', 'flavor_text')
    words = []

    for f in fields:
        if f in card.keys():
            words += preprocess_sentence(card[f].split(" "))

    return card['name'], words


corpus_embedding = OrderedDict()
names = []
image_uris = []
vectors = []

for i, x in tqdm(enumerate(input_json[:30_000])):
    # if 'flavor_text' in x.keys() and 'oracle_text' in x.keys():
    if 'type_line' in x.keys():
        if 'image_uris' in x.keys() and 'small' in x['image_uris'].keys():
            name, corp = build_bag_of_words(x)

            vec = corp_2_vec(corp)

            corpus_embedding[name] = vec

            names.append(name)
            image_uris.append(x['image_uris']['small'])
            # Normalize vector
            vec = vec / np.linalg.norm(vec, ord=2)
            vectors.append(vec)


print(f"THERE ARE {len(vectors)}")


vec_array = np.nan_to_num(np.array(vectors), posinf=0, neginf=0, nan=0)
print(vec_array[0])

print(np.shape(vec_array))

# start_t = time.time()
# N = 100_000
# for i in range(N):
#     similarity_matrix = np.dot(vec_array[0], vec_array[0])
#
# end_t = time.time()
#
# print(end_t-start_t, f"TOTAL TIME TO DO {N} COSINE SIMILARITY CALCULATIONS WITH 100 DIM VECTORS")
# exit()
# similarity_matrix = cosine_similarity(vec_array)

similarity_matrix = np.dot(vec_array, vec_array.T)

print(np.max(similarity_matrix))
plt.hist(similarity_matrix.flatten())
plt.show()

print(np.shape(similarity_matrix))

# plt.figure(figsize=(10, 7))
# sn.heatmap(similarity_matrix, xticklabels=ordered_names, yticklabels=ordered_names,
#            annot=True, fmt='.2f')


def top_recommendations(sim_scores, noms, images):
    sorted_sim_scores = [b[0] for b in sorted(enumerate(sim_scores), reverse=True, key=lambda j: j[1])]

    return [(noms[y], sim_scores[y], images[y]) for y in sorted_sim_scores[0:8]]


rows = 10
gs = gridspec.GridSpec(rows, 8)
gs.update(wspace=0, hspace=0)

for i in range(len(similarity_matrix[:rows])):
    reccomendations = top_recommendations(sim_scores=similarity_matrix[i], noms=names, images=image_uris)
    for j, x in enumerate(reccomendations):
        print(j)
        ax = plt.subplot(gs[i, j])
        im = plt.imread(x[2], format='jpg')

        ax.imshow(im)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

plt.show()
exit()

#
# exit()

print(np.shape(np.array([x for x in corpus_embedding.values()])))
# similarity_matrix = cosine_similarity([x for x in corpus_embedding.values()], [x for x in corpus_embedding.values()])
print([row for row in similarity_matrix])
exit()
