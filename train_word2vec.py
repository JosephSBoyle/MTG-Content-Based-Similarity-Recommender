import time

import json
import re
from gensim.models import Word2Vec, TfidfModel

from gensim.models.callbacks import CallbackAny2Vec
import matplotlib.pyplot as plt


class LossCallback(CallbackAny2Vec):
    """Callback to print loss after each epoch."""

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0
        self.loss_history = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        self.loss_history.append(loss_now)
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1


def preprocess_sentence(sentence):
    punc_stripped = [re.sub(r'[^a-zA-Z]+', ' ', j.lower().strip("\n").replace("\n", " ").rstrip('s')) for j in sentence]

    return [word for word in punc_stripped if word not in stripped_stopwords and len(word) > 2 and ' ' not in word]


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

stripped_stopwords = {x.rstrip("s") if len(x.rstrip("s")) > 2 else None for x in stopwords}
print(stripped_stopwords)

#input_json = json.load(open("data/default-cards-20210417090311.json", encoding='utf-8'))
"""only unique cards"""
input_json = json.load(open("data/oracle-cards-20210429090441.json", encoding='utf-8'))


sentences = []
fields = ('name', 'oracle_text', 'type_line', 'flavor_text')
for card in input_json:
    for f in fields:
        if f in card.keys():
            sentence = card[f].split(" ")
            sentence = preprocess_sentence(sentence)
            sentences.append(sentence)


print(f"There are:{sum([ len(listElem) for listElem in sentences])} total words in the training corpus and {len(sentences)} total sentences")

EPOCHS = 200
WINDOW_SIZE = 3
VECTOR_SIZE = 300

cb = LossCallback()
start_time = time.time()
model = Word2Vec(sentences, vector_size=VECTOR_SIZE, alpha=0.12, workers=16, min_count=10, epochs=EPOCHS, window=WINDOW_SIZE, compute_loss=True, callbacks=[cb])
end_time = time.time()
print("--- %s TOTAL MINUTES TRAINING TIME  ---" % ((time.time() - start_time) / 60))
plt.plot([x for x in range(len(cb.loss_history))], cb.loss_history)
plt.xlabel('Epoch')
plt.ylabel('Total Log. Loss')
plt.show()

# model = Word2Vec.load(f'./data/epochs_{EPOCHS}_window_{WINDOW_SIZE}_model.kv')

[print(x) for x in model.wv.key_to_index]
print(len(model.wv.key_to_index))
words = model.wv.key_to_index
print(len(words))

print(model.wv.most_similar(positive=['elf'], topn=10))
print(model.wv.most_similar(negative=['elf'], topn=10))
print("--")
print(model.wv.most_similar(positive=['bolt'], topn=10))
print(model.wv.most_similar(negative=['bolt'], topn=10))
print("--")
print(model.wv.most_similar(positive=['dragon'], topn=10))
print(model.wv.most_similar(negative=['dragon'], topn=10))
print("--")
print(model.wv.most_similar(positive=['eldrazi'], topn=10))
print(model.wv.most_similar(negative=['eldrazi'], topn=10))
print("--")
print(model.wv.most_similar(positive=['jeskai'], topn=10))
print(model.wv.most_similar(negative=['jeskai'], topn=10))
print("--")
print(model.wv.most_similar(positive=['instant', 'sorcery'], topn=10))
print(model.wv.most_similar(negative=['instant', 'sorcery'], topn=10))

print("########################################################")
print(model.wv.most_similar(positive=['elf'], topn=10, restrict_vocab=5_000))
print(model.wv.most_similar(negative=['elf'], topn=10, restrict_vocab=5_000))
print("--")
print(model.wv.most_similar(positive=['bolt'], topn=10, restrict_vocab=5_000))
print(model.wv.most_similar(negative=['bolt'], topn=10, restrict_vocab=5_000))
print("--")
print(model.wv.most_similar(positive=['dragon'], topn=10, restrict_vocab=5_000))
print(model.wv.most_similar(negative=['dragon'], topn=10, restrict_vocab=5_000))
print("--")
print(model.wv.most_similar(positive=['eldrazi'], topn=10, restrict_vocab=5_000))
print(model.wv.most_similar(negative=['eldrazi'], topn=10, restrict_vocab=5_000))
print("--")
print(model.wv.most_similar(positive=['jeskai'], topn=10, restrict_vocab=1_000))
print(model.wv.most_similar(negative=['jeskai'], topn=10, restrict_vocab=1_000))
print("--")
print(model.wv.most_similar(positive=['instant', 'sorcery'], topn=10))
print(model.wv.most_similar(negative=['instant', 'sorcery'], topn=10))


model.wv.save(f'./data/epochs_{EPOCHS}_window_{WINDOW_SIZE}_{VECTOR_SIZE}_vectors.kv')
model.save(f'./data/epochs_{EPOCHS}_window_{WINDOW_SIZE}_{VECTOR_SIZE}model.kv')

print(f'./data/epochs_{EPOCHS}_window_{WINDOW_SIZE}_{VECTOR_SIZE}_vectors.kv')
print(f'./data/epochs_{EPOCHS}_window_{WINDOW_SIZE}_{VECTOR_SIZE}model.kv')