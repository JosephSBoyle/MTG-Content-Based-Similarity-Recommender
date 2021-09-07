from gensim.models import Word2Vec

model = Word2Vec.load(f'./data/epochs_{10000}_window_{3}_model.kv')

[print(x) for x in model.wv.key_to_index]
print(len(model.wv.key_to_index))