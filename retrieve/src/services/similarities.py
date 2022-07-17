from gensim.models import KeyedVectors
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import hazm
import tqdm
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoConfig, AutoModel

20000 * 8 * 50000


class Similarities:

    def __init__(self, documents):

        documents = np.array(documents)
        self.documents = documents

        self.pipe = Pipeline([('count', CountVectorizer(analyzer='word', ngram_range=(1, 3), max_features=20_000)),
                              ('tfidf', TfidfTransformer(sublinear_tf=True))]).fit(documents)

        self.word2vec = KeyedVectors.load_word2vec_format('../resources/farsi_literature_word2vec_model.txt')

        self.word_idf = dict(zip(self.pipe['count'].get_feature_names_out(), self.pipe['tfidf'].idf_))

        self.boolean_vectors = self.pipe['count'].transform(documents).toarray().astype(bool).astype(int)
        self.tfidf_vectors = self.pipe.transform(documents).toarray()

        self.word_embedding = np.array(
            [self._word_embed(self.word2vec, self.word_idf, element) for element in documents])

        model_name = 'HooshvareLab/bert-base-parsbert-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, config=AutoConfig.from_pretrained(model_name))
        self.sent_embedding = self.get_transformer_embeddings(documents.tolist())

    @staticmethod
    def _batch_series(iterable, n=2_000):
        length = len(iterable)
        for ndx in range(0, length, n): yield iterable[ndx:min(ndx + n, length)]

    @staticmethod
    def _word_embed(word2vec, idf, element):
        def get_word2vector(word):
            return word2vec[word] if word in word2vec else np.zeros(100)

        return np.mean(
            [get_word2vector(wo) * idf.get(wo, 0) for wo in hazm.word_tokenize(element)], axis=0
        ).tolist()

    def get_transformer_embeddings(self, documents):

        result = None
        for batch in tqdm.tqdm(self._batch_series(documents, 2_000)):
            output = self.model(**self.tokenizer(batch, return_tensors='pt', padding=True))
            output = np.mean(output.last_hidden_state.detach().numpy(), axis=1)
            result = np.concatenate((result, output)) if result is not None else output

        return result

    @staticmethod
    def get_similar_by_cosine_distance(vector, documents, n=5):
        sq_vector = np.squeeze(vector)
        similarity = documents.dot(sq_vector) / (np.linalg.norm(documents, axis=1) * np.linalg.norm(sq_vector) + 1e-10)
        sorted_idx = np.argsort(similarity)

        return sorted_idx[-n:], similarity[sorted_idx[-n:]]

    def get_similar_by_tfidf(self, text, n):
        tfidf_vector = self.pipe.transform([text]).toarray()
        idx, _dist = self.get_similar_by_cosine_distance(tfidf_vector, self.tfidf_vectors, n)
        return list(zip(self.documents[idx], _dist))

    def get_similar_by_boolean(self, text, n):
        boolean_vector = self.pipe['count'].transform([text]).toarray().astype(bool).astype(int)
        idx, _dist = self.get_similar_by_cosine_distance(boolean_vector, self.boolean_vectors, n)
        return list(zip(self.documents[idx], _dist))

    def get_similar_by_word_embedding(self, text, n):
        word_embedding = self._word_embed(self.word2vec, self.word_idf, text)
        idx, _dist = self.get_similar_by_cosine_distance(word_embedding, self.word_embedding, n)
        return list(zip(self.documents[idx], _dist))

    def get_similar_by_sentence_embedding(self, text, n):
        sent_embedding = self.get_transformer_embeddings([text])
        idx, _dist = self.get_similar_by_cosine_distance(sent_embedding, self.sent_embedding, n)
        return list(zip(self.documents[idx], _dist))


if __name__ == '__main__':

    normalizer = hazm.Normalizer(token_based=True)

    poems = pd.read_csv('../resources/src-dataset.csv')['text']
    poems = poems.apply(normalizer.normalize).to_numpy()
    np.random.shuffle(poems)

    sample, poems = poems[0], poems[1:]
    model = Similarities(poems)

    print('-' * 100)
    for poem, dist in model.get_similar_by_tfidf(sample, 10):
        print("tfidf: {:50s} \t with similarity of {:.2f}".format(poem, dist))

    print('-' * 100)
    for poem, dist in model.get_similar_by_boolean(sample, 10):
        print("bools: {:50s} \t with similarity of {:.2f}".format(poem, dist))

    print('-' * 100)
    for poem, dist in model.get_similar_by_word_embedding(sample, 10):
        print("word: {:50s} \t with similarity of {:.2f}".format(poem, dist))

    print('-' * 100)
    for poem, dist in model.get_similar_by_sentence_embedding(sample, 10):
        print("sent: {:50s} \t with similarity of {:.2f}".format(poem, dist))
