import hazm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import pickle
import tqdm
import os
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel, AutoConfig

sns.set()
logger = logging.getLogger(__name__)


class Clustering:

    def __init__(self, documents, k, pca_dim=8, max_iter=2_000, _checkpoint=None):

        model_name = 'HooshvareLab/bert-base-parsbert-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=True)
        self.embedder = AutoModel.from_pretrained(
            model_name, local_files_only=True, config=AutoConfig.from_pretrained(model_name))

        self.directory = _checkpoint
        self.documents = np.array(documents)

        try:

            self.load_pca()
            self.load_kmeans()
            return

        except:
            logger.warning('It is not possible to load models. Again the models are trained.')

        self.embeddings = self.get_transformer_embeddings(self.documents.tolist())

        self.pca = PCA(n_components=pca_dim)
        self.embeddings = self.pca.fit_transform(self.embeddings)

        self.kmeans = KMeans(n_clusters=k, max_iter=max_iter)
        self.labels = self.kmeans.fit_predict(self.embeddings)

        self.directory = '../../resources/clustering/'
        if not os.path.exists(self.directory): os.mkdir(self.directory)

        self.save_pca()
        self.save_kmeans()

    @staticmethod
    def _batch_series(iterable, n=2_000):
        for ndx in range(0, len(iterable), n): yield iterable[ndx:min(ndx + n, len(iterable))]

    def get_transformer_embeddings(self, documents):

        result = None
        for batch in tqdm.tqdm(self._batch_series(documents, 2_000)):
            output = self.embedder(**self.tokenizer(batch, return_tensors='pt', padding=True))
            output = np.mean(output.last_hidden_state.detach().numpy(), axis=1)
            result = np.concatenate((result, output)) if result is not None else output

        return result

    def load_pca(self):
        self.pca = pickle.load(
            open(os.path.join(self.directory, 'pca_model.dump'), 'rb'))
        self.embeddings = pickle.load(
            open(os.path.join(self.directory, 'pca_embeddings.dump'), 'rb'))

    def save_pca(self):
        pickle.dump(
            self.pca, open(os.path.join(self.directory, 'pca_model.dump'), 'wb'))
        pickle.dump(
            self.embeddings, open(os.path.join(self.directory, 'pca_embeddings.dump'), 'wb'))

    def load_kmeans(self):
        self.kmeans = pickle.load(
            open(os.path.join(self.directory, 'kmeans_model.dump'), 'rb'))
        self.labels = pickle.load(
            open(os.path.join(self.directory, 'kmeans_labels.dump'), 'rb'))

    def save_kmeans(self):
        pickle.dump(
            self.kmeans, open(os.path.join(self.directory, 'kmeans_model.dump'), 'wb'))
        pickle.dump(
            self.labels, open(os.path.join(self.directory, 'kmeans_labels.dump'), 'wb'))

    def predict_cluster(self, element):
        embedding = self.get_transformer_embeddings([element])
        embedding = self.pca.transform(embedding)
        return self.kmeans.predict(embedding)

    def plot_clusters(self, n=1_000):

        pca = PCA(n_components=2)
        sample = np.random.randint(0, self.embeddings.shape[0], n)
        mini_embeddings = pca.fit_transform(self.embeddings[sample])

        sns.scatterplot(
            x=mini_embeddings[:, 0],
            y=mini_embeddings[:, 1], c=self.labels[sample], cmap='cool')

        sns.scatterplot(
            x=self.kmeans.cluster_centers_[:, 0],
            y=self.kmeans.cluster_centers_[:, 1], c=['black'])

        plt.show()


if __name__ == '__main__':
    normalizer = hazm.Normalizer(token_based=True)

    poems = pd.read_csv('../../resources/shahnameh-dataset.csv')['text']
    poems = poems.apply(normalizer.normalize)

    checkpoint = '../../resources/clustering'
    Clustering(poems, 9, _checkpoint=None).plot_clusters()
