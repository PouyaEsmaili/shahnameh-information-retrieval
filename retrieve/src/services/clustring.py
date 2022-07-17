import hazm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import tqdm
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel, AutoConfig

sns.set()


class Clustering:

    def __init__(self, documents, k):

        model_name = 'HooshvareLab/bert-base-parsbert-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedder = AutoModel.from_pretrained(model_name, config=AutoConfig.from_pretrained(model_name))

        self.documents = np.array(documents)

        self.embeddings = self.get_transformer_embeddings(self.documents.tolist())

        self.pca = PCA(n_components=5)
        self.embeddings = self.pca.fit_transform(self.embeddings)

        self.model = KMeans(n_clusters=k, max_iter=1_000)
        self.labels = self.model.fit_predict(self.embeddings)

    @staticmethod
    def _batch_series(iterable, n=2_000):
        length = len(iterable)
        for ndx in range(0, length, n): yield iterable[ndx:min(ndx + n, length)]

    def get_transformer_embeddings(self, documents):

        result = None
        for batch in tqdm.tqdm(self._batch_series(documents, 2_000)):
            output = self.embedder(**self.tokenizer(batch, return_tensors='pt', padding=True))
            output = np.mean(output.last_hidden_state.detach().numpy(), axis=1)
            result = np.concatenate((result, output)) if result is not None else output

        return result

    def predict_cluster(self, element):
        embedding = self.get_transformer_embeddings([element])
        embedding = self.pca.transform(embedding)
        return self.model.predict(embedding)

    def plot_clusters(self):

        minis_embeddings = TSNE(
            n_components=2, learning_rate='auto', init='random').fit_transform(self.embeddings)

        centers = self.model.cluster_centers_
        sns.scatterplot(
            x=minis_embeddings[:, 0],
            y=minis_embeddings[:, 1],
            c=self.labels,
            cmap='cool')

        sns.scatterplot(x=centers[:, 0], y=centers[:, 1], c=['black'])
        plt.show()


if __name__ == '__main__':

    normalizer = hazm.Normalizer(token_based=True)

    poems = pd.read_csv('../resources/src-dataset.csv')['text'].sample(1_000)
    poems = poems.apply(normalizer.normalize)
    Clustering(poems, 9).plot_clusters()
