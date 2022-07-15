import io
from tqdm import tqdm
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch

with io.open('data/shahnameh-ferdosi.htm', 'r', encoding='utf-8') as file:
    html = file.read()


def filter_poems(tag):
    return tag.name == 'span' and tag.has_attr('class') and 'content_text' in tag.get('class')


def filter_labels(tag):
    return tag.name == 'h2' and tag.has_attr('class') and 'content_h2' in tag.get('class')


def filter_poems_labels(tag):
    return filter_poems(tag) or filter_labels(tag)


class Extractor:

    def __init__(self, es_host):
        self.es = Elasticsearch(hosts=es_host)
        self.skip_labels = ['مشخصات کتاب', 'معرفی']

    def write_to_es(self, mesra1, mesra2, label):
        doc = {
            'mesra1': mesra1,
            'mesra2': mesra2,
            'beyt': ' '.join([mesra1, mesra2]),
            'label': label,
        }
        self.es.index(index='ferdosi', body=doc)

    def extract(self):
        print('Extracting...')
        soup = BeautifulSoup(html, 'html.parser')
        poems_and_labels = soup.find_all(filter_poems_labels)

        buffered_text = None

        label = None
        for item in tqdm(poems_and_labels):
            if filter_labels(item):
                label = item.get_text()
            elif filter_poems(item) and label and label not in self.skip_labels:
                text = item.get_text()
                if '****' not in text:
                    buffered_text = text
                    continue

                if buffered_text:
                    text = ' '.join([buffered_text, text])
                    buffered_text = None

                mesras = text.split('****')
                self.write_to_es(mesras[0], mesras[1], label)

    def run(self):
        try:
            self.es.indices.get(index='ferdosi')
            print('Index already exists')
        except:
            self.extract()
