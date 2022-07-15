import os
from extractor import Extractor


def main():
    url = os.environ.get('ELASTICSEARCH_URL', 'http://localhost:9200')
    print('Starting...')
    extractor = Extractor(url)
    extractor.run()
    print('Done')


if __name__ == '__main__':
    main()
