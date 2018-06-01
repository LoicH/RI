
This folder contains the search engine for scientific articles and the diversity-based search engine for images.

## Search engine for scientific articles

Main notebook [here](https://github.com/LoicH/RI/blob/master/1-text/main.ipynb).

Overview of the code:
- [`indexation.py`](https://github.com/LoicH/RI/blob/master/1-text/indexation.py) parses the data input and stores it to make retrieval faster.
- [`modeles.py`](https://github.com/LoicH/RI/blob/master/1-text/modeles.py) is used to transform texts document and queries into vectors (tf-idf weights, binary weights...) and implements various ways of retrieving relevant results, such as unigram language, Okapi, PageRank, or HITS
- [`evaluation.py`](https://github.com/LoicH/RI/blob/master/1-text/evaluation.py) is used to benchmark our different models with metrics such as precision or recall.


## Diversity search engine

Main notebook [here](https://github.com/LoicH/RI/blob/master/1-text/part3.ipynb)
This search engine uses models in `modeles.py` to ensure the diversity of the results.
Example of algorithms: greedy algorithm, Precision-Recall clustering.

