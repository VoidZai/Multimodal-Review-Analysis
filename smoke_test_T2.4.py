import pandas as pd

from cragb.retrieval.bm25 import BM25Retriever


df = pd.read_parquet(
    "data/processed/corpus_v1.parquet"
)

retriever = BM25Retriever()

retriever.index(df)


queries = [
    "shirt too small",
    "colour faded after washing",
    "good quality jacket"
]


for q in queries:

    print("="*50)
    print("QUERY:", q)

    results = retriever.search(q, k=5)

    print("Number of results:", len(results))

    for result in results:

        row = df.loc[int(result.doc_id)]

        print("\nRank:", result.rank)
        print("Score:", round(result.score, 3))
        print("Rating:", row["rating"])
        print("Text:", row["text"][:150])