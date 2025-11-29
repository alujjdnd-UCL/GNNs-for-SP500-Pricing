from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import time


def get_wikidata_mapping(tickers):
    """
    Maps cleaned tickers to Wikidata QIDs using SPARQL.
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    tickers_str = " ".join(f'"{ticker}"' for ticker in tickers)

    query = f"""
    SELECT ?ticker ?id WHERE {{
      ?id p:P414 [ ps:P414 ?exchange; pq:P249 ?ticker ].
      VALUES ?exchange {{ wd:Q13677 wd:Q82059 }}  # NYSE/NASDAQ
      VALUES ?ticker {{ {tickers_str} }}
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        mapping = {}
        for result in results["results"]["bindings"]:
            ticker = result["ticker"]["value"]
            qid = result["id"]["value"].split("/")[-1]  # Extract QID
            mapping[ticker] = qid
        return mapping
    except Exception as e:
        print(f"Error: {e}")
        return {}


def preprocess_ticker(ticker):
    """Remove suffixes like .K or .O."""
    return ticker.split('.')[0]


# Load your CSV
df = pd.read_csv("../../data/SP500/filtered_symbols.csv")
df["TickerClean"] = df["Symbol"].apply(preprocess_ticker)


# Fetch Wikidata QIDs in batches
def batch(iterable, n=50):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


all_mapping = {}
unique_tickers = df["TickerClean"].unique().tolist()

for ticker_batch in batch(unique_tickers, n=50):
    batch_mapping = get_wikidata_mapping(ticker_batch)
    all_mapping.update(batch_mapping)
    time.sleep(1)  # Avoid rate limits

# Map QIDs back to DataFrame
df["Wikidata"] = df["TickerClean"].map(all_mapping)
print(df.head())