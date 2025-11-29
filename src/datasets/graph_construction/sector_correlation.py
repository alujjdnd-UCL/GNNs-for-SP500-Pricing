import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import time
import os
import math
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON

############################
# --- BASIC CACHING SETUP ---
############################

CACHE_FILE_QIDS = "wikidata_symbol_qids.json"    # For symbol→QID caching
WIKIDATA_RELATIONS_CACHE_FILE = "wikidata_relations_adj.npy"  # For adjacency from P127, P199, P190
FINAL_ADJ_CACHE_FILE = "final_adj.npy"           # Final adjacency (if you wish to use)
FINAL_ADJ_CSV_FILE = "final_adj.csv"


def _resolve_cache_path(file_name: str, cache_dir: Optional[str] = None) -> str:
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return os.path.join(cache_dir, file_name)
    return file_name


def load_cached_qids(cache_path=CACHE_FILE_QIDS):
    """Load cached symbol→QID mappings from a JSON file, if it exists."""
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cached_qids(symbol_qid_map, cache_path=CACHE_FILE_QIDS):
    """Save symbol→QID mappings to a JSON file."""
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(symbol_qid_map, f, ensure_ascii=False, indent=2)


############################
# --- SPARQL HELPERS ---
############################

EXCHANGE_QIDS = {
    "NASDAQ": "Q13677",   # NASDAQ
    "NYSE": "Q13678",     # New York Stock Exchange
    "ASX": "Q82059",      # Australian Securities Exchange
    "LSE": "Q82059",      # London Stock Exchange (example)
    "TSE": "Q82059",      # Tokyo Stock Exchange (example)
    "EURONEXT": "Q463391" # Euronext
}


def strip_exchange_suffix(symbol):
    """Remove common exchange suffixes (.O, .N, .TO) for better matching."""
    return symbol.split('.')[0] if '.' in symbol else symbol


def get_wikidata_id(symbol,
                    exchange="NASDAQ",
                    user_agent="MyWikidataBot/1.0 (https://example.org/contact)",
                    cache_dir: Optional[str] = None,
                    allow_external_requests: bool = True,
                    verbose: bool = False):
    """
    Retrieve the Wikidata QID for a single symbol, or a batch of symbols.
    - If symbol is a string, return a single QID (or None).
    - If symbol is a list/tuple/set, return {sym -> QID or None}.

    This uses local caching (CACHE_FILE_QIDS) to avoid repeated queries.
    """
    cache_path = _resolve_cache_path(CACHE_FILE_QIDS, cache_dir)
    cached_qids = load_cached_qids(cache_path)
    exchange_qid = EXCHANGE_QIDS.get(exchange.upper(), "Q13677")  # default NASDAQ

    # --- Single symbol case ---
    if isinstance(symbol, str):
        stripped_sym = strip_exchange_suffix(symbol)
        if stripped_sym in cached_qids:
            return cached_qids[stripped_sym]

        # Not in cache → do a mini-batch of size 1
        results_dict = get_wikidata_id([symbol], exchange, user_agent, cache_dir=cache_dir,
                                       allow_external_requests=allow_external_requests, verbose=verbose)
        return results_dict.get(symbol, None)

    # --- Collection of symbols case ---
    if not isinstance(symbol, (list, tuple, set)):
        raise TypeError("`symbol` must be a string or list/tuple/set of strings.")

    symbols_list = list(symbol)
    stripped_symbols = [strip_exchange_suffix(s) for s in symbols_list]

    # Filter out those already in cache
    to_query = []
    for i, stripped_sym in enumerate(stripped_symbols):
        if stripped_sym not in cached_qids:
            to_query.append(symbols_list[i])  # original symbol with suffix

    # If everything is found, build the result directly
    if not to_query:
        return {sym: cached_qids.get(strip_exchange_suffix(sym), None) for sym in symbols_list}

    if not allow_external_requests:
        # Populate newly requested symbols with None and persist cache
        for stripped_sym in to_query_stripped:
            cached_qids.setdefault(stripped_sym, None)
        save_cached_qids(cached_qids, cache_path)
        if verbose:
            print(f"[Wikidata] External requests disabled. Returning cached QIDs only for {len(symbols_list)} symbols.")
        return {sym: cached_qids.get(strip_exchange_suffix(sym), None) for sym in symbols_list}

    # Build SPARQL: match any "P249 = ticker" (on any exchange) –
    # you can add ps:P414 wd:Q13677 if you want to force only NASDAQ,
    # but here's the flexible approach that picks up ANY listing with that ticker:
    to_query_stripped = [strip_exchange_suffix(s) for s in to_query]
    values_str = " ".join(f"\"{sym}\"" for sym in to_query_stripped)
    query = f"""
    SELECT ?ticket ?company WHERE {{
      VALUES ?ticket {{ {values_str} }}
      ?company p:P414 ?exchangeStatement.
      ?exchangeStatement pq:P249 ?ticket.
    }}
    """

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent=user_agent)
    sparql.setTimeout(30)
    sparql.setMethod('POST')
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        time.sleep(0.5)
        bindings = results.get("results", {}).get("bindings", [])

        # symbol -> QID from this batch
        symbol_to_qid = {}
        for row in bindings:
            found_symbol = row["ticket"]["value"]  # e.g. "AAPL"
            found_qid = row["company"]["value"].split('/')[-1]
            symbol_to_qid[found_symbol] = found_qid

        # Update the cache
        for s in to_query_stripped:
            if s in symbol_to_qid:
                cached_qids[s] = symbol_to_qid[s]
            else:
                # store None so we won't re-query for the same symbol
                cached_qids[s] = None

        save_cached_qids(cached_qids, cache_path)

    except Exception as e:
        if verbose:
            print(f"[Wikidata] Error retrieving QID for batch {to_query}: {e}")
        # Mark as None in the cache
        for s in to_query_stripped:
            if s not in cached_qids:
                cached_qids[s] = None
        save_cached_qids(cached_qids, cache_path)

    # Build output for the full set
    output_dict = {}
    for s_orig, s_stripped in zip(symbols_list, stripped_symbols):
        output_dict[s_orig] = cached_qids.get(s_stripped, None)

    return output_dict


############################
# --- GENERIC PROPERTY ADJACENCY ---
############################

def batch_property_query(qid_list_1, qid_list_2, property_id,
                         user_agent="MyWikidataBot/1.0 (https://example.org/contact)",
                         verbose: bool = False):
    """
    Query Wikidata for all pairs (qidA, qidB) in qid_list_1 × qid_list_2
    such that qidA has a relationship via `property_id` to qidB
    OR qidB has a relationship via `property_id` to qidA.

    Example property_ids:
    - P127 (owned by)
    - P199 (supplier)
    - P190 (partner) but note it's often "twin cities" for city items, not always companies.

    Returns a list of (qidA, qidB) pairs that match in either direction.
    """
    if not qid_list_1 or not qid_list_2:
        return []
    values_str_1 = " ".join(f"wd:{q}" for q in qid_list_1)
    values_str_2 = " ".join(f"wd:{q}" for q in qid_list_2)

    query = f"""
    SELECT ?company1 ?company2 WHERE {{
      VALUES ?company1 {{ {values_str_1} }}
      VALUES ?company2 {{ {values_str_2} }}

      {{
        ?company1 wdt:{property_id} ?company2 .
        FILTER(?company1 != ?company2)
      }}
      UNION
      {{
        ?company2 wdt:{property_id} ?company1 .
        FILTER(?company1 != ?company2)
      }}
    }}
    """

    sparql = SPARQLWrapper("https://query.wikidata.org/sparql", agent=user_agent)
    sparql.setTimeout(60)
    sparql.setMethod('POST')
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    pairs = []
    try:
        results = sparql.query().convert()
        time.sleep(0.5)
        for binding in results.get("results", {}).get("bindings", []):
            comp1 = binding["company1"]["value"].rsplit("/", 1)[-1]
            comp2 = binding["company2"]["value"].rsplit("/", 1)[-1]
            pairs.append((comp1, comp2))
    except Exception as e:
        if verbose:
            print(f"[Wikidata] Error in batch_property_query (P={property_id}): {e}")
    return pairs


def get_property_adjacency_matrix_chunked(stock_symbols, symbol_qid_map,
                                          property_id,
                                          chunk_size=5,
                                          user_agent="MyWikidataBot/1.0 (https://example.org/contact)",
                                          allow_external_requests: bool = True,
                                          verbose: bool = False):
    """
    Build an adjacency matrix (N×N) for the given Wikidata property (e.g. P127, P199, P190).
    - If any pair of companies (A, B) has that property in either direction,
      we set adj[A,B] = 1 and adj[B,A] = 1.

    :param stock_symbols: list of symbols, used to define adjacency order
    :param symbol_qid_map: dict of {symbol -> QID}
    :param property_id: e.g. "P127" (owned by), "P199" (supplier), "P190" (partner)
    :return: an N×N NumPy array of 0/1
    """
    n = len(stock_symbols)
    adj = np.zeros((n, n), dtype=int)

    # Build (qid, index_in_adjacency) pairs
    qid_index_pairs = []
    for idx, sym in enumerate(stock_symbols):
        qid = symbol_qid_map.get(sym)
        if qid:
            qid_index_pairs.append((qid, idx))

    # If not enough QIDs are known, there's nothing to link
    if len(qid_index_pairs) < 2:
        return adj

    # Local chunker
    def chunk_list(lst, sz):
        for i in range(0, len(lst), sz):
            yield lst[i : i + sz]

    chunked_list = list(chunk_list(qid_index_pairs, chunk_size))
    pairs_of_chunks = [(i, j) for i in range(len(chunked_list)) for j in range(i, len(chunked_list))]

    if not allow_external_requests:
        return adj

    for (i, j) in tqdm(pairs_of_chunks, desc=f"Chunked {property_id} queries", disable=not verbose):
        chunk_i = chunked_list[i]
        chunk_j = chunked_list[j]
        qids_i = [c[0] for c in chunk_i]
        qids_j = [c[0] for c in chunk_j]

        # Query for relationships
        pairs = batch_property_query(qids_i, qids_j, property_id, user_agent)

        # Build a quick lookup: QID -> adjacency index
        dict_i = {q: idx for (q, idx) in chunk_i}
        dict_j = {q: idx for (q, idx) in chunk_j}

        # Mark adjacency = 1
        for (qidA, qidB) in pairs:
            if qidA in dict_i and qidB in dict_j:
                a = dict_i[qidA]
                b = dict_j[qidB]
            elif qidA in dict_j and qidB in dict_i:
                a = dict_j[qidA]
                b = dict_i[qidB]
            else:
                continue
            adj[a, b] = 1
            adj[b, a] = 1

    return adj


############################
# --- MAIN GRAPH FUNCTION ---
############################

def get_sector_graph_wikidata(symbol_industry_map_pd,
                              stocks_prices_pd,
                              threshold=0.95,
                              visualise=False,
                              save_adjacency_matrix_path=None,
                              save_as_csv=False,
                              what_edges_to_form=None,
                              exchange="NASDAQ",
                              user_agent="MyWikidataBot/1.0 (https://example.org/contact)",
                              chunk_size=50,
                              indicator_pivot_pds=None,
                              enable_wikidata: bool = True,
                              allow_external_requests: bool = True,
                              cache_dir: Optional[str] = None,
                              verbose: bool = False):
    """
    Create a stock graph by combining adjacency from:
      - Shared Owner (P127) if "wikidata" in what_edges_to_form
      - Supplier (P199) if "supplier" in what_edges_to_form
      - Partner (P190) if "partner" in what_edges_to_form
      - Sector adjacency ("sectors")
      - Price correlation adjacency ("correlation")

    The combined adjacency matrix is returned as a NumPy array.
    It may also be cached to disk (FINAL_ADJ_CACHE_FILE) and optionally saved to CSV.

    :param symbol_industry_map_pd: DataFrame with index=stock symbols, column='Sector'
    :param stocks_prices_pd: DataFrame of historical stock prices (columns = symbols)
    :param threshold: correlation threshold for adjacency
    :param visualise: whether to visualize with NetworkX
    :param save_adjacency_matrix_path: optional .npy path to save the adjacency
    :param save_as_csv: if True, also saves adjacency as CSV
    :param what_edges_to_form: list of edges to include, e.g. ["wikidata","sectors","correlation","supplier","partner"]
    :param exchange: default "NASDAQ" – used for get_wikidata_id calls
    :param chunk_size: chunk size for property queries
    :param user_agent: user-agent for SPARQL
    :return: combined adjacency (NumPy array, shape N×N)
    """
    if what_edges_to_form is None:
        # Default to the original three
        what_edges_to_form = ["wikidata", "sectors", "correlation"]

    # 1) Identify symbols that exist in both DataFrames
    common_symbols = list(symbol_industry_map_pd.index.intersection(stocks_prices_pd.columns))
    n = len(common_symbols)
    if n < 2:
        print("Not enough symbols overlap between sector info and price data.")
        return np.zeros((n, n), dtype=int)

    # 2) Retrieve QIDs in a single batch
    if verbose:
        print("Retrieving Wikidata QIDs for each symbol (in batch)...")
    batch_qid_results = get_wikidata_id(common_symbols, exchange, user_agent,
                                        cache_dir=cache_dir,
                                        allow_external_requests=allow_external_requests,
                                        verbose=verbose)
    symbol_qid_map = {sym: batch_qid_results.get(sym, None) for sym in common_symbols}

    # 3) Initialize adjacency with zeros
    adj_combined = np.zeros((n, n), dtype=int)

    # 4) (A) If "wikidata" in edges, either load or compute (P127/P199/P190) adjacency from cache
    if "wikidata" in what_edges_to_form and enable_wikidata:
        if verbose:
            print("Checking if we can load cached Wikidata adjacency (P127, P199, P190)...")

        relations_cache_path = _resolve_cache_path(WIKIDATA_RELATIONS_CACHE_FILE, cache_dir)

        # If the file exists and shape matches our symbol set, load it
        if os.path.exists(relations_cache_path):
            cached_adj = np.load(relations_cache_path)
            if cached_adj.shape == (n, n):
                if verbose:
                    print("Loaded cached Wikidata adjacency from disk.")
                adj_wiki = cached_adj
            else:
                if verbose:
                    print("Cached adjacency shape does not match current symbol list. Recomputing...")
                adj_wiki = build_wikidata_relations_matrix(
                    common_symbols, symbol_qid_map, chunk_size, user_agent,
                    cache_dir=cache_dir,
                    allow_external_requests=allow_external_requests,
                    verbose=verbose
                )
                np.save(relations_cache_path, adj_wiki)
        else:
            if allow_external_requests:
                # File doesn't exist → build and cache
                adj_wiki = build_wikidata_relations_matrix(
                    common_symbols, symbol_qid_map, chunk_size, user_agent,
                    cache_dir=cache_dir,
                    allow_external_requests=allow_external_requests,
                    verbose=verbose
                )
                np.save(relations_cache_path, adj_wiki)
            else:
                if verbose:
                    print("[Wikidata] External requests disabled and no cached adjacency available. Skipping Wikidata edges.")
                adj_wiki = np.zeros((n, n), dtype=int)

        # Merge
        adj_combined = np.clip(adj_combined + adj_wiki, 0, 1)

    # 4) (B) Sector adjacency
    if "sectors" in what_edges_to_form:
        if verbose:
            print("Building adjacency for shared Sector ...")
        adj_sectors = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i+1, n):
                s1 = common_symbols[i]
                s2 = common_symbols[j]
                if (symbol_industry_map_pd.loc[s1, 'Sector']
                    == symbol_industry_map_pd.loc[s2, 'Sector']
                    and s1 != s2):
                    adj_sectors[i, j] = 1
                    adj_sectors[j, i] = 1
        adj_combined = np.clip(adj_combined + adj_sectors, 0, 1)

    # 4) (C) Correlation adjacency
    if "correlation" in what_edges_to_form:
        if verbose:
            print("Building adjacency for correlation ...")
        # Use partial slice or entire price data
        prices = stocks_prices_pd.loc[:, common_symbols]
        subset_size = int(len(prices) * 0.75)
        corr_matrix = prices.iloc[:subset_size].corr().to_numpy()
        adj_correlation = (corr_matrix > threshold).astype(int)
        np.fill_diagonal(adj_correlation, 0)
        adj_combined = np.clip(adj_combined + adj_correlation, 0, 1)

        if indicator_pivot_pds is not None:
            for indicator_pd in indicator_pivot_pds:
                value = indicator_pd.loc[:, common_symbols]
                subset_size = int(len(value) * 0.75)
                corr_matrix = value.iloc[:subset_size].corr().to_numpy()
                adj_correlation = (corr_matrix > threshold).astype(int)
                np.fill_diagonal(adj_correlation, 0)
                adj_combined = np.clip(adj_combined + adj_correlation, 0, 1)

    # 5) Optionally save to .npy and/or CSV
    if save_adjacency_matrix_path:
        np.save(save_adjacency_matrix_path, adj_combined)
    if save_as_csv:
        final_csv_path = _resolve_cache_path(FINAL_ADJ_CSV_FILE, cache_dir)
        df_final = pd.DataFrame(adj_combined, index=common_symbols, columns=common_symbols)
        df_final.to_csv(final_csv_path)

    # 6) (Optional) Visualize the final graph
    if visualise and n > 1:
        print("Visualizing the final combined graph ...")
        graph = nx.from_numpy_array(adj_combined)
        graph = nx.relabel_nodes(graph, dict(enumerate(common_symbols)))

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph, k=0.5, seed=42)
        nx.draw(
            graph, pos, with_labels=True, node_size=500,
            font_size=8, font_weight='bold', font_color='black'
        )
        plt.title("Combined Stock Graph: " + ", ".join(what_edges_to_form))
        plt.show()

    # 7) Print number of edges
    num_edges = np.count_nonzero(np.triu(adj_combined, k=1))
    if verbose:
        print(f"[Result] Number of edges among {n} symbols: {num_edges}")

    return adj_combined


def build_wikidata_relations_matrix(common_symbols, symbol_qid_map, chunk_size, user_agent,
                                    cache_dir: Optional[str] = None,
                                    allow_external_requests: bool = True,
                                    verbose: bool = False):
    """
    Builds the adjacency from P127, P199, and P190 for the given symbols.
    Combines them in a single adjacency matrix, which we then can cache.
    """
    n = len(common_symbols)
    # Start with all zeros
    adj_wiki = np.zeros((n, n), dtype=int)

    # P127: Owned by
    if verbose:
        print("Building adjacency for Shared Owner (P127) ...")
    adj_p127 = get_property_adjacency_matrix_chunked(
        stock_symbols=common_symbols,
        symbol_qid_map=symbol_qid_map,
        property_id="P127",
        chunk_size=chunk_size,
        user_agent=user_agent,
        allow_external_requests=allow_external_requests,
        verbose=verbose
    )
    adj_wiki = np.clip(adj_wiki + adj_p127, 0, 1)

    # P199: Organisational Divisions
    if verbose:
        print("Building adjacency for Org Divs (P199) ...")
    adj_p199 = get_property_adjacency_matrix_chunked(
        stock_symbols=common_symbols,
        symbol_qid_map=symbol_qid_map,
        property_id="P199",
        chunk_size=chunk_size,
        user_agent=user_agent,
        allow_external_requests=allow_external_requests,
        verbose=verbose
    )
    adj_wiki = np.clip(adj_wiki + adj_p199, 0, 1)

    return adj_wiki
