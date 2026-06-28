#!/usr/bin/env python3
"""
Pipeline to scrape Wikipedia articles based on question prompts,
chunk the text using LangChain, and build a FAISS index with embeddings.
"""

import argparse
import json
import os
import re
import time

import faiss
import numpy as np
import pandas as pd
import requests
import wikipedia
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


BOILERPLATE_REGEX = re.compile(
    r'^(Pick the best possible answer|Select the most accurate option|Determine the correct option|'
    r'Identify the correct statement|Which of the following is correct|Choose the correct answer):\s*',
    flags=re.IGNORECASE,
)

QUALIFIER_REGEX = re.compile(
    r'\s+(used for|in physics|in astrophysics|in quantum mechanics|in canonical quantization|'
    r'based on|from the|among the|according to|from the following choices|carefully|from the options below).*$',
    flags=re.IGNORECASE,
)

UNWANTED_SECTIONS = {
    "see also",
    "references",
    "external links",
    "further reading",
    "notes",
    "bibliography",
    "explanatory notes",
    "citations",
    "discography",
}

HEADERS = {
    'User-Agent': 'ScienceExamScraper/1.0 (contact: research@example.com)'
}
wikipedia.set_user_agent('ScienceExamScraper/1.0 (contact: research@example.com)')


def extract_core_topic(prompt: str) -> str:
    """Extract the main topic or entity from a question prompt."""
    cleaned = BOILERPLATE_REGEX.sub('', prompt).strip()

    # Look for "What is/are <topic>?"
    match = re.search(
        r'What (?:is|are|was|were)\s+(?:the|a|an)?\s*([^?\n,]+)',
        cleaned,
        flags=re.IGNORECASE,
    )
    if match:
        topic = match.group(1).strip()
        topic = QUALIFIER_REGEX.sub('', topic).strip()
        if len(topic) >= 3:
            return topic

    # Look for "Which [of the following] <topic>?"
    match2 = re.search(
        r'(?:Which|Identify|Determine)\s+(?:of the following is|is|the)?\s*([^?\n,]+)',
        cleaned,
        flags=re.IGNORECASE,
    )
    if match2:
        topic = match2.group(1).strip()
        topic = QUALIFIER_REGEX.sub('', topic).strip()
        if len(topic) >= 3:
            return topic

    return QUALIFIER_REGEX.sub('', cleaned).strip()


def clean_wikipedia_content(text: str) -> str:
    """Remove references, unwanted sections, inline citations, and extra whitespace."""
    if not text:
        return ""

    lines = text.split('\n')
    cleaned_lines = []
    skip_section = False

    for line in lines:
        stripped = line.strip()
        match = re.match(r'^={2,6}\s*(.*?)\s*={2,6}$', stripped)
        if match:
            section_title = match.group(1).lower()
            if any(unwanted in section_title for unwanted in UNWANTED_SECTIONS):
                skip_section = True
                continue
            else:
                skip_section = False

        if not skip_section:
            cleaned_lines.append(line)

    content = "\n".join(cleaned_lines)

    # Strip bracketed citations, HTML tags, and file links
    content = re.sub(r'\[\d+\]', '', content)
    content = re.sub(r'\[citation needed\]', '', content, flags=re.IGNORECASE)
    content = re.sub(r'<[^>]+>', '', content)
    content = re.sub(r'\[\[File:.*?\]\]', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\[\[Image:.*?\]\]', '', content, flags=re.IGNORECASE)

    # Normalize newlines and spaces
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = re.sub(r'[ \t]+', ' ', content)

    return content.strip()


def search_wikipedia_topics(query: str, max_results: int = 2) -> list[str]:
    """Search Wikipedia API for relevant article titles."""
    url = "https://en.wikipedia.org/w/api.php"

    core_topic = extract_core_topic(query)

    # Fallback to key words if core topic search is empty
    stop_words = {
        'what', 'is', 'are', 'the', 'term', 'used', 'in', 'to', 'describe',
        'resulting', 'from', 'of', 'and', 'or', 'a', 'an', 'which', 'select',
        'identify', 'choose', 'correct', 'option', 'answer',
    }
    clean_words = [
        w for w in re.sub(r'[^\w\s-]', '', query).split()
        if w.lower() not in stop_words
    ]
    keyword_query = " ".join(clean_words[:8]) if clean_words else query

    search_queries = [core_topic]
    if keyword_query != core_topic:
        search_queries.append(keyword_query)

    for q in search_queries:
        if not q.strip():
            continue
        params = {
            "action": "query",
            "list": "search",
            "srsearch": q,
            "format": "json",
            "srlimit": max_results,
        }

        for attempt in range(3):
            try:
                res = requests.get(url, headers=HEADERS, params=params, timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    search_results = data.get("query", {}).get("search", [])
                    titles = [item["title"] for item in search_results]
                    if titles:
                        return titles
                    break
                elif res.status_code == 429:
                    print(f"Rate limited (HTTP 429). Retrying in 2s (Attempt {attempt + 1}/3)...")
                    time.sleep(2.0)
            except Exception:
                time.sleep(0.5)

    return []


def fetch_wikipedia_article(title: str) -> str | None:
    """Fetch and clean full Wikipedia article text."""
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return clean_wikipedia_content(page.content)
    except wikipedia.DisambiguationError as e:
        try:
            if e.options:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                return clean_wikipedia_content(page.content)
        except Exception:
            return None
    except Exception:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": True,
            "titles": title,
            "format": "json",
        }
        try:
            res = requests.get(url, headers=HEADERS, params=params, timeout=5)
            pages = res.json().get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id != "-1":
                    raw_text = page_data.get("extract", "")
                    return clean_wikipedia_content(raw_text)
        except Exception:
            pass
    return None


def is_junk_bibliography_chunk(text: str) -> bool:
    """Check if chunk consists mostly of references or ISBN listings."""
    isbn_count = len(re.findall(r'ISBN\s*978', text, flags=re.IGNORECASE))
    doi_count = len(re.findall(r'doi\s*:', text, flags=re.IGNORECASE))
    return isbn_count >= 2 or doi_count >= 3


def run_pipeline(
    train_path: str = "train.csv",
    test_path: str = "test.csv",
    output_parquet: str = "wikipedia_rag_chunks.parquet",
    faiss_index_path: str = "wikipedia_faiss.index",
    faiss_meta_path: str = "wikipedia_faiss_meta.json",
    chunk_size: int = 1200,
    chunk_overlap: int = 250,
    embedding_model_name: str = "BAAI/bge-small-en-v1.5",
    max_questions: int | None = None,
):
    print("Starting Wikipedia scraping and FAISS indexing pipeline...")

    prompts = []
    for path in [train_path, test_path]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'prompt' in df.columns:
                prompts.extend(df['prompt'].dropna().tolist())
                print(f"Loaded {len(df)} questions from '{path}'")

    if max_questions is not None and max_questions > 0:
        prompts = prompts[:max_questions]
        print(f"Limiting to first {max_questions} questions.")

    print(f"Total unique questions to process: {len(prompts)}")

    scraped_articles = {}
    searched_titles = set()

    start_time = time.time()

    for idx, prompt in enumerate(prompts, 1):
        titles = search_wikipedia_topics(prompt, max_results=2)

        for title in titles:
            if title not in searched_titles:
                searched_titles.add(title)
                article_text = fetch_wikipedia_article(title)
                if article_text and len(article_text) >= 200:
                    scraped_articles[title] = article_text
                    print(f"[{idx}/{len(prompts)}] Scraped: '{title}' ({len(article_text)} chars)")
                time.sleep(0.1)

    print(f"Scraping complete in {time.time() - start_time:.2f}s. Scraped {len(scraped_articles)} articles.")

    if not scraped_articles:
        print("No articles scraped. Exiting.")
        return

    print(f"Chunking articles (size={chunk_size}, overlap={chunk_overlap})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunk_records = []
    chunk_counter = 0

    for title, text in scraped_articles.items():
        sub_chunks = splitter.split_text(text)
        for chunk_idx, chunk_text in enumerate(sub_chunks):
            if is_junk_bibliography_chunk(chunk_text):
                continue

            formatted_text = f"Title: {title}\n{chunk_text}"
            chunk_records.append({
                "chunk_id": f"chunk_{chunk_counter}",
                "article_title": title,
                "chunk_index": chunk_idx,
                "text": formatted_text,
                "raw_text": chunk_text,
                "char_len": len(chunk_text),
            })
            chunk_counter += 1

    chunks_df = pd.DataFrame(chunk_records)
    print(f"Created {len(chunks_df)} chunks.")

    chunks_df.to_parquet(output_parquet, index=False)
    print(f"Saved Parquet dataset to: '{output_parquet}' ({os.path.getsize(output_parquet) / 1e6:.2f} MB)")

    print(f"Loading embedding model '{embedding_model_name}'...")
    embedder = SentenceTransformer(embedding_model_name)

    texts_to_embed = chunks_df["text"].tolist()
    print(f"Generating embeddings for {len(texts_to_embed)} chunks...")

    embeddings = embedder.encode(
        texts_to_embed,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    faiss.write_index(faiss_index, faiss_index_path)
    print(f"FAISS index saved to: '{faiss_index_path}' ({os.path.getsize(faiss_index_path) / 1e6:.2f} MB)")

    metadata = chunks_df[["chunk_id", "article_title", "chunk_index", "text"]].to_dict(orient="records")
    with open(faiss_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved to: '{faiss_meta_path}'")
    print("Pipeline finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Wikipedia and build FAISS RAG Index.")
    parser.add_argument("--train", type=str, default="train.csv", help="Path to train.csv")
    parser.add_argument("--test", type=str, default="test.csv", help="Path to test.csv")
    parser.add_argument("--parquet", type=str, default="wikipedia_rag_chunks.parquet", help="Output Parquet path")
    parser.add_argument("--index", type=str, default="wikipedia_faiss.index", help="Output FAISS index path")
    parser.add_argument("--meta", type=str, default="wikipedia_faiss_meta.json", help="Output metadata JSON path")
    parser.add_argument("--chunk_size", type=int, default=1200, help="Chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=250, help="Chunk overlap")
    parser.add_argument("--max_questions", type=int, default=None, help="Limit number of questions to process")

    args = parser.parse_args()

    run_pipeline(
        train_path=args.train,
        test_path=args.test,
        output_parquet=args.parquet,
        faiss_index_path=args.index,
        faiss_meta_path=args.meta,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_questions=args.max_questions,
    )
