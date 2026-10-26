
import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_FILE = BASE_DIR / "output" / "embeddings.npy"
METADATA_FILE = BASE_DIR / "output" / "embedding_metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10
BROAD_TOP_K = 5
MIN_WORDS = 20

STOP_WORDS = {
    "what", "are", "is", "the", "a", "an", "of", "in",
    "to", "and", "for", "how", "does", "do", "with", "can",
    "on", "by", "from", "about", "explain", "describe",
    "main", "major", "list", "give", "tell", "me",
    "discussed", "notes", "please", "which", "their",
}

APPLICATION_HEADINGS = (
    "sentiment analysis",
    "machine translation",
    "text extraction",
    "text classification",
    "speech recognition",
    "chatbot",
    "email filter",
    "email filtering",
    "question answering",
    "named entity recognition",
    "information extraction",
    "voice assistant",
    "search engine",
    "spam filter",
    "grammar checker",
    "language translation",
)


def tokenize(text):
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"course\s*outcomes", "courseoutcomes", text)
    return text.strip()


def is_metadata_chunk(text):
    normalized = normalize_text(text)
    beginning = normalized[:700]

    markers = (
        "unit –i:",
        "unit-i:",
        "courseoutcomes:",
        "course outcomes:",
        "reference books:",
        "references:",
    )

    return any(marker in beginning for marker in markers)


def is_broad_application_question(question):
    q = normalize_text(question)

    return any(
        term in q
        for term in (
            "application",
            "applications",
            "use case",
            "use cases",
        )
    )


def has_applications_overview(text):
    beginning = " ".join(normalize_text(text).split()[:25])

    return any(
        heading in beginning
        for heading in (
            "applications of nlp",
            "applications of natural language processing",
            "use cases of nlp",
        )
    )


def detect_application_heading(text):
    """Return the application heading if it occurs near the chunk start."""
    beginning = " ".join(normalize_text(text).split()[:25])

    for heading in APPLICATION_HEADINGS:
        pattern = (
            r"(?:^|\s)(?:\d+\s*[.)]?\s*)?"
            + re.escape(heading)
            + r"(?:\b|$)"
        )

        if re.search(pattern, beginning):
            return heading

    return None


def build_idf(metadata):
    document_frequency = Counter()

    for chunk in metadata:
        document_frequency.update(set(tokenize(chunk["text"])))

    total = len(metadata)

    return {
        term: np.log(1 + total / (1 + frequency))
        for term, frequency in document_frequency.items()
    }


def weighted_keyword_score(query_terms, text, idf):
    if not query_terms:
        return 0.0

    chunk_terms = set(tokenize(text))

    total_weight = sum(idf.get(term, 1.0) for term in query_terms)
    matched_weight = sum(
        idf.get(term, 1.0)
        for term in query_terms
        if term in chunk_terms
    )

    return matched_weight / total_weight if total_weight else 0.0


def main():
    if not EMBEDDINGS_FILE.exists() or not METADATA_FILE.exists():
        raise FileNotFoundError(
            "Embedding files not found. Run python embed_text.py first."
        )

    embeddings = np.load(EMBEDDINGS_FILE)
    metadata = json.loads(
        METADATA_FILE.read_text(encoding="utf-8")
    )

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Embedding and metadata counts do not match. "
            "Run python embed_text.py again."
        )

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    question = input(
        "\nAsk a question about your NLP notes: "
    ).strip()

    if not question:
        print("Please enter a question.")
        return

    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    semantic_scores = embeddings @ query_vector

    query_terms = set(tokenize(question)) - STOP_WORDS

    if "nlp" in query_terms:
        query_terms.update({"natural", "language", "processing"})

    if {"natural", "language", "processing"}.issubset(query_terms):
        query_terms.add("nlp")

    broad_question = is_broad_application_question(question)
    idf = build_idf(metadata)

    valid_indices = [
        i
        for i, chunk in enumerate(metadata)
        if len(chunk["text"].split()) >= MIN_WORDS
        and not is_metadata_chunk(chunk["text"])
    ]

    if not valid_indices:
        print("No suitable text chunks found.")
        return

    scores = {}

    for i in valid_indices:
        text = metadata[i]["text"]

        keyword_score = weighted_keyword_score(
            query_terms, text, idf
        )

        semantic_score = float(semantic_scores[i])

            # Prevent keyword overlap from rescuing semantically unrelated chunks.
        if semantic_score < 0.20:
            score = 0.0
        else:
            score = (
                0.75 * semantic_score
                + 0.25 * keyword_score
            )

        if broad_question:
            if has_applications_overview(text):
                score += 0.12

            if detect_application_heading(text):
                score += 0.10

            normalized = normalize_text(text)
            application_count = sum(
                1 for heading in APPLICATION_HEADINGS
                if heading in normalized
            )
            score += min(application_count * 0.015, 0.06)

        scores[i] = score

    ranked = sorted(
        valid_indices,
        key=lambda i: scores[i],
        reverse=True,
    )

    # For broad questions, select distinct application sections.
    if broad_question:
        selected = []
        selected_set = set()
        seen_headings = set()

        # The overview is useful because it summarizes several applications.
        for i in ranked:
            if has_applications_overview(metadata[i]["text"]):
                selected.append(i)
                selected_set.add(i)
                break

        # Prefer different application headings over repeated content.
        for i in ranked:
            if i in selected_set:
                continue

            heading = detect_application_heading(metadata[i]["text"])

            if heading and heading not in seen_headings:
                selected.append(i)
                selected_set.add(i)
                seen_headings.add(heading)

            if len(selected) >= BROAD_TOP_K:
                break

        # Fill any remaining slots with the best unselected passages.
        for i in ranked:
            if len(selected) >= BROAD_TOP_K:
                break

            if i not in selected_set:
                selected.append(i)
                selected_set.add(i)

        results = selected[:BROAD_TOP_K]
    else:
        results = ranked[:TOP_K]

    print("\nMost relevant passages")
    print("=" * 60)

    for rank, index in enumerate(results, start=1):
        chunk = metadata[index]

        print(f"\nRank: {rank}")
        print(f"Combined score: {scores[index]:.4f}")
        print(f"Semantic similarity: {semantic_scores[index]:.4f}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(
            "Source pages: "
            + ", ".join(map(str, chunk["source_pages"]))
        )
        print("-" * 60)
        print(chunk["text"][:1200])
        print()


if __name__ == "__main__":
    main()
