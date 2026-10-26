import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

import search as retrieval


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npy"
METADATA_FILE = OUTPUT_DIR / "embedding_metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MIN_WORDS = 20

# Retrieval scoring settings
MIN_SEMANTIC_SCORE = 0.20
SEMANTIC_WEIGHT = 0.75
KEYWORD_WEIGHT = 0.25

TEST_QUESTIONS = [
    {
        "question": "What is email filtering in NLP?",
        "expected": "answerable",
    },
    {
        "question": "What is sentiment analysis, and how is it used?",
        "expected": "answerable",
    },
    {
        "question": "What are the limitations of the NLP model discussed in my notes?",
        "expected": "partially answerable",
    },
    {
        "question": "What is quantum computing?",
        "expected": "out of scope",
    },
]


def score_question(question, model, embeddings, metadata):
    query_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    semantic_scores = embeddings @ query_vector

    query_terms = (
        set(retrieval.tokenize(question))
        - retrieval.STOP_WORDS
    )

    if "nlp" in query_terms:
        query_terms.update(
            {"natural", "language", "processing"}
        )

    if {
        "natural",
        "language",
        "processing",
    }.issubset(query_terms):
        query_terms.add("nlp")

    idf = retrieval.build_idf(metadata)

    broad_question = retrieval.is_broad_application_question(
        question
    )

    results = []

    for i, chunk in enumerate(metadata):
        text = chunk["text"]

        if len(text.split()) < MIN_WORDS:
            continue

        if retrieval.is_metadata_chunk(text):
            continue

        semantic_score = float(semantic_scores[i])

        keyword_score = retrieval.weighted_keyword_score(
            query_terms,
            text,
            idf,
        )

        # A semantic threshold prevents keyword overlap alone
        # from giving an unrelated passage a competitive score.
        if semantic_score < MIN_SEMANTIC_SCORE:
            combined_score = 0.0
        else:
            combined_score = (
                SEMANTIC_WEIGHT * semantic_score
                + KEYWORD_WEIGHT * keyword_score
            )

            # Apply application boosts only to semantically
            # eligible passages.
            if broad_question:
                if retrieval.has_applications_overview(text):
                    combined_score += 0.12

                if retrieval.detect_application_heading(text):
                    combined_score += 0.10

                normalized = retrieval.normalize_text(text)

                application_count = sum(
                    1
                    for heading in retrieval.APPLICATION_HEADINGS
                    if heading in normalized
                )

                combined_score += min(
                    application_count * 0.015,
                    0.06,
                )

        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "pages": chunk.get("source_pages", []),
                "semantic": semantic_score,
                "keyword": keyword_score,
                "combined": combined_score,
                "text": text,
            }
        )

    return sorted(
        results,
        key=lambda item: item["combined"],
        reverse=True,
    )


def main():
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Missing embeddings: {EMBEDDINGS_FILE}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing metadata: {METADATA_FILE}"
        )

    embeddings = np.load(EMBEDDINGS_FILE)

    metadata = json.loads(
        METADATA_FILE.read_text(encoding="utf-8")
    )

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Embedding and metadata counts do not match. "
            "Run python embed_text.py first."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "Expected a 2D embeddings array."
        )

    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            "Embedding row count does not match metadata."
        )

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    for test in TEST_QUESTIONS:
        question = test["question"]

        print("\n" + "=" * 75)
        print(f"Question: {question}")
        print(f"Expected category: {test['expected']}")
        print("-" * 75)

        results = score_question(
            question,
            model,
            embeddings,
            metadata,
        )

        if not results:
            print("No eligible passages found.")
            continue

        print(
            f"{'Chunk':<8} {'Pages':<12} "
            f"{'Semantic':<10} {'Keyword':<10} "
            f"{'Combined':<10}"
        )

        for result in results[:5]:
            pages = ", ".join(
                str(page) for page in result["pages"]
            )

            print(
                f"{result['chunk_id']:<8} "
                f"{pages:<12} "
                f"{result['semantic']:<10.3f} "
                f"{result['keyword']:<10.3f} "
                f"{result['combined']:<10.3f}"
            )

            preview = " ".join(
                result["text"].split()
            )

            print(f"  Preview: {preview[:180]}...")

        best_ranked = results[0]

        best_semantic = max(
            result["semantic"]
            for result in results
        )

        best_combined = max(
            result["combined"]
            for result in results
        )

        print("\nDiagnostic:")
        print(
            "Highest-ranked passage semantic similarity: "
            f"{best_ranked['semantic']:.3f}"
        )
        print(
            "Best semantic similarity across candidates: "
            f"{best_semantic:.3f}"
        )
        print(
            f"Best combined score: {best_combined:.3f}"
        )
        print(
            "Highest-ranked chunk: "
            f"{best_ranked['chunk_id']}"
        )
        print(
            "Highest-ranked source pages: "
            + ", ".join(
                str(page)
                for page in best_ranked["pages"]
            )
        )


if __name__ == "__main__":
    main()
