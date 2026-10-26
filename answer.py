
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

import search as retrieval


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npy"
METADATA_FILE = OUTPUT_DIR / "embedding_metadata.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "gemini-3.1-flash-lite"

TOP_K = 5
MIN_WORDS = 20

MIN_TOP_SCORE = 0.25
MIN_SEMANTIC_SCORE = 0.25
RELEVANCE_MARGIN = 0.14

# Timeout is specified in seconds.
GEMINI_TIMEOUT_SECONDS = 120

LIMITATION_EVIDENCE_PATTERNS = (
    r"\blimitations?\b",
    r"\bdrawbacks?\b",
    r"\bdisadvantages?\b",
    r"\bweakness(?:es)?\b",
    r"\bshortcomings?\b",
    r"\bfails? to\b",
    r"\bfailure to\b",
    r"\blow accuracy\b",
    r"\bpoor accuracy\b",
    r"\binaccurate predictions?\b",
    r"\bmodel is biased\b",
    r"\bmodel bias\b",
    r"\btraining data bias\b",
)





# --------------------------------------------------
# Question classification
# --------------------------------------------------

def is_limitation_question(question):
    """Identify questions asking about limitations."""

    normalized = retrieval.normalize_text(question)

    patterns = (
        r"\blimitations?\b",
        r"\bdrawbacks?\b",
        r"\bdisadvantages?\b",
        r"\bweakness(?:es)?\b",
        r"\bshortcomings?\b",
        r"\bwhat can(?:not|'t) .{0,40}\bdo\b",
    )

    return any(
        re.search(pattern, normalized)
        for pattern in patterns
    )


def has_explicit_limitation_evidence(text):
    """Check whether a passage contains explicit limitation-related evidence."""

    normalized = retrieval.normalize_text(text)

    return any(
        re.search(pattern, normalized)
        for pattern in LIMITATION_EVIDENCE_PATTERNS
    )

# --------------------------------------------------
# Retrieval
# --------------------------------------------------

def retrieve_passages(question, model, embeddings, metadata):
    """Retrieve relevant passages using semantic and keyword scores."""

    start_time = time.perf_counter()

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
        query_terms.update({"natural", "language", "processing"})

    if {"natural", "language", "processing"}.issubset(query_terms):
        query_terms.add("nlp")

    broad_question = retrieval.is_broad_application_question(
        question
    )

    limitation_question = is_limitation_question(question)

    idf = retrieval.build_idf(metadata)

    valid_indices = [
        i
        for i, chunk in enumerate(metadata)
        if len(chunk["text"].split()) >= MIN_WORDS
        and not retrieval.is_metadata_chunk(chunk["text"])
    ]

    # Only retrieve explicit limitation-related passages for
    # limitation questions. Do not invent model limitations.
    if limitation_question:
        valid_indices = [
            i
            for i in valid_indices
            if has_explicit_limitation_evidence(
                metadata[i]["text"]
            )
        ]

        if not valid_indices:
            print(
                "\nNo passages explicitly mentioning limitations "
                "or related weaknesses were found in the notes."
            )
            print(
                f"Retrieval time: "
                f"{time.perf_counter() - start_time:.2f} seconds"
            )
            return []

    if not valid_indices:
        print("No eligible passages were found.")
        return []

    scores = {}

    for i in valid_indices:
        text = metadata[i]["text"]

        keyword_score = retrieval.weighted_keyword_score(
            query_terms,
            text,
            idf,
        )

        score = (
            0.65 * float(semantic_scores[i])
            + 0.35 * keyword_score
        )

        if broad_question:
            if retrieval.has_applications_overview(text):
                score += 0.12

            if retrieval.detect_application_heading(text):
                score += 0.10

            normalized = retrieval.normalize_text(text)

            application_count = sum(
                1
                for heading in retrieval.APPLICATION_HEADINGS
                if heading in normalized
            )

            score += min(application_count * 0.015, 0.06)

        scores[i] = score

    ranked = sorted(
        valid_indices,
        key=lambda i: scores[i],
        reverse=True,
    )

    best_semantic_score = max(
        float(semantic_scores[i])
        for i in valid_indices
    )

    best_combined_score = scores[ranked[0]]

    print(
        f"\nBest semantic similarity: "
        f"{best_semantic_score:.3f}"
    )
    print(
        f"Best combined retrieval score: "
        f"{best_combined_score:.3f}"
    )

    if best_semantic_score < MIN_SEMANTIC_SCORE:
        print("No sufficiently relevant passages found.")
        print(
            f"Retrieval time: "
            f"{time.perf_counter() - start_time:.2f} seconds"
        )
        return []

    if best_combined_score < MIN_TOP_SCORE:
        print("The combined retrieval score is too low.")
        print(
            f"Retrieval time: "
            f"{time.perf_counter() - start_time:.2f} seconds"
        )
        return []

    score_floor = max(
        MIN_TOP_SCORE,
        best_combined_score - RELEVANCE_MARGIN,
    )

    relevant_ranked = [
        i
        for i in ranked
        if scores[i] >= score_floor
        and float(semantic_scores[i]) >= MIN_SEMANTIC_SCORE
    ]

    passages = []

    for i in relevant_ranked[:TOP_K]:
        passage = dict(metadata[i])
        passage["_semantic_score"] = float(semantic_scores[i])
        passage["_retrieval_score"] = float(scores[i])
        passages.append(passage)

    print(
        f"Retrieval time: "
        f"{time.perf_counter() - start_time:.2f} seconds"
    )

    return passages


# --------------------------------------------------
# Context construction
# --------------------------------------------------

def build_context(passages):
    """Build context with explicit source-page labels."""

    parts = []

    for passage in passages:
        source_pages = passage.get("source_pages", [])

        if source_pages:
            page_labels = " ".join(
                f"[Page {page}]"
                for page in source_pages
            )
        else:
            page_labels = "[Page unknown]"

        parts.append(
            f"{page_labels}\n{passage['text']}"
        )

    return "\n\n".join(parts)


# --------------------------------------------------
# Citation validation
# --------------------------------------------------

def validate_citations(answer, passages):
    """Check citation format and retrieved-page membership."""

    allowed_pages = {
        str(page)
        for passage in passages
        for page in passage.get("source_pages", [])
    }

    # Matches [Page 8], [Page 12], etc.
    cited_pages = set(
        re.findall(r"\[Page\s+(\d+)\]", answer)
    )

    warnings = []

    if not cited_pages:
        warnings.append("No [Page N] citations were found.")

    unsupported_pages = cited_pages - allowed_pages

    if unsupported_pages:
        warnings.append(
            "Cited pages not present in retrieved sources: "
            + ", ".join(sorted(unsupported_pages))
        )

    return warnings


# --------------------------------------------------
# Answer generation
# --------------------------------------------------

def generate_answer(client, question, passages):
    """Generate a concise answer grounded in the lecture notes."""

    context = build_context(passages)

    prompt = f"""
Answer the question using only the provided NLP lecture notes.

Instructions:
- Be concise and student-friendly, using about 3-5 sentences.
- Cite factual claims using [Page N].
- Use only page numbers supplied in the context.
- If the context does not answer the question, say so clearly.
- Do not invent facts, examples, or citations.
- Do not confuse general NLP challenges with limitations of a
  specific model or algorithm.

CONTEXT:
{context}

QUESTION:
{question}
"""

    start_time = time.perf_counter()

    print("Sending request to Gemini...")

    try:
        interaction = client.interactions.create(
            model=GENERATION_MODEL,
            input=prompt,
            store=False,
            timeout=GEMINI_TIMEOUT_SECONDS,
        )

    except Exception as error:
        elapsed = time.perf_counter() - start_time

        print(
            f"Gemini request failed after {elapsed:.2f} seconds."
        )
        print(f"Error type: {type(error).__name__}")
        print(f"Error details: {error}")

        raise

    print(
        f"Gemini response time: "
        f"{time.perf_counter() - start_time:.2f} seconds"
    )

    answer = interaction.output_text

    if not answer or not answer.strip():
        raise RuntimeError("The API returned an empty answer.")

    warnings = validate_citations(answer, passages)

    if warnings:
        print("\nCitation Warnings")
        for warning in warnings:
            print(f"- {warning}")

    return answer


# --------------------------------------------------
# Main application
# --------------------------------------------------

def main():
    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Check the .env file "
            "in E:\\DocuMind."
        )

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_FILE}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
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

    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model_start = time.perf_counter()

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print(
        f"Embedding model loaded in "
        f"{time.perf_counter() - model_start:.2f} seconds"
    )

    client = genai.Client(api_key=api_key)

    print(f"Generation model: {GENERATION_MODEL}")
    print("\nDocuMind is ready. Type 'exit' to quit.")

    while True:
        question = input(
            "\nAsk a question about your NLP notes: "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            print("DocuMind session ended.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            passages = retrieve_passages(
                question,
                embedding_model,
                embeddings,
                metadata,
            )

            if not passages:
                print(
                    "\nDocuMind could not find sufficiently relevant "
                    "passages to answer confidently from your notes."
                )
                continue

            print("\nRetrieved Sources")
            print("-" * 60)

            for passage in passages:
                pages = ", ".join(
                    str(page)
                    for page in passage.get("source_pages", [])
                )

                print(
                    f"Chunk {passage['chunk_id']} | "
                    f"Page(s): {pages} | "
                    f"Semantic: "
                    f"{passage['_semantic_score']:.3f} | "
                    f"Combined: "
                    f"{passage['_retrieval_score']:.3f}"
                )

            print("\nGenerating answer...")

            answer = generate_answer(
                client,
                question,
                passages,
            )

            print("\nDocuMind Answer")
            print("=" * 60)
            print(answer)

        except Exception as error:
            print(f"\nRequest failed: {error}")
            print(
                "Check the API key, internet connection, "
                "model availability, SDK compatibility, "
                "and API usage limits."
            )


if __name__ == "__main__":
    main()
