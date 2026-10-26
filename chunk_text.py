
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "output" / "extracted_text.txt"
OUTPUT_FILE = BASE_DIR / "output" / "chunks.json"

CHUNK_SIZE = 250
CHUNK_OVERLAP = 40

PAGE_PATTERN = re.compile(
    r"(?m)^\s*-{2,}\s*PAGE\s+(\d+)\s*-{2,}\s*$",
    re.IGNORECASE,
)

# Recognize common numbered headings and lecture-note headings.
HEADING_PATTERN = re.compile(
    r"(?m)^[ \t]*(?:"
    r"\d+(?:\.\d+)*[.)]?[ \t]+[A-Z][^\n]{0,120}"
    r"|UNIT[ \t\-–IVX0-9]+[^\n]{0,100}"
    r"|(?:Applications?|Introduction|Conclusion|Definition|"
    r"Advantages|Disadvantages|Examples?|Speech Recognition|"
    r"Chatbots?|Email Filter|Search Autocorrect and Autocomplete)"
    r"[ \t]*:?[ \t]*"
    r")[ \t]*$",
    re.IGNORECASE,
)


def load_pages(file_path):
    """Read extracted text and preserve page numbers."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}"
        )

    text = file_path.read_text(encoding="utf-8")
    parts = PAGE_PATTERN.split(text)

    pages = []

    for i in range(1, len(parts), 2):
        page_number = int(parts[i])
        content = parts[i + 1].strip()

        if content and not re.fullmatch(r"\d+", content):
            pages.append((page_number, content))

    return pages


def split_into_sections(pages):
    """Split at detected headings while retaining each heading."""
    sections = []

    for page_number, content in pages:
        matches = list(HEADING_PATTERN.finditer(content))

        if not matches:
            if content.strip():
                sections.append({
                    "page": page_number,
                    "text": content.strip(),
                })
            continue

        # Preserve content before the first detected heading.
        prefix = content[:matches[0].start()].strip()

        if prefix:
            sections.append({
                "page": page_number,
                "text": prefix,
            })

        for i, match in enumerate(matches):
            end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(content)
            )

            section_text = content[match.start():end].strip()

            if section_text:
                sections.append({
                    "page": page_number,
                    "text": section_text,
                })

    return sections


def merge_short_sections(sections, minimum_words=20):
    """Attach short heading-only sections to the following section."""
    merged = []

    for section in sections:
        current = {
            "page": section["page"],
            "text": section["text"].strip(),
        }

        if not current["text"]:
            continue

        if merged and len(merged[-1]["text"].split()) < minimum_words:
            # The previous section is probably a heading or short fragment.
            merged[-1]["text"] += "\n" + current["text"]
        else:
            merged.append(current)

    return merged


def create_chunks(sections):
    """Create overlapping chunks while preserving source page numbers."""
    chunks = []
    seen = set()

    for section in sections:
        words = section["text"].split()

        if not words:
            continue

        start = 0

        while start < len(words):
            part_words = words[start:start + CHUNK_SIZE]
            chunk_text = " ".join(part_words)

            if chunk_text not in seen:
                seen.add(chunk_text)

                chunks.append({
                    "chunk_id": len(chunks),
                    "page_start": section["page"],
                    "page_end": section["page"],
                    "source_pages": [section["page"]],
                    "text": chunk_text,
                    "word_count": len(part_words),
                })

            if start + CHUNK_SIZE >= len(words):
                break

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def main():
    pages = load_pages(INPUT_FILE)
    sections = split_into_sections(pages)
    sections = merge_short_sections(sections)
    chunks = create_chunks(sections)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_FILE.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Text chunking completed!")
    print(f"Pages with meaningful text: {len(pages)}")
    print(f"Sections after merging: {len(sections)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Target chunk size: {CHUNK_SIZE} words")
    print(f"Overlap: {CHUNK_OVERLAP} words")
    print(f"Saved to: {OUTPUT_FILE}")

    short_chunks = sum(
        1 for chunk in chunks if chunk["word_count"] < 20
    )
    print(f"Chunks shorter than 20 words: {short_chunks}")

    if chunks:
        print("\nFirst chunk preview:")
        print(chunks[0]["text"][:500])


if __name__ == "__main__":
    main()


# Temporary inspection of NLP application chunks
import json
from pathlib import Path

file_path = Path("output/chunks.json")
chunks = json.loads(file_path.read_text(encoding="utf-8"))

keywords = [
    "applications of nlp",
    "sentiment analysis",
    "machine translation",
    "speech recognition",
    "chatbots",
]

for chunk in chunks:
    text = chunk["text"]

    if any(word in text.lower() for word in keywords):
        print(f"\nChunk ID: {chunk['chunk_id']}")
        print(f"Pages: {chunk['source_pages']}")
        print(f"Words: {chunk['word_count']}")
        print(text[:700])
        print("-" * 50)