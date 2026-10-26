
from pathlib import Path
from pypdf import PdfReader
from importlib import import_module

try:
    PdfReader = import_module("pypdf").PdfReader
except ImportError:
    try:
        PdfReader = import_module("PyPDF2").PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF extraction requires pypdf. Install it with: pip install pypdf"
        ) from exc

# Define project paths
PROJECT_DIR = Path(__file__).resolve().parent
PDF_PATH = PROJECT_DIR / "data" / "sample.pdf"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "extracted_text.txt"


def extract_pdf_text(pdf_path: Path) -> list[dict]:
    """Extract text from a PDF, preserving page numbers."""

    reader = PdfReader(str(pdf_path))
    extracted_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        extracted_pages.append({
            "page": page_number,
            "text": text.strip()
        })

    return extracted_pages


def main():
    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        print("Place your PDF in the data folder as sample.pdf.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pages = extract_pdf_text(PDF_PATH)

    # Keep page boundaries so we can cite sources later.
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for item in pages:
            file.write(f"\n--- PAGE {item['page']} ---\n")
            file.write(item["text"])
            file.write("\n")

    total_characters = sum(
        len(item["text"]) for item in pages
    )

    print("\nPDF extraction completed!")
    print(f"Total pages: {len(pages)}")
    print(f"Characters extracted: {total_characters}")
    print(f"Saved text to: {OUTPUT_PATH}")

    print("\nPreview of extracted text:\n")
    for item in pages:
        if item["text"]:
            print(f"Page {item['page']}:")
            print(item["text"][:500])
            break
    else:
        print("No selectable text found in this PDF.")


if __name__ == "__main__":
    main()
