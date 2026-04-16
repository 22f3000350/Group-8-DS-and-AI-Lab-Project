"""
ingest.py
---------
Run this ONCE locally to build the ChromaDB vector index
from your agricultural PDF/text documents.

Usage:
    python rag/ingest.py

Place your documents in rag/docs/ following this naming convention:
    {crop}_{disease}_{source}.pdf
    e.g. corn_gray_leaf_spot_icar.pdf
         rice_brown_spot_tnau.pdf
         wheat_yellow_rust_fao.pdf
         potato_late_blight_icar.pdf
         sugarcane_red_rot_icar.pdf

The script auto-tags each chunk with crop + disease metadata
parsed from the filename — this is what makes retrieval precise.
"""

import os
import re
import sys
import chromadb
from chromadb.utils import embedding_functions

# ── Paths ────────────────────────────────────────────────────────────────────
DOCS_DIR    = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_DIR  = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION  = "crop_disease_knowledge"

# ── Embedding model ───────────────────────────────────────────────────────────
# multilingual-e5-large handles Hindi, Tamil, Telugu, Bengali, Malayalam, Kannada
# If slow on your machine, swap to 'paraphrase-multilingual-MiniLM-L12-v2' (faster, smaller)
EMBED_MODEL = "intfloat/multilingual-e5-large"

# ── Crop/disease vocabulary for filename parsing ──────────────────────────────
KNOWN_CROPS = ["corn", "potato", "rice", "wheat", "sugarcane"]

DISEASE_KEYWORDS = [
    "gray_leaf_spot", "common_rust", "northern_leaf_blight",
    "early_blight", "late_blight",
    "brown_spot", "leaf_blast", "neck_blast",
    "brown_rust", "yellow_rust",
    "red_rot", "bacterial_blight",
    "healthy",
]

# Section headings used for smart chunking
SECTION_HEADINGS = re.compile(
    r"(symptoms?|management|treatment|prevention|control|cause|introduction|"
    r"identification|distribution|disease\s+cycle|chemical\s+control|"
    r"biological\s+control|cultural\s+practices?|economic\s+importance)",
    re.IGNORECASE,
)

CHUNK_SIZE    = 400   # tokens (approx characters / 4)
CHUNK_OVERLAP = 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_metadata_from_filename(filename: str) -> dict:
    """
    Extract crop, disease, and source from filename.
    e.g. 'rice_brown_spot_icar.pdf' → {crop:'rice', disease:'brown spot', source:'icar'}
    """
    name = os.path.splitext(filename)[0].lower()
    parts = name.split("_")

    crop = "unknown"
    for c in KNOWN_CROPS:
        if name.startswith(c):
            crop = c
            break

    disease = "general"
    for d in DISEASE_KEYWORDS:
        if d in name:
            disease = d.replace("_", " ")
            break

    # Source = everything after crop_disease
    source_parts = [p for p in parts if p not in crop.split("_") and p not in disease.replace(" ", "_").split("_")]
    source = "_".join(source_parts) if source_parts else "unknown"

    return {"crop": crop, "disease": disease, "source": source, "filename": filename}


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from a PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")


def extract_text_from_txt(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def smart_chunk(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into chunks, preferring section boundaries.
    Falls back to fixed-size chunks if no headings found.
    """
    # Try splitting on section headings first
    sections = SECTION_HEADINGS.split(text)

    chunks = []
    current = ""

    for part in sections:
        # If adding this part keeps us under limit, append
        if len(current) + len(part) < chunk_size * 4:  # *4: chars to approx tokens
            current += " " + part
        else:
            # Save current chunk if non-trivial
            if len(current.strip()) > 100:
                chunks.append(current.strip())
            current = part

    if len(current.strip()) > 100:
        chunks.append(current.strip())

    # If no section splits found (plain text), fall back to fixed windows
    if len(chunks) <= 1:
        words = text.split()
        step  = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + chunk_size])
            if len(chunk.strip()) > 100:
                chunks.append(chunk.strip())

    return chunks


def ingest_document(collection, filepath: str):
    """Process one document and upsert its chunks into ChromaDB."""
    filename = os.path.basename(filepath)
    ext      = os.path.splitext(filename)[1].lower()
    meta     = parse_metadata_from_filename(filename)

    print(f"\n[ingest] Processing: {filename}")
    print(f"         Crop={meta['crop']}, Disease={meta['disease']}, Source={meta['source']}")

    # Extract text
    if ext == ".pdf":
        text = extract_text_from_pdf(filepath)
    elif ext in (".txt", ".md"):
        text = extract_text_from_txt(filepath)
    else:
        print(f"         Skipping unsupported format: {ext}")
        return

    if len(text.strip()) < 100:
        print(f"         Warning: Very little text extracted from {filename}. Check if PDF is scanned/image-based.")
        return

    # Chunk
    chunks = smart_chunk(text)
    print(f"         → {len(chunks)} chunks created")

    # Upsert into ChromaDB
    ids       = [f"{filename}__chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "crop":     meta["crop"],
            "disease":  meta["disease"],
            "source":   meta["source"],
            "filename": filename,
            "chunk_id": i,
        }
        for i in range(len(chunks))
    ]

    # Add in batches of 50 to avoid memory spikes
    batch_size = 50
    for start in range(0, len(chunks), batch_size):
        collection.upsert(
            documents=ids[start : start + batch_size],
            ids=ids[start : start + batch_size],
            metadatas=metadatas[start : start + batch_size],
        )
        # ChromaDB needs the actual text in documents, not ids — fix:
        collection.upsert(
            documents=chunks[start : start + batch_size],
            ids=ids[start : start + batch_size],
            metadatas=metadatas[start : start + batch_size],
        )

    print(f"         ✓ Upserted {len(chunks)} chunks")


def main():
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"[ingest] Created docs directory: {DOCS_DIR}")
        print("[ingest] Add your agricultural PDFs/TXTs there and re-run.")
        sys.exit(0)

    doc_files = [
        f for f in os.listdir(DOCS_DIR)
        if os.path.splitext(f)[1].lower() in (".pdf", ".txt", ".md")
    ]

    if not doc_files:
        print(f"[ingest] No documents found in {DOCS_DIR}")
        print("[ingest] Add PDFs or TXTs named like: rice_brown_spot_icar.pdf")
        sys.exit(0)

    print(f"[ingest] Found {len(doc_files)} documents in {DOCS_DIR}")
    print(f"[ingest] Using embedding model: {EMBED_MODEL}")
    print(f"[ingest] ChromaDB will be saved to: {CHROMA_DIR}")

    # Set up ChromaDB with sentence-transformers embedding
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    for filename in sorted(doc_files):
        ingest_document(collection, os.path.join(DOCS_DIR, filename))

    total = collection.count()
    print(f"\n[ingest] ✓ Done. Total chunks in ChromaDB: {total}")
    print(f"[ingest] Index saved at: {CHROMA_DIR}")
    print("[ingest] Commit the chroma_db/ folder to your repo (or regenerate on Space startup).")


if __name__ == "__main__":
    main()
