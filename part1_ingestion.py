from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Step 1: Read document from file
def load_document(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# Step 2: Sliding window chunking (2-page approximation using character limits)
def create_chunks(text, window_size=400, overlap=100):
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = min(start + window_size, len(text))
        chunk_text = text[start:end]

        chunks.append({
            "chunk_id": chunk_id,
            "raw_text": chunk_text
        })

        if end == len(text):
            break

        start = end - overlap
        chunk_id += 1

    return chunks


# Step 3: Placeholder summary (first 30 words)
def summarize_chunk(text):
    words = text.split()
    return " ".join(words[:30])


# Step 4: Rule-based category labeling
def assign_category(text):
    text_lower = text.lower()

    if any(word in text_lower for word in ["rag", "retrieval", "embedding", "language model"]):
        return "AI"
    elif any(word in text_lower for word in ["math", "equation", "calculate", "number"]):
        return "Math"
    elif any(word in text_lower for word in ["law", "legal", "court", "contract"]):
        return "Legal"
    else:
        return "General"


# Step 5: Distilled knowledge = simple keywords
def extract_keywords(text, top_n=5):
    stop_words = {
        "the", "is", "a", "an", "and", "or", "to", "of", "in", "on", "for",
        "by", "with", "this", "that", "it", "as", "be", "at", "from"
    }

    words = text.lower().replace(".", "").replace(",", "").split()
    filtered = [w for w in words if w not in stop_words and len(w) > 3]

    freq = {}
    for word in filtered:
        freq[word] = freq.get(word, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [word for word, count in sorted_words[:top_n]]
    return keywords


# Step 6: Build Knowledge Pyramid
def build_knowledge_pyramid(chunks):
    pyramid = []

    for chunk in chunks:
        raw_text = chunk["raw_text"]
        summary = summarize_chunk(raw_text)
        category = assign_category(raw_text)
        keywords = extract_keywords(raw_text)

        pyramid.append({
            "chunk_id": chunk["chunk_id"],
            "raw_text": raw_text,
            "summary": summary,
            "category": category,
            "distilled_knowledge": keywords
        })

    return pyramid


# Step 7: Prepare searchable entries from all pyramid levels
def create_search_entries(pyramid):
    entries = []

    for item in pyramid:
        entries.append({
            "chunk_id": item["chunk_id"],
            "level": "raw_text",
            "content": item["raw_text"]
        })

        entries.append({
            "chunk_id": item["chunk_id"],
            "level": "summary",
            "content": item["summary"]
        })

        entries.append({
            "chunk_id": item["chunk_id"],
            "level": "category",
            "content": item["category"]
        })

        entries.append({
            "chunk_id": item["chunk_id"],
            "level": "distilled_knowledge",
            "content": " ".join(item["distilled_knowledge"])
        })

    return entries


# Step 8: Retrieve best match using TF-IDF + cosine similarity
def retrieve_best_match(query, entries):
    documents = [entry["content"] for entry in entries]

    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    best_index = similarities.argmax()

    best_entry = entries[best_index]
    best_score = similarities[best_index]

    return best_entry, best_score


# Step 9: Main function
def main():
    file_path = "sample_document.txt"

    # Load document
    text = load_document(file_path)

    # Create chunks
    chunks = create_chunks(text)

    # Build pyramid
    pyramid = build_knowledge_pyramid(chunks)

    # Print pyramid for understanding
    print("\n=== KNOWLEDGE PYRAMID ===\n")
    for item in pyramid:
        print(f"Chunk ID: {item['chunk_id']}")
        print("RAW TEXT:", item["raw_text"][:100], "...")
        print("SUMMARY:", item["summary"])
        print("CATEGORY:", item["category"])
        print("DISTILLED KNOWLEDGE:", item["distilled_knowledge"])
        print("-" * 60)

    # Create searchable entries
    entries = create_search_entries(pyramid)

    # User query
    query = input("\nEnter your query: ")

    # Retrieve best match
    best_entry, score = retrieve_best_match(query, entries)

    print("\n=== BEST MATCH ===")
    print("Query:", query)
    print("Chunk ID:", best_entry["chunk_id"])
    print("Matched Level:", best_entry["level"])
    print("Similarity Score:", round(score, 4))
    print("Content:", best_entry["content"])


if __name__ == "__main__":
    main()