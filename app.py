import pandas as pd
import numpy as np
import faiss
import pickle

from sentence_transformers import SentenceTransformer
import gradio as gr


# Load books dataset
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Load prebuilt FAISS index and data
with open("book_vector_db/texts.pkl", "rb") as f:
    texts = pickle.load(f)
with open("book_vector_db/isbns.pkl", "rb") as f:
    raw_isbns = pickle.load(f)
    isbns = [int(str(i).strip().replace('"', '').replace("'", "")) for i in raw_isbns]

index = faiss.read_index("book_vector_db/book_index.faiss")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Semantic search function using FAISS
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    query_vector = embedding_model.encode([query]).astype("float32")
    D, I = index.search(query_vector, initial_top_k)
    matched_isbns = [int(isbns[idx]) for idx in I[0]]
    book_recs = books[books["isbn13"].isin(matched_isbns)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

# Display recommendations with thumbnails and description
def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

# Categories and tones
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Gradio dashboard
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Describe a book or theme you like:",
                                placeholder="e.g., A story about friendship and forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Choose category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Choose emotional tone:", value="All")
        submit_button = gr.Button("🔍 Recommend")

    gr.Markdown("## Recommended Books")
    output = gr.Gallery(label="Results", columns=4, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

if __name__ == "__main__":
    dashboard.launch(share=True, debug=True)
