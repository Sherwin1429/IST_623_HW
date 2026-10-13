# --------------------------------------------------
# HW7 - Build News Vector Database
# --------------------------------------------------

import sys

# ChromaDB SQLite compatibility fix
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
import pandas as pd
import chromadb
import streamlit as st

from openai import OpenAI


# --------------------------------------------------
# OpenAI Client
# --------------------------------------------------

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found. "
        "Make sure it exists in .streamlit/secrets.toml "
        "or as an environment variable."
    )

client = OpenAI(api_key=api_key)


# --------------------------------------------------
# Read News CSV
# --------------------------------------------------

news_df = pd.read_csv("news.csv")

print(f"Loaded {len(news_df)} rows from news.csv.")


# --------------------------------------------------
# Remove Duplicate Articles
# --------------------------------------------------

# Remove rows with missing URLs first
news_df = news_df.dropna(
    subset=["URL"]
)

# Remove duplicate URLs
news_df = news_df.drop_duplicates(
    subset=["URL"],
    keep="first"
).reset_index(drop=True)

print(
    f"After removing duplicate URLs: "
    f"{len(news_df)} unique articles."
)


# --------------------------------------------------
# Chunking Function
# --------------------------------------------------

def chunk_text(
    text,
    chunk_size=1200,
    overlap=150
):

    words = str(text).split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# Create Embeddings
# --------------------------------------------------

def create_embeddings(texts):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    return [
        item.embedding
        for item in response.data
    ]


# --------------------------------------------------
# Create Persistent ChromaDB
# --------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path="HW7_ChromaDB"
)

collection = chroma_client.get_or_create_collection(
    name="HW7NewsCollection"
)


# --------------------------------------------------
# Stop If DB Already Exists
# --------------------------------------------------

if collection.count() > 0:

    print(
        f"HW7_ChromaDB already exists "
        f"with {collection.count()} chunks."
    )

    print(
        "Delete HW7_ChromaDB first "
        "if you want to rebuild it."
    )

    sys.exit()


# --------------------------------------------------
# Prepare Article Chunks
# --------------------------------------------------

documents = []
metadatas = []
ids = []

for article_index, row in news_df.iterrows():

    article_text = str(
        row["Document"]
    ).strip()

    company_name = str(
        row["company_name"]
    )

    article_date = str(
        row["Date"]
    )

    article_url = str(
        row["URL"]
    )

    # Skip empty article text
    if not article_text:
        continue

    article_chunks = chunk_text(
        article_text
    )

    for chunk_number, chunk in enumerate(
        article_chunks
    ):

        documents.append(
            chunk
        )

        metadatas.append(
            {
                "article_id": int(
                    article_index
                ),

                "company_name": company_name,

                "date": article_date,

                "url": article_url,

                "chunk_number": int(
                    chunk_number
                )
            }
        )

        ids.append(
            f"article_{article_index}"
            f"_chunk_{chunk_number}"
        )


print(
    f"Created {len(documents)} chunks "
    f"from {len(news_df)} unique articles."
)


# --------------------------------------------------
# Embed + Store in Batches
# --------------------------------------------------

batch_size = 50

for start in range(
    0,
    len(documents),
    batch_size
):

    end = min(
        start + batch_size,
        len(documents)
    )

    batch_documents = documents[
        start:end
    ]

    batch_metadatas = metadatas[
        start:end
    ]

    batch_ids = ids[
        start:end
    ]

    print(
        f"Embedding chunks "
        f"{start + 1} to {end}..."
    )

    batch_embeddings = create_embeddings(
        batch_documents
    )

    collection.add(
        ids=batch_ids,
        documents=batch_documents,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas
    )


# --------------------------------------------------
# Finished
# --------------------------------------------------

print()
print(
    "-----------------------------------"
)
print(
    "HW7 vector database created!"
)
print(
    "-----------------------------------"
)

print(
    f"Unique articles: "
    f"{len(news_df)}"
)

print(
    f"Total chunks: "
    f"{collection.count()}"
)

print(
    "Database folder: "
    "HW7_ChromaDB"
)