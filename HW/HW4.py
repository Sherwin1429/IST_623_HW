import sys

# Fix SQLite version for ChromaDB
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
from openai import OpenAI
import chromadb
from bs4 import BeautifulSoup
import os


# --------------------------------------------------
# Page Title
# --------------------------------------------------

st.title("HW 4 - iSchool Student Organization Chatbot")

st.write(
    "Ask me questions about Syracuse University student organizations."
)


# --------------------------------------------------
# OpenAI Client
# --------------------------------------------------

openai_api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI(
    api_key=openai_api_key
)


# --------------------------------------------------
# Function to Read HTML
# --------------------------------------------------

def read_html(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        html_content = file.read()

    # BeautifulSoup removes HTML tags and gives us
    # the readable text from the webpage.
    soup = BeautifulSoup(
        html_content,
        "html.parser"
    )

    # Remove script and style content because it does
    # not contain useful student organization information.
    for element in soup(["script", "style"]):
        element.decompose()

    text = soup.get_text(
        separator=" ",
        strip=True
    )

    return text


# --------------------------------------------------
# Function to Split Each HTML Document
# into Exactly Two Mini-Documents
# --------------------------------------------------

def split_into_two_chunks(text):

    # CHUNKING METHOD:
    # Each HTML document is divided into two approximately
    # equal halves. We split near the middle of the document
    # at a word boundary so that words are not cut in half.
    #
    # I chose this method because the assignment requires
    # exactly two mini-documents for every HTML document.
    # Two balanced chunks preserve information from both
    # halves of each organization page while giving the
    # vector database smaller pieces of text to search.

    words = text.split()

    middle = len(words) // 2

    first_chunk = " ".join(
        words[:middle]
    )

    second_chunk = " ".join(
        words[middle:]
    )

    return [
        first_chunk,
        second_chunk
    ]


# --------------------------------------------------
# Function to Create Embedding
# --------------------------------------------------

def create_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


# --------------------------------------------------
# Create / Load Persistent Vector Database
# --------------------------------------------------

# Use a persistent Chroma database so the 513 HTML files
# do not have to be embedded again every time the app runs.

chroma_client = chromadb.PersistentClient(
    path="HW4_ChromaDB"
)

collection = chroma_client.get_or_create_collection(
    name="HW4Collection"
)


# --------------------------------------------------
# Add HTML Documents Only If DB Is Empty
# --------------------------------------------------

if collection.count() == 0:

    with st.spinner(
        "Creating vector database from student organization pages..."
    ):

        html_folder = "HW4_Data"

        html_files = [
            file
            for file in os.listdir(html_folder)
            if file.lower().endswith(".html")
        ]

        document_id = 0

        for filename in html_files:

            file_path = os.path.join(
                html_folder,
                filename
            )

            document_text = read_html(
                file_path
            )

            # Create exactly two mini-documents
            chunks = split_into_two_chunks(
                document_text
            )

            for chunk_number, chunk in enumerate(chunks):

                # Skip a completely empty chunk
                if not chunk.strip():
                    continue

                embedding = create_embedding(
                    chunk
                )

                collection.add(
                    ids=[str(document_id)],
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[
                        {
                            "filename": filename,
                            "chunk": chunk_number + 1
                        }
                    ]
                )

                document_id += 1


# Store the collection in Streamlit session state
st.session_state.HW4_VectorDB = collection


# --------------------------------------------------
# Chat Memory
# --------------------------------------------------

if "hw4_messages" not in st.session_state:
    st.session_state.hw4_messages = []


# Display previous messages
for message in st.session_state.hw4_messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Chat Input
# --------------------------------------------------

if prompt := st.chat_input(
    "Ask about a student organization..."
):

    user_message = {
        "role": "user",
        "content": prompt
    }

    st.session_state.hw4_messages.append(
        user_message
    )

    with st.chat_message("user"):
        st.markdown(prompt)


    # --------------------------------------------------
    # RAG Retrieval
    # --------------------------------------------------

    query_embedding = create_embedding(
        prompt
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=6
    )

    retrieved_documents = (
        results["documents"][0]
    )

    retrieved_metadata = (
        results["metadatas"][0]
    )


    # --------------------------------------------------
    # Build RAG Context
    # --------------------------------------------------

    rag_context = ""

    for document, metadata in zip(
        retrieved_documents,
        retrieved_metadata
    ):

        rag_context += (
            f"\n\nSource: {metadata['filename']}\n"
            f"{document}"
        )


    # --------------------------------------------------
    # System Prompt
    # --------------------------------------------------

    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful Syracuse University "
            "student organization chatbot. "

            "Answer questions using simple and clear language. "

            "Answer using only the retrieved student organization "
            "information provided below. "

            "When the retrieved documents contain the answer, begin with "
            "'Based on the retrieved student organization information:' "
            "and answer the question. "

            "If the retrieved information contains only part of the answer, "
            "explain what information was found and clearly state what "
            "information was not found. "

            "If the answer cannot be found in the retrieved information, say "
            "'I could not find this information in the provided student "
            "organization documents.' "

            "Do not invent organization names, activities, contact "
            "information, or other details. "

            "\n\nRetrieved student organization information:\n"
            + rag_context
        )
    }


    # --------------------------------------------------
    # Conversation Memory
    # Keep Last 5 Complete Interactions
    # --------------------------------------------------

    previous_messages = (
        st.session_state.hw4_messages[:-1]
    )

    exchanges = []
    current_exchange = []

    for message in previous_messages:

        if message["role"] == "user":

            if current_exchange:
                exchanges.append(
                    current_exchange
                )

            current_exchange = [message]

        elif (
            message["role"] == "assistant"
            and current_exchange
        ):

            current_exchange.append(
                message
            )

            exchanges.append(
                current_exchange
            )

            current_exchange = []


    recent_messages = []

    # Homework requires memory of up to
    # the last 5 interactions.
    for exchange in exchanges[-5:]:

        recent_messages.extend(
            exchange
        )


    # Add current user question
    recent_messages.append(
        user_message
    )


    # --------------------------------------------------
    # Send to OpenAI
    # --------------------------------------------------

    messages_for_api = [
        system_prompt,
        *recent_messages
    ]


    with st.chat_message("assistant"):

        stream = client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages_for_api,
            stream=True
        )

        response = st.write_stream(
            stream
        )


    # Save Assistant Response
    st.session_state.hw4_messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )