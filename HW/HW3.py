import streamlit as st
from openai import OpenAI
from google import genai
import requests
from bs4 import BeautifulSoup


def read_url_content(url):
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        return soup.get_text(separator=" ", strip=True)

    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None


st.title("Homework 3 - URL Chatbot")

st.write(
    "This chatbot answers questions using the webpage URLs you provide. "
    "You can provide one or two URLs and choose which LLM to use. "
    "The webpage content stays available as context throughout the conversation. "
    "The chatbot remembers the last 3 user-assistant exchanges."
)


# Sidebar URL inputs
url1 = st.sidebar.text_input(
    "URL 1",
    placeholder="https://example.com"
)

url2 = st.sidebar.text_input(
    "URL 2 (optional)",
    placeholder="https://example.com"
)


# LLM selection
llm_choice = st.sidebar.selectbox(
    "Choose an LLM",
    [
        "OpenAI",
        "Gemini"
    ]
)


# Premium models
if llm_choice == "OpenAI":
    model_name = "gpt-5-mini"
else:
    model_name = "gemini-3.5-flash"

st.sidebar.caption(f"Selected model: {model_name}")


# API keys
openai_api_key = st.secrets["OPENAI_API_KEY"]
gemini_api_key = st.secrets["GEMINI_API_KEY"]


# API clients
openai_client = OpenAI(
    api_key=openai_api_key
)

gemini_client = genai.Client(
    api_key=gemini_api_key
)


# Read URL content
webpage_context = ""

if url1:
    with st.spinner("Reading URL 1..."):
        content1 = read_url_content(url1)

    if content1:
        webpage_context += (
            "\n\n--- WEBPAGE 1 ---\n"
            + content1
        )


if url2:
    with st.spinner("Reading URL 2..."):
        content2 = read_url_content(url2)

    if content2:
        webpage_context += (
            "\n\n--- WEBPAGE 2 ---\n"
            + content2
        )


# Reset conversation if URLs or model change
current_settings = (url1, url2, llm_choice)

if "current_settings" not in st.session_state:
    st.session_state.current_settings = current_settings

if st.session_state.current_settings != current_settings:
    st.session_state.messages = []
    st.session_state.current_settings = current_settings


# Initialize conversation memory
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Only enable chatbot when at least one URL is provided
if not webpage_context:

    st.info(
        "Enter at least one webpage URL in the sidebar to start chatting."
    )

else:

    # System prompt containing permanent webpage context
    system_content = (
        "You are a helpful chatbot. "
        "Answer the user's questions using the webpage content provided below. "
        "Use the webpage content as your main source of information. "
        "If the answer cannot be found in the provided webpage content, "
        "say that the provided webpages do not contain enough information "
        "to answer the question.\n\n"
        "WEBPAGE CONTEXT:\n"
        f"{webpage_context}"
    )


    # Get user question
    if prompt := st.chat_input(
        "Ask a question about the webpage..."
    ):

        user_message = {
            "role": "user",
            "content": prompt
        }

        st.session_state.messages.append(user_message)

        with st.chat_message("user"):
            st.markdown(prompt)


        # Keep the last 3 completed user-assistant exchanges
        previous_messages = st.session_state.messages[:-1]

        exchanges = []
        current_exchange = []

        for message in previous_messages:

            if message["role"] == "user":

                if current_exchange:
                    exchanges.append(current_exchange)

                current_exchange = [message]

            elif (
                message["role"] == "assistant"
                and current_exchange
            ):

                current_exchange.append(message)
                exchanges.append(current_exchange)
                current_exchange = []


        recent_messages = []

        for exchange in exchanges[-3:]:
            recent_messages.extend(exchange)


        # Add current user question
        recent_messages.append(user_message)


        # OpenAI
        if llm_choice == "OpenAI":

            messages_for_api = [
                {
                    "role": "system",
                    "content": system_content
                },
                *recent_messages
            ]

            try:

                with st.chat_message("assistant"):

                    stream = openai_client.chat.completions.create(
                        model=model_name,
                        messages=messages_for_api,
                        stream=True
                    )

                    response = st.write_stream(stream)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response
                    }
                )

            except Exception as e:

                st.error(f"OpenAI error: {e}")


        # Gemini
        elif llm_choice == "Gemini":

            conversation_text = ""

            for message in recent_messages:

                conversation_text += (
                    f"\n{message['role'].upper()}: "
                    f"{message['content']}\n"
                )


            gemini_prompt = (
                f"{system_content}\n\n"
                "RECENT CONVERSATION:\n"
                f"{conversation_text}\n\n"
                "Answer the most recent USER question."
            )


            try:

                with st.chat_message("assistant"):

                    response = (
                        gemini_client.models.generate_content(
                            model=model_name,
                            contents=gemini_prompt
                        )
                    )

                    answer = response.text

                    st.markdown(answer)


                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(f"Gemini error: {e}")