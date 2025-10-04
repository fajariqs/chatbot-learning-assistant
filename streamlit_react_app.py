# Import the necessary libraries
import streamlit as st  # For creating the web app interface
from langchain_google_genai import ChatGoogleGenerativeAI  # For interacting with Google Gemini via LangChain
from langgraph.prebuilt import create_react_agent  # For creating a ReAct agent
from langchain_core.messages import HumanMessage, AIMessage  # For message formatting

# --- 1. Page Configuration and Title ---

st.title("📚 Chatbot Asisten Belajar")
st.caption("Teman belajar interaktif dengan Google Gemini + LangGraph")

# --- 2. Sidebar for Settings ---

with st.sidebar:
    st.subheader("⚙️ Settings")
    google_api_key = st.text_input("Google AI API Key", type="password")
    mode = st.selectbox("Pilih Mode Belajar", ["umum", "Matematika", "IPA", "Bahasa Indonesia", "Bahasa Inggris"])
    reset_button = st.button("Reset Conversation", help="Clear all messages and start fresh")

# --- Tambahin fungsi untuk mode belajar ---
def get_base_prompt(mode):
    if mode == "umum":
        return "Kamu adalah asisten belajar ramah. Jawablah dengan sederhana."
    elif mode == "Matematika":
        return "Kamu adalah guru Matematika. Selalu jelaskan langkah-langkah dengan rumus."
    elif mode == "IPA":
        return "Kamu adalah guru IPA. Jelaskan konsep dengan contoh sehari-hari."
    elif mode == "Bahasa Indonesia":
        return "Kamu adalah guru Bahasa Indonesia. Koreksi tata bahasa dengan baik."
    elif mode == "Bahasa Inggris":
        return "You are an English teacher. Answer in simple English, then give the Indonesian translation below."
    return "Kamu adalah asisten belajar."

# --- 3. API Key and Agent Initialization ---

if not google_api_key:
    st.info("Please add your Google AI API key in the sidebar to start chatting.", icon="🗝️")
    st.stop()

if ("agent" not in st.session_state) or (getattr(st.session_state, "_last_key", None) != google_api_key):
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.5
        )

        # Gunakan prompt sesuai mode
        base_prompt = get_base_prompt(mode)

        st.session_state.agent = create_react_agent(
            model=llm,
            tools=[],
            prompt=base_prompt
        )

        st.session_state._last_key = google_api_key
        st.session_state.pop("messages", None)
    except Exception as e:
        st.error(f"Invalid API Key or configuration error: {e}")
        st.stop()

# --- 4. Chat History Management ---

if "messages" not in st.session_state:
    st.session_state.messages = []

if reset_button:
    st.session_state.pop("agent", None)
    st.session_state.pop("messages", None)
    st.rerun()

# --- 5. Display Past Messages ---

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. Handle User Input and Agent Communication ---

prompt = st.chat_input("Type your message here...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        # Gabungkan mode dengan pertanyaan user
        full_prompt = get_base_prompt(mode) + "\n\nPertanyaan siswa: " + prompt

        response = st.session_state.agent.invoke({"messages": [HumanMessage(content=full_prompt)]})

        if "messages" in response and len(response["messages"]) > 0:
            answer = response["messages"][-1].content
        else:
            answer = "Maaf, aku tidak bisa menjawab sekarang."

    except Exception as e:
        answer = f"Terjadi error: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
