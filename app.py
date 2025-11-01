import streamlit as st
import google.generativeai as genai
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- Page Configuration ---
st.set_page_config(page_title="Chat with Faisal", page_icon="🤖")
st.title("🤖 Chat with Faisal's Profile")
st.caption("Ask me anything about Faisal's resume, projects, or background.")

# Load profile data from a text file in your repo
@st.cache_data
def load_document():
    # Make sure you have a file named 'profile.txt' in your GitHub repo
    try:
        with open("profile.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        st.error("Error: 'profile.txt' file not found in the repository root.")
        st.stop()

document_text = load_document()

# --- RAG Setup (In-Memory) ---
# This block runs only once and is cached
@st.cache_resource
def setup_rag_pipeline(api_key):
    # 1. Configure the Google API
    genai.configure(api_key=api_key)
    
    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(document_text)
    
    # 3. Create Google embeddings
    # Make sure your API key has the "Generative Language API" enabled
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Google Embeddings: {e}")
        st.info("Please ensure your Google API key is correct and has the 'Generative Language API' enabled in your Google AI Studio project.")
        st.stop()

    # 4. Create the FAISS in-memory vector store
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    # 5. Create the retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Get top 3 chunks
    
    # 6. Initialize the Generative Model
    llm = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    return retriever, llm

# --- Get API Key from Streamlit Secrets ---
google_api_key = st.secrets.get("GOOGLE_API_KEY")

if not google_api_key:
    st.info("Please add your Google AI Studio API key to this app's secrets.")
    st.stop()

# --- Initialize RAG ---
try:
    retriever, llm = setup_rag_pipeline(google_api_key)
except Exception as e:
    st.error(f"Failed to initialize the RAG pipeline: {e}")
    st.stop()


# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Hi! How can I help you with Faisal's profile today?"
    }]

for message in st.session_state.messages:
    # Use "model" for assistant and "user" for user
    role = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. Retrieve relevant context
            retrieved_docs = retriever.get_relevant_documents(prompt)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # 2. Build the prompt for the LLM
            final_prompt = f"""
            You are a helpful assistant for Faisal Rahman Chowdhury. 
            Answer the user's question based *only* on the context provided below.
            If the answer is not in the context, say "I'm sorry, I don't have that information in Faisal's profile."
            
            CONTEXT:
            {context}
            
            QUESTION:
            {prompt}
            
            ANSWER:
            """
            
            # 3. Generate the response
            try:
                response = llm.generate_content(final_prompt)
                answer = response.text
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error generating response: {e}")
