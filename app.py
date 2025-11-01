import streamlit as st
import os

# --- AWS & LangChain Imports ---
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# --- Page Configuration ---
st.set_page_config(page_title="Chat with Faisal", page_icon="🤖")
st.title("🤖 Chat with Faisal's Profile")
st.caption("Ask me anything about Faisal's resume, projects, or background.")

# --- Get AWS Credentials from Streamlit Secrets ---
# We will set these in the Streamlit Cloud settings in the next step
AWS_ACCESS_KEY_ID = st.secrets.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = st.secrets.get("AWS_REGION", "us-east-1") # Default to us-east-1

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    st.info("Please add your AWS Access Key and Secret Key to this app's secrets.")
    st.stop()

# --- Load Document ---
@st.cache_data
def load_document():
    try:
        # Make sure you have a file named 'profile.txt' in your GitHub repo
        with open("profile.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        st.error("Error: 'profile.txt' file not found in the repository root.")
        st.stop()

document_text = load_document()

# --- RAG Setup (In-Memory) ---
@st.cache_resource
def setup_rag_pipeline():
    try:
        # 1. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(document_text)
        
        # 2. Create AWS Bedrock Embeddings client
        embeddings = BedrockEmbeddings(
            credentials_profile_name=None, # Use keys from secrets
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            model_id="amazon.titan-embed-text-v1" # This is the model for embeddings
        )
        
        # 3. Create the FAISS in-memory vector store
        vector_store = FAISS.from_texts(chunks, embeddings)
        
        # 4. Create the retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 5. Initialize the Bedrock Chat Model (Claude)
        llm = ChatBedrock(
            credentials_profile_name=None,
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            model_id="anthropic.claude-3-sonnet-20240229-v1:0", # This is the model for chatting
            model_kwargs={"temperature": 0.1}
        )
        
        return retriever, llm
        
    except Exception as e:
        # Catch potential new-user Anthropic error
        if "is not authorized to perform" in str(e) and "anthropic" in str(e):
            st.error(f"AWS Error: {e}")
            st.info("""
                This is a common first-time user error for Anthropic models. 
                Please go to your AWS Bedrock console, find 'Model catalog' in the menu,
                select 'Claude 3 Sonnet', and click 'Request access'. 
                You may need to submit a brief use-case form.
                After you get access, please reboot this app.
            """)
            st.stop()
        else:
            st.error(f"Failed to initialize the RAG pipeline: {e}")
            st.stop()

# --- Initialize RAG ---
retriever, llm = setup_rag_pipeline()

# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Hi! How can I help you with Faisal's profile today?"
    }]

for message in st.session_state.messages:
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
            # Claude 3 models use a different prompt structure
            final_prompt = f"""
            Human: You are a helpful assistant for Faisal Rahman Chowdhury. 
            Answer the user's question based *only* on the context provided below.
            If the answer is not in the context, say "I'm sorry, I don't have that information in Faisal's profile."
            
            CONTEXT:
            {context}
            
            QUESTION:
            {prompt}
            
            Assistant:
            """
            
            # 3. Generate the response
            try:
                response = llm.invoke(final_prompt) # LangChain .invoke
                answer = response.content # LangChain .content
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error generating response: {e}")
