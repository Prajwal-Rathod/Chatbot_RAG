import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## RAG Chatbot
Upload the PDF file and ask a question.
""")

# Get API key from secrets
api_key = st.secrets["general"]["google_api_key"]

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "The answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.header("Document Genie Chatbot")

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Processing complete. You can now ask questions.")
        
        # Add developer information at the bottom
        st.markdown("---")
        st.markdown("Developed by Prajwal C S")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        for speaker, message in st.session_state.conversation:
            st.markdown(f"**{speaker}:** {message}")

    # Input at the bottom
    user_question = st.text_input("Ask a question from the PDF files", key="user_question", on_change=lambda: handle_question(st.session_state.user_question))

def handle_question(user_question):
    if user_question:
        st.session_state.conversation.append(("You", user_question))
        answer = user_input(user_question, api_key)
        st.session_state.conversation.append(("AI", answer))
        # Clear the input after submission
        st.session_state.user_question = ""

    # Re-render the chat interface with the new conversation
    with st.container():
        for speaker, message in st.session_state.conversation:
            st.markdown(f"**{speaker}:** {message}")

if __name__ == "__main__":
    main()
