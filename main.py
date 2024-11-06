import streamlit as st
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from pdf import extract_text_from_pdf, chunk_text

# Initialize Ollama embeddings with Llama 2
embeddings = OllamaEmbeddings(model="llama2")

# Streamlit application
def main():
    st.title("Tamil Document Retrieval with Llama 2 Embeddings")
    st.text("Upload your Tamil PDF and ask a query to retrieve relevant information.")

    # File uploader for PDF files
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file:
        # Extract and split text into chunks
        text = extract_text_from_pdf(uploaded_file)
        text_chunks = chunk_text(text)

        # Create Chroma vector store with embedded text chunks
        vector_store = Chroma.from_texts(text_chunks, 
                                         embedding=embeddings, 
                                         persist_directory="./chroma_db")

        # User query input
        query = st.text_input("Ask your query:")

        if query:
            # Embed the query and search for relevant chunks
            relevant_docs = vector_store.similarity_search(query, k=3)  # Top 3 results
            
            # Display relevant chunks
            st.subheader("Relevant Information")
            for i, doc in enumerate(relevant_docs):
                st.write(f"### Chunk {i+1}")
                st.write(doc.page_content)


if __name__ == "__main__":
    main()
