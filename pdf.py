import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

# STEP 1: open PDF and convert to text
def extract_text_from_pdf(pdf_file) -> str:
    text_output = ''
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    # Check if the PDF is encrypted
    if pdf_reader.is_encrypted:
        try:
            pdf_reader.decrypt("")  # Attempt decryption with an empty password
        except:
            raise ValueError("The PDF is encrypted and requires a password to open.")
    
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:  # Ensure text exists
            text_output += page_text
        else:
            text_output += "\n[No text found on this page]\n"  # Handle pages with no extractable text
            
    return text_output

# Function to perform chunking of text for better retrieval
def chunk_text(text, chunk_size=512, overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                   chunk_overlap=overlap)
    return text_splitter.split_text(text)
