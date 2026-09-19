import streamlit as st

# Import DocumentResearchAssistant class from main.py to use its functions
from main import DocumentResearchAssistant

assistant = DocumentResearchAssistant()

# Streamlit UI 

# Divide Page into 3 Sections
col1, col2, col3 = st.columns([1,9,1])

# Content Inside second section of the page
with col2:
    st.image("src/document_logo.jpg", width=600)
    st.title("Document Search Assistant")
    st.write("""
    Upload a PDF and let AI help you explore its content. Ask questions in natural
    language and get relevant answers based on the information available in your document.
    """)
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])


# File Upload
if uploaded_file:
    file_path = "documents/" + uploaded_file.name

    with open(file_path,"wb") as f:
        f.write(uploaded_file.getbuffer())  

    assistant.ingestion(file_path)


# ChatBox 
question = st.chat_input("""Ask a question about your document...""")
if question:
    query, answer, contexts = assistant.retrieval(question)
    st.write("Answer:", answer)

