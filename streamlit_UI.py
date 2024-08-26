import streamlit as st
from streamlit_backend import pdf_docs, chroma_vector_store, similarity_search, delete_all_ids

def main():
    st.title("Resume Similarity Score Finder")

    # Input fields for text and number of outputs
    input_text = st.text_input("Enter the text to search for similarity:")
    no_of_outfiles = st.number_input("Enter the number of similar resumes to fetch:", min_value=1, max_value=10)

    # File uploader for optional PDF uploads
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if st.button("Get Data"):
        documents = []
        if uploaded_files:
            documents = pdf_docs(uploaded_files)

        vector_store = chroma_vector_store(documents if documents else None)

        # Perform similarity search and display results
        if input_text and no_of_outfiles > 0:
            result_dict = similarity_search(vector_store, input_text, no_of_outfiles)
            st.write(result_dict)

            # Clear old documents from the vector store
            answer = delete_all_ids()
            st.write(answer)

if __name__ == "__main__":
    main()
