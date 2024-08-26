import streamlit as st
from streamlit_backend import pdf_docs, qdrant_vector_store, similarity_search, delete_all_ids

# Set custom tab title and favicon
st.set_page_config(
    page_title="Resume Score Finder TRIPATHI",  # Custom tab title
    page_icon=":page_facing_up:",  # Use emoji as favicon
    # Alternatively, use a local file or URL for the icon
    # page_icon="path/to/resume_icon.png",
)

def main():
    st.markdown("<h1 style='color:#800000;'>TRIPATHI   UTKARSH</h1>", unsafe_allow_html=True)
    st.title("Resume Similarity Score Finder")

    input_text = st.text_input("Enter the text to search for similarity:")
    no_of_outfiles = st.number_input("Enter the number of similar resumes to fetch:", min_value=1, max_value=10)

    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if st.button("Get Data"):
        if not input_text:
            st.warning("Please enter Job Description OR text to search for similarity.")
        if not no_of_outfiles:
            st.warning("Please enter the number of similar resumes to fetch.")
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")

        if input_text and no_of_outfiles and uploaded_files:
            documents = pdf_docs(uploaded_files)
            vector_store = qdrant_vector_store(documents)

            result_dict = similarity_search(vector_store, input_text, no_of_outfiles)
            st.write(result_dict)

            answer = delete_all_ids()
            st.write(answer)

if __name__ == "__main__":
    main()
