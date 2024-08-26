from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import streamlit as st
import plotly.graph_objs as go
import os

load_dotenv()

embedding = OpenAIEmbeddings() 

TEMP_DIR = Path("./temp_files")
TEMP_DIR.mkdir(exist_ok=True) 

def pdf_docs(uploaded_files):
    all_pdf_docs = []

    for uploaded_file in uploaded_files:
        temp_file_path = TEMP_DIR / f"{uuid4()}.pdf"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        docs = PyMuPDFLoader(str(temp_file_path)).load()
        if docs:
            doc = docs[0]
            doc.metadata['file_name'] = uploaded_file.name  # Use 'name' instead of 'filename'
            all_pdf_docs.append(doc)
        os.remove(temp_file_path)
    return all_pdf_docs


def get_vector_store():
        vector_store = Chroma(collection_name="Resume_collection",
                          embedding_function=embedding,
                          persist_directory="./chroma_langchain_db")
        return vector_store


def chroma_vector_store(documents=None):
    vector_store = Chroma(collection_name="Resume_collection",
                          embedding_function=embedding,
                          persist_directory="./chroma_langchain_db")
    if documents:
        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents)
    return vector_store



def similarity_search(vector_store, user_query, no_of_docs):
    results = vector_store.similarity_search_with_relevance_scores(user_query, k=int(no_of_docs))
    
    if len(results) == 0:
        st.write("No results found for the query.")
        return {}

    st.write(f"Retrieved {len(results)} results from similarity search.")
    data = {}

    # Create columns for the first row of graphs
    num_cols = 2  # Number of columns per row
    cols = st.columns(num_cols)

    for i, (res, score) in enumerate(results):
        file_name = res.metadata.get('file_name', 'Unknown File')
        data[file_name] = {score * 100: {"page_content": res.page_content}}
        
        # Create a meter graph for each document
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            title={'text': f"{file_name}", 'font': {'size': 15}},
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': "darkblue",
                    'tickmode': 'array',
                    'tickvals': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    'ticktext': ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
                },
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': score * 100
                }
            }
        ))

        fig.update_layout(
            width=300,  # Adjust the width of the graph
            height=300,  # Adjust the height of the graph
            margin=dict(l=20, r=20, t=50, b=20)  # Adjust the margins
        )

        # Plot in the current column
        with cols[i % num_cols]:
            st.markdown(f"**<span style='color: black;'>{file_name}</span>**", unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, key=f"plot_{file_name}")
    
    return data




def delete_all_ids():
    vector_store = get_vector_store()
    ids = vector_store.get()['ids']
    if not ids:
        return {"message": "Vecotod Database is already empty, no IDs found. please add pdfs only"}
    else:
        vector_store.delete(ids)
        return {"message": "All old IDs deleted successfully."}



