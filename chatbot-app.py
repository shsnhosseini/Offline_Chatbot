# Full Offline Mode Setup with No Network Call
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_community.document_loaders import PyMuPDFLoader, PyPDFium2Loader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import tempfile
import torch  

# Streamlit UI
st.title("Offline Chatbot", text_alignment="center")
uploaded_file = st.file_uploader("Upload a PDF file and ask questions about its content.",
                                  type=["pdf"])

# Setting up HuggingFace embeddings (Offline)
def embeddings():
    print(">>> Setting up embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="./all-mpnet-base-v2",
                                       model_kwargs={"local_files_only": True},
                                       encode_kwargs={"normalize_embeddings": True} # improves cosine similarity calculations used by Elasticsearch
                                       )
    return embeddings

em = embeddings()

# Setting up Elasticsearch vector store
# Using Elasticsearch for storing vector embeddings of the PDF content for efficient similarity search.
vector_db = ElasticsearchStore(embedding=em, 
                               es_url="http://localhost:9200", index_name="pdf_embeddings")
                               

# Loading PDF and saving to Elasticsearch
def loadpdf(file):
    loader2 = PyMuPDFLoader(file)
    data2 = loader2.load()
    vector_db.from_documents(data2, embedding=em, es_url="http://localhost:9200", index_name="pdf_embeddings")
    st.session_state.pdf = True #  PDF successfully processed and ready for use


# LLM Setup (Offline)
def flan_t5_small_model():
    """
    Initializes and returns a LangChain LLM wrapper for the Flan-T5-Small model.
    
    Loads the model and tokenizer from a local path, configures the pipeline 
    for text-to-text generation, and maps the model to available GPU/CPU automatically.
    
    Returns:
        HuggingFacePipeline: A LangChain-compatible LLM instance.
        
    Note:
        Requires 'accelerate' library for device_map='auto'.
        Assumes model is downloaded to './google_flan_t5_small'.
    """
    print(">>> Setting up Flan-T5-small model...")
    model_path= "./google_flan_t5_small"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # Load model onto GPU automatically if available
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

    device = 0 if torch.cuda.is_available() else -1

    pipe = pipeline("text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    truncation=True)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def llm_chain():
    prompt_template = """Given the following context, answer the question: {question}
    Context: {context}. when you don't know the answer, say "I don't know"."""
    prompt = PromptTemplate.from_template(prompt_template)
    llm = flan_t5_small_model()
    output_parser = StrOutputParser()
    # Chain them together
    chain = prompt | llm | output_parser
    return chain

# llm_chain = llm_chain() 
if "llm" not in st.session_state:
    llm_chain = llm_chain()
    st.session_state.llm = llm_chain
else:
    llm_chain = st.session_state.llm

if "pdf" not in st.session_state:
    st.session_state.pdf = False


if uploaded_file and not st.session_state.pdf:
    with st.spinner("Processing PDF..."):
        temp_file_path = tempfile.mktemp(suffix=".pdf")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        loadpdf(temp_file_path)
        os.remove(temp_file_path)  # Clean up the temporary file
        st.success("PDF processed and indexed successfully!")


def ask(question):
    similar_doc = vector_db.similarity_search(question)
    # saving App from crashing. User gets an honest, non-hallucinated answer. LLM replies "I don't know"
    context = similar_doc[0].page_content if similar_doc else "No relevant information found in the PDF." 
    answer = llm_chain.invoke({"question": question, "context": context})
    return answer 


user_question = st.text_area("Please Enter Your Question")
if user_question:
    with st.spinner("Generating answer..."):
        answer = ask(user_question)
        st.write("Answer:", answer)
