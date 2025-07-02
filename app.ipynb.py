# prompt: why my code give same answer to all different question

from google.colab import files

# Install necessary libraries
!pip install -q langchain langchain-community faiss-cpu sentence-transformers transformers accelerate pymupdf gradio duckduckgo-search

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
from langchain.llms.base import LLM
from pydantic import Extra, Field
from typing import Optional, List, Any
from langchain.chains import RetrievalQA

# Upload the file
uploaded = files.upload()

# Assuming only one file is uploaded and its name is "Kartikey Tiwari resume 4.pdf"
# If the file name is different, update the line below accordingly
file_name = list(uploaded.keys())[0] # Get the uploaded file name
if file_name not in uploaded:
    print(f"Error: File '{file_name}' not found in uploaded files.")
else:
    # Load and process the document
    loader = PyMuPDFLoader(file_name)
    docs = loader.load()
    print(f"✅ Loaded pages: {len(docs)}")

    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    separators=["\n\n", "\n", ".", " ", ""]

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embedding_model)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Define and instantiate the LocalLLM
    class LocalLLM(LLM):
        pipeline: Any = Field(default=None)
        class Config:
            extra = Extra.allow

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Consider moving pipeline loading outside __init__ if memory is a concern
            # and you have multiple instances, though here it's a singleton.
            self.pipeline = pipeline("text2text-generation", model="google/flan-t5-large")

        @property
        def _llm_type(self) -> str:
            return "flan_t5"

        def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
            # FLAN-T5 has a max input length, truncating the prompt if necessary
            return self.pipeline(prompt[:1500], max_new_tokens=256)[0]["generated_text"]

    llm = LocalLLM()

    # Create the RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Ask multiple questions using the same qa object
    print("\n--- Asking Questions ---")

    question1 = "Summarize the main idea of the uploaded PDF."
    print(f"\nQ: {question1}")
    result1 = qa.invoke(question1)
    print(f"A: {result1['result']}")
    # print source documents if needed: print(f"Source Documents: {result1['source_documents']}")

    question2 = "What is Kartikey Tiwari’s education background?"
    print(f"\nQ: {question2}")
    result2 = qa.invoke(question2)
    print(f"A: {result2['result']}")

    question3 = "List some projects mentioned in the resume."
    print(f"\nQ: {question3}")
    result3 = qa.invoke(question3)
    print(f"A: {result3['result']}")

    question4 = "What skills are listed?"
    print(f"\nQ: {question4}")
    result4 = qa.invoke(question4)
    print(f"A: {result4['result']}")