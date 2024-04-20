from src.helper import load_pdf, text_split, download_huggingface_embeddings
from langchain.vectorstores import Chroma

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_huggingface_embeddings()

persist_directory = 'db'
vectordb = Chroma.from_documents(documents = text_chunks,
                                 embedding = embeddings,
                                 persist_directory = persist_directory)

vectordb.persist()

vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)
