from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embeddings
from src.prompt import *
from langchain.vectorstores import Chroma
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

app = Flask(__name__)

embeddings = download_huggingface_embeddings()

persist_directory = 'db'
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)

retriever = vectordb.as_retriever(search_kwargs={"k":2})

PROMPT = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question'])

chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type='llama',
                    config={'max_new_tokens': 512,
                    'temperature': 0.8})

qa_chain = RetrievalQA.from_chain_type(llm = llm,
                                       chain_type='stuff',
                                       retriever=retriever,
                                       return_source_documents=True,
                                       chain_type_kwargs=chain_type_kwargs)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa_chain({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(debug=True)

