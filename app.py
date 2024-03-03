from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
import os

url = "Financebench-1.csv"

loader = CSVLoader(file_path=url)

embeddings = HuggingFaceEmbeddings()

transcript = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(transcript)
db = FAISS.from_documents(docs, embeddings)

os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
  return 'Hello from Flask!'


@app.route('/search', methods=['POST'])
async def search():
  k=1
  docs = db.similarity_search(request.json['question'], k=k)
  docs_page_content = " ".join([d.page_content for d in docs])

  llm = HuggingFaceHub(repo_id="Xenova/text-davinci-003", task="text-generation")

  prompt = PromptTemplate(
      input_variables=["question", "docs"],
      template="""
        Answer the following question: {question}
        By searching the following video transcript: {docs}
      """,
  )
  chain = LLMChain(llm=llm, prompt=prompt)

  response = chain.run(question=request.json['question'], docs=docs_page_content)
  response = response.replace("\n", "")
  return jsonify({
    "answer": response
  })

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5002)