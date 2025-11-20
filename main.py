from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA


#using mistral model
llm = Ollama(model="mistral")
#loading and splitting
loader = TextLoader("speech.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=230,
    chunk_overlap=35
)
chunks = text_splitter.split_documents(documents)

#storing embeddings in the vector db
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="db")
retriever = vectordb.as_retriever()

#building a QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)
#checking the system 
query = "What does the speech say about the shastras?"
result = qa_chain({"query": query})
print("Answer:", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(doc.page_content)




