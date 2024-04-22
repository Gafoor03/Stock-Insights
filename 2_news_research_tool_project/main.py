import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import FAISS
from langchain.embeddings import OpenAIEmbeddings
from bs4 import BeautifulSoup
import requests

from dotenv import load_dotenv
load_dotenv()  

st.title("Stock Insights And Analysis Using RAG MODEL 📈")
st.title("ENTER THE STOCK NAME")



user_input = st.text_input("")
url = 'https://www.google.co.in/search?q={}finance news'.format(user_input )
response = requests.get(url)
with open('sample.txt','w') as f:
    f.write(response.text)
soup = BeautifulSoup(response.content, 'html5lib')
urls = []
for i in  soup.find_all("a"):
    urls.append(i['href'][7:])
#for i in range(3):
    #url = st.sidebar.text_input(f"URL {i+1}")
    #urls.append(url)
process_url_clicked = st.button("PROCESS")
#process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9, max_tokens=500)

if process_url_clicked:
   
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
  
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

   
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
           
            st.header("Answer")
            st.write(result["answer"])

           
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n") 
                for source in sources_list:
                    st.write(source)
