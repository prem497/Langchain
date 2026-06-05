#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 1. Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# 2. Setup LLM and Embeddings
llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-pro")
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 3. Load your document
with open("sample.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 4. Split it into chunks
splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=50)
chunks = splitter.split_text(raw_text)
documents = [Document(page_content=chunk) for chunk in chunks]

# 5. Compress the documents using LLMChainExtractor
compressor = LLMChainExtractor.from_llm(llm)
compressed_docs = compressor.compress_documents(documents, query="Summarize the key points")

print("✅ Compressed Summary:")
for doc in compressed_docs:
    print("-", doc.page_content)

# 6. Now ask a question about the compressed content using a custom LLMChain
qa_prompt = PromptTemplate.from_template(
    "Given the context below, answer the question:\n\nContext:\n{context}\n\nQuestion: {question}"
)
qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

response = qa_chain.invoke({
    "context": "\n".join([doc.page_content for doc in compressed_docs]),
    "question": "What is the main idea of the document?"
})

print("\n💡 Answer to your question:")
print(response["text"])


# In[ ]:




