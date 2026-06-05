#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_classic.agents import Tool, initialize_agent, AgentType
from langchain_classic.memory import ConversationBufferMemory

# Load API keys
load_dotenv(".env")
openai_api_key = os.getenv("OPENAI_API_KEY")

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Tool: Simple QA
qa_prompt = PromptTemplate.from_template("Answer clearly: {question}")
qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
qa_tool = Tool(
    name="Simple QA",
    func=qa_chain.run,
    description="Answer factual questions clearly"
)



# ![Model Diagram](memory.png)

# In[ ]:


#https://python.langchain.com/api_reference/langchain/memory.html?utm_source=chatgpt.com


# 

# In[6]:


#🧠 1. ConversationBufferMemory
from langchain_classic.memory import ConversationBufferMemory

# Memory (stores chat history)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,handle_parsing_errors=True)

# Agent with tool + memory
agent = initialize_agent(
    tools=[qa_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,# ZERO_SHOT_REACT_DESCRIPTION
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

# Run multiple interactions
print("1️⃣ First Question")
res1 = agent.run("What is LangChain?")
print("\nAnswer:", res1)

print("\n2️⃣ Follow-up Question")
res2 = agent.run("Who created it?")
print("\nAnswer:", res2)

print("\n3️⃣ Ask again about previous topic")
res3 = agent.run("Explain it simply again.")
print("\nAnswer:", res3)

#print(memory.chat_memory.messages)

for msg in memory.chat_memory.messages:
    print(f"{msg.type.upper()}: {msg.content}")


# In[7]:


#2)ConversationBufferWindowMemory

from langchain_classic.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentType

memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history")

agent = initialize_agent(
    tools=[qa_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

# Run multiple interactions
print("1️⃣ First Question")
res1 = agent.run("What is LangChain?")
print("\nAnswer:", res1)

print("\n2️⃣ Follow-up Question")
res2 = agent.run("Who created it?")
print("\nAnswer:", res2)

print("\n3️⃣ Ask again about previous topic")
res3 = agent.run("Explain it simply again.")
print("\nAnswer:", res3)

for msg in memory.chat_memory.messages:
    print(f"{msg.type.upper()}: {msg.content}")


# In[10]:


#🧠 3. ConversationSummaryMemory
from langchain_classic.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history")
# Agent with tool + memory
agent = initialize_agent(
    tools=[qa_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)


# Run multiple interactions
print("1️⃣ First Question")
res1 = agent.run("What is LangChain?")
print("\nAnswer:", res1)

print("\n2️⃣ Follow-up Question")
res2 = agent.run("Who created it?")
print("\nAnswer:", res2)

print("\n3️⃣ Ask again about previous topic")
res3 = agent.run("Explain it simply again.")
print("\nAnswer:", res3)

for msg in memory.chat_memory.messages:
    print(f"{msg.type.upper()}: {msg.content}")


# In[14]:


from langchain_classic.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()
retriever = FAISS.from_texts(["initial memory"], embedding).as_retriever()

memory = VectorStoreRetrieverMemory(retriever=retriever,memory_key="chat_history")
# Agent with tool + memory
agent = initialize_agent(
    tools=[qa_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

# Run multiple interactions
print("1️⃣ First Question")
res1 = agent.run("What is LangChain?")
print("\nAnswer:", res1)

print("\n2️⃣ Follow-up Question")
res2 = agent.run("Who created it?")
print("\nAnswer:", res2)

print("\n3️⃣ Ask again about previous topic")
res3 = agent.run("Explain it simply again.")
print("\nAnswer:", res3)


# In[11]:


# Check what's stored in FAISS retriever
docs = retriever.vectorstore.similarity_search("memory")
for i, doc in enumerate(docs):
    print(f"🔹 Document {i+1}: {doc.page_content}")


# In[12]:


# See all texts in FAISS index
print("All indexed texts:")
for i, doc in enumerate(retriever.vectorstore.docstore._dict.values()):
    print(f"📄 {i+1}: {doc.page_content}")


# In[15]:


query = "LangChain founder"
results = memory.retriever.get_relevant_documents(query)

print(f"\n🧠 Retrieval for: '{query}'")
for i, doc in enumerate(results):
    print(f"🔹 {i+1}: {doc.page_content}")


# In[ ]:


memory.save_context(
    {"input": "Who is the founder of LangChain?"},
    {"output": "Harrison Chase is the founder of LangChain."}
)


# In[ ]:


#🧠 6. PostgresChatMessageHistory, RedisChatMessageHistory, etc.

from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import PostgresChatMessageHistory

history = PostgresChatMessageHistory(session_id="abc123", connection_string="postgresql://...")
memory = ConversationBufferMemory(chat_memory=history)


# In[ ]:




