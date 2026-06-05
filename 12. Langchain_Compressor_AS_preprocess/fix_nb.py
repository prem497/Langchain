import json

filename = "12)Langchain_Compressor_As_preprocessing.ipynb"

with open(filename, "r", encoding="utf-8") as f:
    data = json.load(f)

for cell in data.get("cells", []):
    if cell.get("cell_type") == "code":
        new_source = []
        for line in cell.get("source", []):
            line = line.replace("from langchain.chat_models import ChatOpenAI", "from langchain_openai import ChatOpenAI")
            line = line.replace("from langchain.embeddings import OpenAIEmbeddings", "from langchain_openai import OpenAIEmbeddings")
            line = line.replace("from langchain.text_splitter import CharacterTextSplitter", "from langchain_text_splitters import CharacterTextSplitter")
            line = line.replace("from langchain.vectorstores import FAISS", "from langchain_community.vectorstores import FAISS")
            line = line.replace("from langchain.schema.document import Document", "from langchain_core.documents import Document")
            line = line.replace("from langchain.prompts import PromptTemplate", "from langchain_core.prompts import PromptTemplate")
            line = line.replace("from langchain.retrievers.document_compressors import LLMChainExtractor", "from langchain_classic.retrievers.document_compressors import LLMChainExtractor")
            line = line.replace("from langchain.chains import LLMChain", "from langchain_classic.chains import LLMChain")
            new_source.append(line)
        cell["source"] = new_source

with open(filename, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)

print("Notebook fixed.")
