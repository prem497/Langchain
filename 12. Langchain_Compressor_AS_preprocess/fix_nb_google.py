import json

filename = "12)Langchain_Compressor_As_preprocessing.ipynb"

with open(filename, "r", encoding="utf-8") as f:
    data = json.load(f)

for cell in data.get("cells", []):
    if cell.get("cell_type") == "code":
        new_source = []
        for line in cell.get("source", []):
            line = line.replace('llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-pro")', 'llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-flash-latest")')
            line = line.replace('embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")', 'embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")')
            new_source.append(line)
        cell["source"] = new_source

with open(filename, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)

print("Notebook updated to use gemini-flash-latest and gemini-embedding-2.")
