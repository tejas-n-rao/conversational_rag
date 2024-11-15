import os

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from pinecone import Pinecone, ServerlessSpec


load_dotenv()

# Load the document
text_loader = TextLoader("data/document_test.txt")
raw_text = text_loader.load()


# Splitting Data using Semantic Chunker: Works better than RecursiveTextSplitter
text_splitter = SemanticChunker(embeddings = OpenAIEmbeddings(api_key = os.getenv("OPENAI_API_KEY")))
text_chunks = text_splitter.split_documents(raw_text)
chunks_text = [chunk.page_content for chunk in text_chunks]


# Choosing Embedding model: OpenAI here.
embedding_model = OpenAIEmbeddings(api_key = os.getenv("OPENAI_API_KEY"))

# Embedding chunks and storing in chunk_embeds
chunk_embeds = embedding_model.embed_documents(chunks_text)


spec = ServerlessSpec(cloud="aws", region="us-east-1") # spec instance
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY")) # pinecone object


index_name = "courses-ds"

# Deleting indexes with the same name: To avoid redundancy
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)


# Creating Index 
dimensions = 1536
pc.create_index(name=index_name, dimension=dimensions, spec=spec, metric="cosine")
index = pc.Index(index_name)
# index.describe_index_stats()

# Upserting text into Index to create VectorDB
ids = [str(i) for i in range(len(chunks_text))]
texts = [{"text": chunks_text[i]} for i in range(len(chunks_text))]
index.upsert(vectors=list(zip(ids, chunk_embeds, texts)))
