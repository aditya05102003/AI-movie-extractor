from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv("MISTRAL_API_KEY"))
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import MistralAIEmbeddings

embedding = MistralAIEmbeddings(
    model="mistral-embed"
)

vector = embedding.embed_query("you are gen ai")
texts =[
    "hello i am aditya i am very good boy"
]
vector = embedding.embed_documents(texts)

print(vector[:5])
print(len(vector))
