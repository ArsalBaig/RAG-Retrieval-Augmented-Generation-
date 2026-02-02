import warnings
warnings.filterwarnings("ignore")
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

load_dotenv()

# QA pipeline
pipe = pipeline(
    "question-answering",
    model="deepset/tinyroberta-squad2"
)

# Tokenizer + Model.
tokenizer = AutoTokenizer.from_pretrained("deepset/tinyroberta-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/tinyroberta-squad2")

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Chroma vector DB
persist_dir = "./chroma_db"

db = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings
)


def ask_question(question):
    # Convert db into retriever interface.
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question) # invoke() performs the similarity search.

    print(f"\nFound {len(docs)} documents.\n")

    # Combine document text
    context = "\n".join([doc.page_content for doc in docs])

    # HuggingFace QA
    answer = pipe({
        "question": question,
        "context": context
    })
# anwer is a dict so answer["answer"] to get the actual answer.
    print("Answer:", answer["answer"])
    return answer["answer"]


def start_chat():
    print("Welcome to the QA System. Type 'exit' to quit.")
    while True:
        q = input("\nEnter question: ")
        if q.lower() == "exit":
            break
        ask_question(q)


if __name__ == "__main__":
    start_chat()
