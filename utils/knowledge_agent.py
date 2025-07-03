from langchain.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings


from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = "AIzaSyBkwtK1jqLRsrb2b0CMSXg2LTBpNGD8-As" #os.environ.get('GOOGLE_API_KEY', None)
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing. Please set it in your environment variables.")


# --- 1. Load & Split Documents ---

loader = TextLoader("utils/data/xxon_doc.md", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# --- 2. Embeddings & Vector Store ---

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(chunks, embedding_model)
retriever = vectordb.as_retriever()

# --- 3. LLM Setup ---

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0,
    verbose=True,
)

# --- 4. Prompt Templates ---

system_prompt = """
You are a helpful assistant. Use only the provided context to answer the question. 
If you don't know the answer, say you don't know.
"""

human_prompt = """
Context:
{context}

Question:
{question}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt),
])

humanizer_prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant that rewrites answers into clear, concise, and human-friendly language.

Original Question: {question}
Raw Answer: {raw_answer}

Please rewrite the answer accordingly."""
)

humanizer_chain = humanizer_prompt | llm | StrOutputParser()

# --- 5. Define the RAG pipeline function ---

def knowledge_agent(question: str, chat_history: list[tuple[str, str]] = None):
    """
    Retrieve relevant docs for the question and answer with LLM.
    Optionally rephrase answer into more natural language.
    
    chat_history currently not used, but you can extend to include it in prompt if needed.
    """
    # 1. Retrieve documents
    relevant_docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    
    # 2. Run prompt chain with context + question
    try:
        raw_answer = prompt_template | llm | StrOutputParser()
        raw_response = raw_answer.invoke({
            "context": context,
            "question": question
        })
        
        # 3. Optionally, humanize the raw answer
        formatted_response = humanizer_chain.invoke({
            "question": question,
            "raw_answer": raw_response
        })
        
        return formatted_response
    
    except Exception as e:
        print(f"Error during RAG agent run: {e}")
        return f"Sorry, I encountered an error: {e}"

# --- 6. RunnableLambda wrapper for possible pipeline composition ---
full_rag_chain = RunnableLambda(knowledge_agent)


if __name__ == "__main__":
    question = "What’s Plasticizers?"
    answer = knowledge_agent(question)
    print(answer)