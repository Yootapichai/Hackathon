from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dataframe import transactions_master_df, inventory_master_df, storage_cost_df, transfer_cost_df
from langchain_core.runnables import RunnableLambda

import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', None)
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing. Please set it in your environment variables.")

# We Add explicit instructions and an example of the expected output format.
data_dictionary_prefix = """

"""

# llm = OpenAI(temperature=0)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0,
    verbose=True,          
)

# Give the agent a LIST of all your clean dataframes
agent = create_pandas_dataframe_agent(
    llm,
    [transactions_master_df, inventory_master_df, storage_cost_df, transfer_cost_df],
    prefix=data_dictionary_prefix,
    allow_dangerous_code=True,
    verbose=True,
    agent_executor_kwargs=dict(
        handle_parsing_errors=True
    )
)

# 2. The Formatting Chain (the "humanizer")
humanizer_prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Your job is to rephrase a raw data answer into a natural, human-readable sentence.
    
    Original Question: {question}
    Raw Data Answer: {raw_answer}
    
    Please answer the question in a clear, concise, and human-readable format. Focus on providing the key information requested.
    """
)

# 3. Create a simple chain that combines the agent with the humanizer
def process_question(inputs):
    question = inputs["question"]
    
    try:
        # Get raw answer from agent
        raw_answer = agent.invoke({"input": question})["output"]
        
        formatted_response = humanizer_chain.invoke({
            "question": question,
            "raw_answer": raw_answer
        })
        
        return formatted_response
    
    except Exception as e:
        print(f"Error processing question: {e}")
        return f"I encountered an error while processing your question: {str(e)}"

humanizer_chain = humanizer_prompt | llm | StrOutputParser()

full_chain = RunnableLambda(process_question)


def intent_router(question: str, chat_history: list = None):
    """
    Route the question to the appropriate intent handler.
    This function classifies the user's question into different intents.
    """
    if chat_history is None:
        chat_history = []
    
    prompt_system = """
    You are a helpful assistant with expertise in warehousing and inventory management.
    
    Your task is to:

    Identify the intent of the user's question.
    """

    prompt_user = """
    INSTRUCTIONS — Rewrite and Classify Input Question Based on Context and Chat History

      ======================================

      Step 1: Rewrite the Input Question

      **ALWAYS use the provided context and chat history to rewrite the input question.**

      Your goal is to make the question clear, self-contained, and unambiguous — even if it's a follow-up to previous questions.
      ✘ DO NOT add assumptions or new terms not supported by the input or context or chat history.
      ✘ Do NOT rewrite the question to add the word `How` in the input question unless originally specified in the input question.

      ---

      1.1 Handle Follow-up Questions
      If the question is a follow-up (e.g., uses `how about`, `what about`, `split by`, `group`, or vague pronouns like `this`, `those`, `it`), ALWAYS refer to the **most recent relevant Q&A** in chat history.

      ✔ Inherit only what is clearly mentioned or inferable from prior context and/or chat history:
      - **Timeframe** (e.g., `last week`, `last month`, `all time`, etc.)
      - **Entities** (e.g., PLANT_NAME, MATERIAL_NAME)
      - **Comparison logic** (e.g., top 3, split by, trend)

      ✘ DO NOT assign to `general_question` just because the follow-up is short/unclear — always try to complete it using the previous Q&A.

      **Summary:** For every follow-up, exhaustively use all available chat history and context to resolve the question. Only assign to `general_question` if, after all attempts, the question is still completely ambiguous.

      When rewriting the question, ALWAYS apply the above rules to resolve vague references or missing details using the most recent relevant context and/or chat history.
      This ensures follow-ups like (if that was the last analysis)

      ---

      1.2 Clarify the Question
      - Use context and history only to **resolve**, not expand.
      - Pronouns like `it`, `this`, `that`, as well as ordinal references like `first`, `second`, `last`, etc., must be tied to a specific, prior reference.
      - If no metric is specified and none is clearly inferable, leave it general — do NOT assume ads/campaigns/platforms.

      ---

      1.3 If the input is already clear and self-contained, keep it as-is.

      1.4 If the question remains vague even after checking and combining with context and chat history, assign to `general_question`.

      ---

      Step 2: Answer to Relevant Intent(s)

      Classify the rewritten question into one :

      - `knowledge_question`: the question is about warehousing and product details, such as how to manage inventory, product specifications, or general warehousing practices. 
    
      - `inventory_question`: the question is about specific inventory transactions, such as inbound or outbound transactions, stock levels, or inventory movements.

      - `general_question`: the question is too vague or general, and cannot be classified into any specific intent. This includes questions that do not relate to warehousing or inventory management, or are too broad to be answered specifically.
     
    YOU MUST answer only the question intent. must have one of the following values:
        - `knowledge_question`
        - `inventory_question`
        - `general_question`
      ======================================

      Input Question: {question}

      Answer:
    """
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_system),
        ("human", prompt_user)
    ])

    # Create the chain
    intent_chain = prompt | llm | StrOutputParser()
    
    try:
        result = intent_chain.invoke({
            "question": question
        })
        
        intent = result.strip()
        
        print(f"Intent classified as: {intent}")
        return intent
        
    except Exception as e:
        print(f"Error in intent classification: {e}")
        return "general_question"

# Test the intent router
if __name__ == "__main__":
    print("Testing intent router...")
    test_result = intent_router(
        question="What date is the last OUTBOUND sale on material MAT-0354?",
        chat_history=[]
    )
    print(f"Intent classification result: {test_result}")
    
