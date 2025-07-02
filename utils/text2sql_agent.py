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


def text2sql_agent(question: str, chat_history: list = None):
    """
    Convert natural language question to SQL query for warehouse and inventory data.
    This function generates SQL queries based on the user's question.
    """
    if chat_history is None:
        chat_history = []
    
    prompt_system = """
    You are an expert SQL query generator for warehouse and inventory management systems.
    
    Your task is to:
    Generate accurate SQL queries based on natural language questions about warehouse and inventory data.
    """

    prompt_user = """
    INSTRUCTIONS — Generate SQL Query Based on Natural Language Question

      ======================================

      You have access to the following database tables and their schemas:

      1. TRANSACTIONS_MASTER - Contains all inventory transactions
         - Common columns: transaction_id, material_code, plant_code, transaction_type, quantity, date, etc.

      2. INVENTORY_MASTER - Contains current inventory levels
         - Common columns: material_code, plant_code, stock_quantity, location, etc.

      3. STORAGE_COST - Contains storage cost information
         - Common columns: location, material_code, cost_per_unit, etc.

      4. TRANSFER_COST - Contains transfer cost information
         - Common columns: from_location, to_location, material_code, cost_per_unit, etc.

      Based on the natural language question, generate a SQL query that:
      1. Uses appropriate table joins when needed
      2. Includes proper WHERE clauses for filtering
      3. Uses correct aggregate functions (SUM, COUNT, AVG, etc.) when required
      4. Handles date ranges and comparisons appropriately
      5. Returns relevant columns for the question asked

      Guidelines:
      - Use standard SQL syntax
      - Include table aliases for clarity
      - Use proper date formatting
      - Handle case-insensitive string comparisons when appropriate
      - Return only the SQL query without additional explanation
      - DO NOT include any comments in the SQL query (no -- comments or /* */ comments)
      - Generate clean, executable SQL without any explanatory text or comments
      - The output should be pure SQL that can be executed directly
      - Do not wrap the SQL in code blocks or markdown formatting
      - Return only the raw SQL statement

      Question: {question}

      SQL Query:
    """
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_system),
        ("human", prompt_user)
    ])

    # Create the chain
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({
            "question": question
        })
        
        sql = result.strip()
        
        print(f"sql: {sql}")
        return sql
        
    except Exception as e:
        raise Exception(f"Error in text2sql : {e}")
        

# Test the text2sql function
if __name__ == "__main__":
    print("Testing text2sql...")
    test_result = text2sql_agent(
        question="What are the total inbound transactions for each plant?",
        chat_history=[]
    )
    
