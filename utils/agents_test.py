# from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from operator import itemgetter
from dataframe import transactions_master_df, inventory_master_df, storage_cost_df, transfer_cost_df
from langchain_core.runnables import RunnableLambda

import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', None)
if not GOOGLE_API_KEY:
    print("NO KEY")

# We Add explicit instructions and an example of the expected output format.
data_dictionary_prefix = """
You are a supply chain data analyst expert.
You have access to a list of pandas dataframes.
The dataframes are available in the python REPL as `df1`, `df2`, `df3`, and `df4`.

Here is the mapping of the dataframe names to their content:
- **df1**: `transactions_master_df`. A log of all inbound and outbound material movements, enriched with material details.
- **df2**: `inventory_master_df`. A monthly snapshot of all inventory batches, enriched with material details.
- **df3**: `storage_cost_df`. A lookup table for the cost of storing 1 MT of material per day in a specific plant.
- **df4**: `transfer_cost_df`. A lookup table for the cost of transferring one container via a specific transport mode.

When you are asked a question, you MUST refer to the dataframes by their variable names `df1`, `df2`, etc. in your python code.

Example: To find the total outbound quantity, your python code should look like:
`print(df1[df1['TRANSACTION_TYPE'] == 'OUTBOUND']['NET_QUANTITY_MT'].sum())`

Do not try to redefine the dataframes. They are already loaded for you.

Here is the detailed data dictionary for each columns:

- TRANSACTION_DATE: The date the transaction occurred.
- PLANT_NAME: Name of the plant/warehouse.
- MATERIAL_NAME: The specific product being transacted.
- NET_QUANTITY_MT: The quantity of the transaction in Metric Tons.
- TRANSACTION_TYPE: 'INBOUND' for incoming materials, 'OUTBOUND' for outgoing materials.
- POLYMER_TYPE: The general type of the material.
- BALANCE_AS_OF_DATE: The date the inventory snapshot was taken.
- UNRESTRICTED_STOCK: Total quantity that is physically available.
- STOCK_SELL_VALUE: The total sell value of that specific inventory record.
- PLANT_NAME: The plant name.
- STORAGE_COST_PER_MT_DAY: The cost to store one metric ton of material for one day.
- MODE_OF_TRANSPORT: How the material was shipped.
- TRANSFER_COST_PER_CONTAINER: The cost to transfer one container.

When answering a question, you MUST follow this format:
Thought: The thought process behind the action.
Action: The tool to use, which is `python_repl_ast`.
Action Input: The Python code to execute.

Example:
Thought: I need to see the first few rows of the transactions dataframe to understand its structure.
Action: python_repl_ast
Action Input: print(transactions_master_df.head())

Now, begin.
"""

# llm = OpenAI(temperature=0)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0,
    verbose=True,                 # show prompt + response logs
)

# Give the agent a LIST of all your clean dataframes
agent = create_pandas_dataframe_agent(
    llm,
    [transactions_master_df, inventory_master_df, storage_cost_df, transfer_cost_df],
    prefix=data_dictionary_prefix,
    allow_dangerous_code=True,
    verbose=True,
    # agent_executor_kwargs=dict(
    #     handle_parsing_errors=True
    # )
)

# 2. The Formatting Chain (the "humanizer")
humanizer_prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Your job is to rephrase a raw data answer into a natural, human-readable sentence.
    
    Original Question: {question}
    Raw Data Answer: {raw_answer}
    
    Please formulate a friendly and clear response based on the data.
    """
)

# 3. Create a simple chain that combines the agent with the humanizer
def process_question(inputs):
    question = inputs["question"]
    
    # Get raw answer from agent
    raw_answer = agent.invoke({"input": question})["output"]
    
    # Format the answer using the humanizer
    formatted_response = humanizer_chain.invoke({
        "question": question,
        "raw_answer": raw_answer
    })
    
    return formatted_response

# The formatting chain
humanizer_chain = humanizer_prompt | llm | StrOutputParser()

# Create the full chain using RunnableLambda
full_chain = RunnableLambda(process_question)

# 4. Invoke the full chain
config = {"configurable": {"thread_id": "1"}}
question = "What date is the last OUTBOUND sale on material MAT-0354?"
final_response = full_chain.invoke({"question": question}, config)

print(final_response)