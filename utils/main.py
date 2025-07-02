from text2sql_agent import text2sql_agent
from intent import intent_router

test_questions = [
    "What are the total inbound transactions for each plant?",
    "Show me the current inventory levels by material code",
    "What is the average storage cost per unit for each location?",
    "Which materials have the highest stock quantity?",
    "What are the transfer costs between different locations?",
    "Show me all transactions in the last month",
    "What is the total quantity of material X in plant Y?"
]

for i, question in enumerate(test_questions, 1):
    print(f"\n{i}. Question: {question}")
    print("-" * 50)
    
    try:
        
        intent = intent_router(question)
        sql_result = text2sql_agent(question)
        print(f"Generated SQL:\n{sql_result}")
        print("✅ SUCCESS")
        

        
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print("-" * 50)