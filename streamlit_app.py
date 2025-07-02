import streamlit as st
import os
import glob
from datetime import datetime
from dotenv import load_dotenv
from utils.supply_chain_agent import SupplyChainAgent
from utils.logger import log_streamlit_event

load_dotenv()

def main():
    st.set_page_config(
        page_title="Supply Chain Analytics Assistant", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🏭 Supply Chain Analytics Assistant")
    st.markdown("Ask questions about inventory, transactions, costs, and get insights with visualizations")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Available Data")
        st.markdown("""
        **Datasets Available:**
        - 📦 Transaction History (Inbound/Outbound)
        - 📋 Inventory Snapshots
        - 🏪 Material Master Data
        - 💰 Storage & Transfer Costs
        """)
        
        st.header("💡 Sample Questions")
        sample_questions = [
            "Show inventory levels for all materials",
            "What are the top 5 materials by outbound volume?",
            "Plot monthly transaction trends",
            "Which plants have the highest storage costs?",
            "Show me materials with low stock levels"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{hash(question)}"):
                st.session_state.user_input = question
        
        st.header("🔧 Settings")
        if st.button("🗑️ Clear Chat History"):
            log_streamlit_event("clear_chat_history")
            st.session_state.messages = []
            if "agent" in st.session_state:
                # Clear both LangChain and LangGraph memory
                st.session_state.agent.clear_memory(st.session_state.get("thread_id", "default"))
            st.rerun()
        
        # Debug Panel
        with st.expander("🔍 Debug & Logs"):
            st.subheader("Log Files")
            
            # Show available log files
            log_files = glob.glob("logs/*.log")
            if log_files:
                selected_log = st.selectbox("Select log file:", log_files)
                
                if st.button("📖 View Logs"):
                    try:
                        with open(selected_log, 'r') as f:
                            logs = f.read()
                        st.text_area("Log Contents:", logs, height=300)
                    except Exception as e:
                        st.error(f"Error reading log file: {e}")
            else:
                st.info("No log files found. Logs will appear after using the agent.")
            
            # Real-time agent status
            st.subheader("Agent Status")
            if "agent" in st.session_state:
                st.success("✅ Agent initialized")
                try:
                    table_names = st.session_state.agent.db.get_usable_table_names()
                    st.info(f"Database tables available: {len(table_names)}")
                    st.text(f"Tables: {', '.join(table_names)}")
                except:
                    st.info("SQL database connected")
                
                # Show memory status
                try:
                    history = st.session_state.agent.get_conversation_history(st.session_state.thread_id)
                    st.info(f"Conversation messages in memory: {len(history)}")
                except:
                    st.warning("Could not retrieve conversation history")
            else:
                st.warning("⚠️ Agent not initialized")
            
            # Session info
            st.subheader("Session Info")
            st.json({
                "messages_count": len(st.session_state.get("messages", [])),
                "thread_id": st.session_state.get("thread_id", "default"),
                "timestamp": datetime.now().isoformat()
            })
            
            # Memory info
            st.subheader("Memory Status")
            if "agent" in st.session_state:
                st.info("✅ LangGraph SQLite persistence enabled")
                st.info("✅ LangChain window memory (5 exchanges)")
            else:
                st.warning("⚠️ Memory not initialized")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "thread_id" not in st.session_state:
        import uuid
        st.session_state.thread_id = str(uuid.uuid4())
    
    if "agent" not in st.session_state:
        with st.spinner("🔄 Loading supply chain data and initializing agent..."):
            st.session_state.agent = SupplyChainAgent()
    
    # Display chat messages
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["type"] == "text":
                st.markdown(message["content"])
            elif message["type"] == "plotly":
                st.plotly_chart(message["content"], use_container_width=True, key=f"plotly_{idx}_{id(message)}")
            elif message["type"] == "dataframe":
                if message["content"] is not None and not message["content"].empty:
                    st.dataframe(message["content"], use_container_width=True)
            elif message["type"] == "sql_query":
                if message["content"] and message["content"] != "No SQL query captured":
                    with st.expander("🔍 SQL Query Used"):
                        st.code(message["content"], language="sql")
    
    # Chat input
    user_input = st.chat_input("Ask me about your supply chain data...")
    
    # Handle sample question clicks
    if "user_input" in st.session_state and st.session_state.user_input:
        user_input = st.session_state.user_input
        st.session_state.user_input = ""
    
    if user_input:
        log_streamlit_event("user_query", {"query": user_input[:100]})
        
        # Add user message
        st.session_state.messages.append({"role": "user", "type": "text", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your data..."):
                try:
                    response = st.session_state.agent.process_query(
                        user_input, 
                        thread_id=st.session_state.thread_id
                    )
                    
                    # Handle different response types
                    if response["type"] == "text":
                        st.markdown(response["content"])
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "type": "text", 
                            "content": response["content"]
                        })
                        log_streamlit_event("response_text", {"length": len(response["content"])})
                    
                    elif response["type"] == "text_with_chart":
                        st.markdown(response["text"])
                        st.plotly_chart(response["chart"], use_container_width=True, key=f"chart_{st.session_state.thread_id}_{len(st.session_state.messages)}")
                        st.session_state.messages.extend([
                            {"role": "assistant", "type": "text", "content": response["text"]},
                            {"role": "assistant", "type": "plotly", "content": response["chart"]}
                        ])
                        log_streamlit_event("response_chart", {"text_length": len(response["text"])})
                    
                    elif response["type"] == "text_with_dataframe":
                        st.markdown(response["text"])
                        st.dataframe(response["dataframe"], use_container_width=True)
                        st.session_state.messages.extend([
                            {"role": "assistant", "type": "text", "content": response["text"]},
                            {"role": "assistant", "type": "dataframe", "content": response["dataframe"]}
                        ])
                        log_streamlit_event("response_dataframe", {"text_length": len(response["text"])})
                    
                    elif response["type"] == "text_with_sql_and_dataframe":
                        # Display natural language answer
                        st.markdown(response["text"])
                        
                        # Display DataFrame if available
                        if response["dataframe"] is not None and not response["dataframe"].empty:
                            st.subheader("📊 Raw Data Results")
                            st.dataframe(response["dataframe"], use_container_width=True)
                        
                        # Display SQL query in expandable section
                        if response["sql_query"] and response["sql_query"] != "No SQL query captured":
                            with st.expander("🔍 SQL Query Used"):
                                st.code(response["sql_query"], language="sql")
                        
                        # Store in message history
                        st.session_state.messages.extend([
                            {"role": "assistant", "type": "text", "content": response["text"]},
                            {"role": "assistant", "type": "dataframe", "content": response["dataframe"]},
                            {"role": "assistant", "type": "sql_query", "content": response["sql_query"]}
                        ])
                        log_streamlit_event("response_sql_and_dataframe", {
                            "text_length": len(response["text"]),
                            "dataframe_shape": response["dataframe"].shape if response["dataframe"] is not None else None,
                            "sql_length": len(response["sql_query"]) if response["sql_query"] else 0
                        })
                    
                    elif response["type"] == "error":
                        st.error(f"❌ {response['content']}")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "type": "text", 
                            "content": f"❌ {response['content']}"
                        })
                        log_streamlit_event("response_error", {"error": response['content']})
                        
                except Exception as e:
                    error_msg = f"❌ An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "type": "text", 
                        "content": error_msg
                    })
                    log_streamlit_event("streamlit_error", {"error": str(e)})

if __name__ == "__main__":
    main()