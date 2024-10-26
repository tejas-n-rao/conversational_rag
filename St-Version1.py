# TO-DO: Host on AWS/Azure


# Imports
import os
import uuid

from dotenv import load_dotenv

import pinecone
from pinecone import Pinecone, ServerlessSpec

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

import streamlit as st


load_dotenv()
spec = ServerlessSpec(cloud="aws", region="us-east-1") # spec instance
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY")) # pinecone object


# Accessing Indices on pinecone
index_name = "courses-ds" 
index = pc.Index(index_name)

# embedding model
embedding_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")) # load.env

# function to retrieve relevant chunks from pinecone vector database
def retrieve_from_pinecone(query, top_k=10): # top_k indicates how many chunks we want to retrive from database
    query_embedding = embedding_model.embed_query(query)
    results = index.query(vector=[query_embedding], top_k=top_k, include_metadata=True) # querying database for 10 relevant chunks
    
    relevant_chunks = [match["metadata"].get("text") for match in results["matches"]] 

    # Returning relevant chunks
    return relevant_chunks

# llm instance
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,  # Experiment with different Temperatures
)

# Variable storing how many messages (10) will be remembered (Human + Bot): For conversationalness
memory_messages = 10 # Update to change context lengths

# Converting query to a prompt that has context from the documents
def query_to_prompt(query, state_messages):
    relevant_chunks = retrieve_from_pinecone(query) # retrieveing relevant chunks to make context
    context = "\n\n -------------- \n\n".join(chunk for chunk in relevant_chunks)
    context = f"\n\nCONTEXT: {context}"
    
    # Play around with this.
    sys_message_template = (
        "Imagine you are a helpful assistant at Krea University who students come to clarify doubts regarding the university's Data Science curriculum. "
        "Answer the query only based on the context provided. Think step by step before providing a detailed answer. "
        "If you can't answer the query from the context, say that you can't. Do not hallucinate.\n"
        "The CONTEXT is as follows:\n{}"
    )
    formatted_sys_message = SystemMessage(sys_message_template.format(context))
    
    # Remove any existing SystemMessage from state_messages
    messages = [msg for msg in state_messages if not isinstance(msg, SystemMessage)]
    
    # Start with the last 'memory_messages' messages from the state
    if len(messages) > memory_messages - 1:
        messages = messages[-(memory_messages - 1):]  # Reserve one spot for SystemMessage
    
    # Insert the new SystemMessage at the beginning
    messages.insert(0, formatted_sys_message)
    
    # Add the current user query
    messages.append(HumanMessage(content=query))
    
    return {"messages": messages} # returning the prompt

# Function to call Rag Model
def call_rag_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    
    # Update the state with the new response
    state["messages"].append(response)
    
    # Keep only the last few messages in the state
    if len(state["messages"]) > memory_messages:
        state["messages"] = state["messages"][-1 * (memory_messages):] # Update this for more context
    
    # Maybe limit the context to say 1k tokens?? Maybe use an LLM or to summarise contexts?
    
    return {"messages": state["messages"]}

# New StageGraph from LangGraph for memory
workflow = StateGraph(state_schema=MessagesState)

workflow.add_edge(START, "model")
workflow.add_node("model", call_rag_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# uuid for thread_id for configurable 
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}
#print(thread_id) 

# Getting response from llm
def get_response(query, state):
    input_data = query_to_prompt(query, state["messages"])
    output = app.invoke(input_data, config)

    # updating state history
    state["messages"] = output["messages"] 

    llm_response = output["messages"][-1].content

    return llm_response, state

# Title of Webpage
st.title("RAG BOT") # Change this

# Captioning Page
st.caption("As this chatbot is built on an LLM, it can make mistakes. Contact discipline co-ordinator if doubts persist.")

# Creating section for example prompts
st.markdown("""-----------------------------------------------------""")

# Opening Questions.txt
with open ("Questions.txt", "r") as f:
    sample_questions = f.read().split("\n\n")

i,j,k = nprand.randint(0, len(sample_questions) -1 , 3) # Random Numbers to determine Questions

# Displaying Example Prompts
st.markdown("""###### Example Prompts""")
st.markdown(f"""
* {sample_questions[i]}
* {sample_questions[j]}
* {sample_questions[k]}
""")
st.markdown("""-----------------------------------------------------""")

# Chat window section
st.markdown("""### Chat Window""")

# tracking history of session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am a chatbot to help with all your Data Science curriculum related queries. How can I help you?"}] # Initial message

# To display Chat History:
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Initialising state
if "state_messages" not in st.session_state:
    st.session_state.state_messages = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am a chatbot to help with all your curriculum-related queries. How can I help you?"}
    ]


state = {"messages": st.session_state.state_messages}


# Secrets file for api keys 

# For Streaming: To be done later
def message_generator(message_content):
    for chunk in message_content.split(". "):  # Splitting by sentence
        yield chunk + ".\n"


query = st.chat_input("Enter your queries here: ")
if query is not None and query != "":
    st.session_state.messages.append({"role": "user", "content": query})

    # Initial input
    with st.chat_message("user"):
        st.write(query)

    # Getting Response from Conversational RAG
    llm_response, state = get_response(query, state)

    # Updating session history
    st.session_state.state_messages = state["messages"]
    st.session_state.messages.append({"role": "assistant", "content": llm_response})

    # Displaying chatbot output
    with st.chat_message("assistant"):
        st.write(llm_response)
        # For Streaming
        #st.write_stream(message_generator(llm_response))
