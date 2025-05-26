import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json

# Streamlit UI setup
st.set_page_config(page_title=" Gadget Support Chatbot")
st.title("Gadget FAQ Chatbot Using LLM)")

# Load Hugging Face model pipeline once
@st.cache_resource
def load_phi_model():
    model_name = "microsoft/phi-1_5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id
    )
    return pipe

# Load FAQ data and create FAISS vector store
@st.cache_resource
def load_vectorstore():
    with open("faq.json") as f:
        faq_data = json.load(f)
    text_blocks = [
        f"Q: {item['question']}\nA: {item['answer']}" for item in faq_data
    ]
    full_text = "\n\n".join(text_blocks)
    docs = [Document(page_content=full_text)]
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(split_docs, embeddings)

# Build prompt with conversation history (last 2 turns), FAQ context, and current question
def build_prompt(conversation_history, faq_context, current_question):
    # Keep last 2 exchanges only
    recent_history = conversation_history[-2:] if len(conversation_history) > 2 else conversation_history

    history_text = ""
    for q, a in recent_history:
        history_text += f"User: {q}\nAssistant: {a}\n"

    prompt = f"""
You are a helpful and precise customer support assistant. Use ONLY the information in the FAQs below to answer customer questions.

If the answer is not in the FAQs, respond: "I'm not sure about that."

FAQs:
{faq_context}

Conversation so far:
{history_text}

Now, answer the customer's question below using ONLY the FAQs.

Customer's question: {current_question}
Answer:"""
    return prompt

# Clean model output to remove trailing unrelated text
def clean_answer(generated_text):
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[-1].strip()
    else:
        answer = generated_text.strip()

    # Stop generation at common markers
    stop_tokens = ["User:", "Assistant:", "Q:", "Question:", "FAQs:"]
    for token in stop_tokens:
        if token in answer:
            answer = answer.split(token)[0].strip()
    return answer

# Initialize session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Load models & vectorstore
phi = load_phi_model()
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# User input
query = st.text_input("Ask a question about products, orders, or policies:")

if query:
    with st.spinner("Answering..."):
        # Retrieve relevant FAQ docs
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Build prompt with conversation history and FAQ context
        prompt = build_prompt(st.session_state.history, context, query)
        
        # Generate answer
        output = phi(prompt)[0]["generated_text"]
        answer = clean_answer(output)

        # Handle uncertain answers
        if not answer or answer.lower().startswith("i don't know") or "i'm not sure" in answer.lower():
            answer = "I'm not sure about that. You may contact our support team for more details."

        # Save turn in history
        st.session_state.history.append((query, answer))

        # Display answer and retrieved FAQ snippets
        st.markdown("### Answer")
        st.success(answer)

        with st.expander(" Retrieved FAQ Snippets"):
            for doc in docs:
                st.markdown(
                    f"<div style='background-color:#eef; padding:10px; border-radius:6px; margin-bottom:10px;'>{doc.page_content}</div>",
                    unsafe_allow_html=True
                )

# Optional: Button to clear conversation history
if st.button("Clear Conversation"):
    st.session_state.history = []
    st.experimental_rerun()
