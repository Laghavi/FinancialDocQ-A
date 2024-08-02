from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import gradio as gr
from collections import deque
import time

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

chain = None
request_times = deque(maxlen=30)


def is_rate_limited():
    current_time = time.time()
    if len(request_times) == 30 and current_time - request_times[0] < 60:
        return True
    request_times.append(current_time)
    return False


def process_file(file):
    global chain

    if is_rate_limited():
        return "Rate limit reached. Please wait a moment before processing a file."

    if file.name.endswith('.pdf'):
        loader = PyPDFLoader(file.name)
    else:
        loader = TextLoader(file.name)

    loaded_doc = loader.load()
    rec_text_split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = rec_text_split.split_documents(loaded_doc)

    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    message = """
           You are an expert financial analyst with deep knowledge of accounting principles and financial reporting standards. Your task is to provide accurate and concise answers to questions about financial statements, using only the information provided in the given context. Follow these guidelines:

        1. Provide direct answers to the specific questions asked.
        2. Use exact figures and data from the financial statements when relevant.
        3. Do not include additional context, background information, or analysis unless explicitly requested.
        4. If the question cannot be fully answered based on the given context, state this briefly.
        5. Base your answers solely on the information provided in the context. Do not make assumptions or add information from external sources.

        Remember, keep your responses brief and to the point, addressing only what was asked.

        Question: {question}

        Context:
        {context}
    """
    prompt = ChatPromptTemplate.from_messages([("human", message)])

    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model

    return "File processed successfully. You can now ask questions about the document."


def chat_with_doc(message, history):
    global chain
    if chain is None:
        return "", history + [("You", message), ("Bot", "Please upload a document first.")]

    if is_rate_limited():
        return "", history + [("You", message),
                              ("System", "Rate limit reached. Please wait a moment before trying again.")]

    response = chain.invoke(message)
    print(response)
    history.append((message, None))
    history.append((None, response.content))
    return "", history


custom_css = """
.user_msg {
    background-color: #2b313e;
}
.bot_msg {
    background-color: #3c4454;
}
"""
# Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# Document Question-Answering System")

    with gr.Row():
        with gr.Column(scale=4):
            file_upload = gr.File(label="Upload Document (PDF or TXT)")
            process_button = gr.Button("Process", size="sm")

    output_text = gr.Textbox(label="Processing Output", interactive=False)

    chatbot = gr.Chatbot(label="Conversation", height=500, elem_classes=["user_msg", "bot_msg"])

    with gr.Row():
        msg = gr.Textbox(label="Ask a question about the document", show_label=False,
                         placeholder="Type your question here...")
        submit_btn = gr.Button("Ask", size="sm")

    clear = gr.Button("Clear Chat", size="sm")

    process_button.click(process_file, inputs=[file_upload], outputs=[output_text])
    msg.submit(chat_with_doc, inputs=[msg, chatbot], outputs=[msg, chatbot])
    submit_btn.click(chat_with_doc, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear.click(lambda: (None, None), outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch()