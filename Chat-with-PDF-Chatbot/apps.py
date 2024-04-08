from flask import Flask, render_template, request, session
import os
import base64
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, AutoConfig
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS
from ingest import ingest_documents

app = Flask(__name__)
app.secret_key = '12345'
device = torch.device('cpu')

# Placeholder for the chat history
app.config['SESSION_CHAT_HISTORY'] = []


persist_directory = "db"




def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
                texts = text_splitter.split_documents(documents)
                # create embeddings here
                embeddings = SentenceTransformerEmbeddings(model_name="custom_model")
                # create vector store here
                db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
                db.persist()
                db = None


custom_model="models/dynamic_memory.pth"

def llm_pipeline():
    # Load your custom model

    torch.save(custom_model.state_dict(), "models/Dynamic_memory.pth")
    custom_model.load_state_dict(torch.load("models/dynamic_memory.pth", map_location = torch.device('cpu')))
    print(type(custom_model))

    def simple_tokenizer(text):
        # Split the input text into tokens using space as a delimiter
        return text.split()

    # Define your pipeline with the loaded custom model
    pipe = pipeline(
        'text2text-generation',
        model=custom_model,  # Use the loaded custom model
        tokenizer=simple_tokenizer,  # Use the simple tokenizer
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        device=device
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="custom_model")
    db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        if 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file:
                file_details = {
                    "Filename": uploaded_file.filename,
                    "File size": len(uploaded_file.read())
                }
                filepath = "docs/" + uploaded_file.filename
                uploaded_file.seek(0)
                with open(filepath, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())

                with open(filepath, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                pdf_view = pdf_display

                with app.test_request_context():
                    data_ingestion()

                return render_template('index.html', file_details=file_details, pdf_view=pdf_view, chat_history=[], user_input="")

        elif 'user_input' in request.form:
            user_input = request.form['user_input']

            if user_input:
                answer = process_answer({'query': user_input})
                chat_history = session.get('chat_history', [])
                chat_history.append(f"User: {user_input}")
                chat_history.append(f"AI: {answer}")
                session['chat_history'] = chat_history

    return render_template('index.html', file_details=None, pdf_view="", chat_history=session.get('chat_history', []), user_input="")

if __name__ == "__main__":
    app.run(debug=True)


