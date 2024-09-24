# import libraries
import io
import os
import uvicorn
from openai import OpenAI
from pinecone import Pinecone
from autogen import ConversableAgent, register_function
from starlette.responses import FileResponse
from fastapi import FastAPI, File, UploadFile

# Load keys
OPENAI_API_KEY = "YOUR OPENAI API KEY"
PINECONE_API_KEY = 'd72b82f0-748c-4b64-9ba1-424e87aa5c36'

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Other constants
indexName = 'sarvam-assignment'
prompt = '''You are a helpful teaching assistant. Use the supplied tools to assist the user.
Add "TERMINATE" in the end when you have responded to the users query.
Example: 
    User: Hello
    Assistant: Hi! How can I help you today? TERMINATE'''
embedding_model = "text-embedding-3-small"
llmmodel = "gpt-4o-mini"
asrmodel = "whisper-1"
tts_model = "tts-1"
tts_voice = "shimmer"
k = 3

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(indexName)

client = OpenAI()

assistant = ConversableAgent(
    name="Assistant",
    system_message=prompt,
    llm_config={"config_list": [{"model": llmmodel, "api_key": OPENAI_API_KEY}]},
    human_input_mode="NEVER",
    max_consecutive_auto_reply=2,
)

user_proxy = ConversableAgent(
    name="User",
    llm_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
)

app = FastAPI()


def embed(text):
    response = client.embeddings.create(input = text, model=embedding_model)
    emb = response.data[0].embedding
    return emb

def retrieve_texts(query:str) -> str:
    embedding = embed(query)
    results = index.query(vector=embedding, include_metadata=True, top_k=k)
    info = ""
    for match in results['matches']:
        info += f"retrieved text_id : {match['id']} \n"
        info += match['metadata']['text'] + "\n \n"
    return info

register_function(
    retrieve_texts,
    caller=assistant,  
    executor=user_proxy,
    name="retrieve_texts",
    # could describe more precisely like theres only one chapter named sound of science book for school students in the DB
    description="Get the relevant texts from the vector database for the given query", )

def process_query(query):
    results = user_proxy.initiate_chat(assistant, 
                                    message=query,
                                    summary_method="last_msg",
                                    max_turns=2,)
    return results.summary

def asr_en(file):
    response = client.audio.translations.create(
        model= asrmodel,
        file=file,
        )
    translation_text = response.text
    return translation_text

def tts(text):
    response = client.audio.speech.create(
    model=tts_model,
    voice=tts_voice,
    input= text
    )
    file_name = "audio.mp3"
    response.write_to_file(file_name)
    return file_name

@app.post("/chat")
def text_query(query: str):
    answers = process_query(query)
    return answers
    
@app.post("/talk")
async def audio_query_audio(audio_file: UploadFile = File(...)):
    file_contents = await audio_file.read()
    file_in_memory = io.BytesIO(file_contents)
    file_in_memory.name = audio_file.filename
    question = asr_en(file_in_memory)
    answers = process_query(question)
    file_name = tts(answers)
    return FileResponse(file_name, media_type="audio/mpeg", filename="output.mp3")

if __name__ == "__main__":
    uvicorn.run(app)