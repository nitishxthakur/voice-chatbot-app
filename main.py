from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import openai, os, uuid

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Serve frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.post("/chat/")
async def chat_with_voice(file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4()}.mp3"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    try:
        transcription = openai.Audio.transcribe("whisper-1", file=open(temp_filename, "rb"))
        user_input = transcription["text"]

        chat_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a personal voice-based ChatGPT."},
                {"role": "user", "content": user_input}
            ]
        )
        bot_reply = chat_response['choices'][0]['message']['content']

        audio_response = openai.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=bot_reply
        )

        audio_filename = f"reply_{uuid.uuid4()}.mp3"
        with open(f"static/{audio_filename}", "wb") as out:
            out.write(audio_response.content)

        return JSONResponse(content={
            "transcription": user_input,
            "reply_text": bot_reply,
            "voice_url": f"/{audio_filename}"
        })

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
