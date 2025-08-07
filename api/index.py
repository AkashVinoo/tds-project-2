from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from app.agent import process_question_file

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Vercel is working and FastAPI is live!"}

@app.post("/api/")
async def analyze(question_file: UploadFile = File(...), attachments: list[UploadFile] = File(default=[])):
    content = await question_file.read()
    answers = await process_question_file(content.decode(), attachments)
    return JSONResponse(content=answers)
