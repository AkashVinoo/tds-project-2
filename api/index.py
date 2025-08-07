from fastapi import FastAPI, UploadFile, File
from app.agent import analyze_file  # Adjust import path if needed

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI app deployed on Vercel!"}

@app.post("/api/")
async def analyze(uploaded_file: UploadFile = File(...)):
    return analyze_file(uploaded_file)
