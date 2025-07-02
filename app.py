
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from utils.data_loader import DataLoader
from utils.text_processor import TextProcessor

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

loader = DataLoader()
processor = TextProcessor()

@app.get("/load-dataset")
def load_dataset():
    df = loader.load_data()
    info = loader.get_dataset_info(df)
    return {"status": "success", "info": info}

@app.post("/clean-text")
async def clean_text(request: Request):
    body = await request.json()
    text = body.get("text", "")
    return {"cleaned": processor.clean_text(text)}
