from fastapi import FastAPI
from pydantic import BaseModel
# Docker
from app.glue_cola.model import BertModel
from app.glue_cola.model import __version__ as model_version
# No Docker
# from glue_cola.model import BertModel
# from glue_cola.model import __version__ as model_version

app = FastAPI()



class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    veredict: str


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    Model = BertModel()
    veredict = Model.predict(payload.text)
    return {"veredict": veredict}