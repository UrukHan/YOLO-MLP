from src import MLP
from fastapi import FastAPI

model = MLP.MODEL()

# Determining how queries work
app = FastAPI()

# Health check
@app.get("/health")
def health_check():
    '''health check'''
    return {"code": 200, "status": "OK"}


# train MLP network
@app.get("/train")
def train_MLP():
    '''train'''
    model.train()
    return {"code": 200, "status": "OK"}

# predict MLP network
@app.get("/predict")
def predict_MLP():
    '''predict'''
    return model.predict()




