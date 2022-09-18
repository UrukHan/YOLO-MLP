# Import libraries
from src import detect
from src import train
from src import slice_frame
import pandas as pd
from src.config import CONFIGURATION  as cfg
from src import video_parser
from fastapi import FastAPI

# Determining how queries work
app = FastAPI()

# Health check
@app.get("/health")
def health_check():
    '''health check'''
    return {"code": 200, "status": "OK"}

# Parse video
@app.get("/parse")
def parse():
    '''parse video'''
    data = pd.read_csv(cfg.DATA + cfg.VID_CSV)
    uuids = data.video_uuid.values.tolist()
    video_parser.parse(uuids[:cfg.MAX_VID], cfg.DATA + cfg.VID_PTH)
    return {"code": 200, "status": "OK"}

# Slice video
@app.get("/slice")
def slice():
    '''slice video'''
    slice_frame.parse()
    return {"code": 200, "status": "OK"}
    
# Train Yolo
@app.get("/train_yolo")
def train_yolo():
    '''train yolo'''
    train.run(data='data.yaml', imgsz=640, batch=16, epochs=50, weights='yolov5s.pt')
    return {"code": 200, "status": "OK"}

# Detect Yolo
@app.get("/detect_yolo")
def detect_yolo():
    '''detect yolo'''
    detect.run(save_crop = True, exist_ok = True, nosave = True, conf_thres = 0.8, min_size = 50)
    return {"code": 200, "status": "OK"}























