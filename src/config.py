from pydantic import BaseSettings

class Configuration(BaseSettings):

    PROJECT_NAME: str = "Child class"
    DATA: str = 'data/'  # Директория храннения данных
    VID_PTH: str = 'vid_pth'  # Папка для загрузки видео
    MAX_VID: int = 50   # Максимальное количество видео для скачивания
    PARSE_FRAMES: str = 'parse_frames/'   # Директория хранения кадров отобранных для детекции YOLO сетью
    N_FRAME: int = 50   # Каждый n-ый на проверку
    DICT_CLASS: str = 'dict_class'   # Словарь классификацированных видео
    VID_CSV: str = 'videos.csv'   # Файл информации о видео
    MLP_TRAIN_DATA: str = 'mlp_data'   # Директория изображений лиц    (на конце 1 - взрослый, 2 - ребенок)
    FACES: str = 'runs/detect/exp/crops/face'
    MODEL: str = 'data/mlp_model'
    CHILD: str = 'child/'
    ADULT: str = 'adult/'
    PLT: str = 'mlp_train.png'
    BATCH_SIZE_MLP: int = 8
    EPOHS_MLP: int = 1024
    IMAGE_SIZE_MLP: int = 64
    PATCH_SIZE_MLP: int = 8
    PB_MLP: int = 128 # Projection units
    DC_MLP: int = 512 # Token-mixing units
    DS_MLP: int = 64 # Channel-mixing units
    NB_MLP: int = 8 # Num of mlp blocks
    OUT_MLP: int = 2 # Count outputs (classes if classify)
    BUFFER_SIZE: int = 8 # Count outputs (classes if classify)
    PRED_MLP_TRH: float = 0.9 # коэфицент уверенности модели для классификации детей

    class Config:
        case_sensitive = True

CONFIGURATION = Configuration()
