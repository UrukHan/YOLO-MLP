
import cv2
from src.config import CONFIGURATION  as cfg
import glob
import pickle

def parse():
    video_paths = sorted(glob.glob(cfg.DATA + cfg.VID_PTH + '/*.mp4'))

    dict_class = {}
    for i in video_paths:
        dict_class[i.split('.')[0].split('\\')[-1]] = -1

    with open(cfg.DATA + cfg.DICT_CLASS, 'wb') as f:
        pickle.dump(dict_class, f)

    print('Preapare videos')
    for i in video_paths:
        vid = cv2.VideoCapture(i)
        success, image = vid.read()
        n = 0
        while(vid.isOpened()):
            ret, frame = vid.read()
            n += 1
            if ret:
                if n % cfg.N_FRAME == 0:
                    cv2.imwrite(cfg.DATA + cfg.PARSE_FRAMES + i.split('.')[0].split('\\')[-1] + '_' + str(n) + ".jpg", frame)
            else:
                break
        vid.release()
        cv2.destroyAllWindows()