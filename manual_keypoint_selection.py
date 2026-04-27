import cv2
from pathlib import Path
import numpy as np
import argparse

BASE_DIR = Path(__file__).resolve().parent



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_video', help='Video file name is required. All other files names are assumed to be based off of input_video file name. Videos are assumed to be in data/videos/')

    args = parser.parse_args()
    VIDEO_PATH = BASE_DIR / "data" / "videos" / f"{args.input_video}.mp4"
    OUTPUT_PATH = BASE_DIR / "data" / f"{args.input_video}_query_txy.npy"


    query_points = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            query_points.append(np.array([0, x, y]))
            cv2.imshow('image', img)


    cap = cv2.VideoCapture(VIDEO_PATH)
    _, img = cap.read()
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(query_points) == 0:
        return


    query_points = np.array([np.array(query_points)])

    print(query_points.shape)

    np.save(OUTPUT_PATH, query_points)


if __name__=="__main__":
    main()