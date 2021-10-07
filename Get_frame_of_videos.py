print("Hey, le code tourne !!")

import cv2
import os
import numpy as np
import subprocess
import glob

from subprocess import Popen, PIPE
from shlex import split

# Get Frame of a video
def getFrame(vidcap, sec, count, seqID, new_image_dir):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasFrames, image = vidcap.read()
    # print("\nType of hasFrames : ",type(hasFrames), " / hasFrames : " ,hasFrames)
    if hasFrames:
        height, width = image.shape[0], image.shape[1]     
        # resize images to width of new_width
        new_width = 400
        print("new image dir in getFrame : ", new_image_dir)
        image = cv2.resize(image,(new_width,np.int(height/(width/new_width))),interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(new_image_dir, seqID+ "_" + format(count, '04d') + ".jpg"), image)
        return hasFrames, image.shape[0], image.shape[1]
    return hasFrames, 0, 0

# Convert Video to JPEG frames
def extract_movie_and_crop(movie_path: str, seq_ID: str):
    vidcap = cv2.VideoCapture(movie_path)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    duration = frame_count/fps
    print('fps = ' + str(fps))
    print('number of frames = ' + str(frame_count))
    duration = frame_count/fps
    print('duration (S) = ' + str(duration))
    minutes = int(duration/60)
    seconds = duration%60
    print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))
    print("int sec : ", int(seconds))


    sec = 0
    frameRate = 1.0*0.1

    new_image_dir = os.path.join(movie_path[:-4],"images")
    print('new_image_dir = ', new_image_dir)
    print("\nnew_image_dir", new_image_dir)
    if not os.path.exists(new_image_dir):
        os.makedirs(new_image_dir)

    count=1
    success, height, width = getFrame(vidcap, sec, count, seq_ID, new_image_dir)
    while success:
        #while sec <= seconds:
        if sec <= seconds:
            count = count + 1
            #sec = round(sec, 2)
            print("sec and seconds : ", sec, " and ", seconds)
            success,_,_ = getFrame(vidcap, sec, count, seq_ID, new_image_dir)
            if sec < int(seconds):
                sec = sec + frameRate
            else:
                break 
        else:
            print("sec in esle : ", sec)
            break
    print("end")
    vidcap.release()

# =============================================================================
#   SETTINGS
# =============================================================================

## Parcourir le dossier de la Dataset 

BASE_DIR = '/home/lenny/Bureau/Données_UCF/'
DATABASE_DIR = 'UCF50-4-classes'
    
    
# =============================================================================
#   RESAMPLE VIDEOS
# =============================================================================

video_folders = sorted(glob.glob(BASE_DIR + DATABASE_DIR + os.sep + "*"))


for index, video_folder in enumerate(video_folders):
    print(index, "/", len(video_folder), video_folder)

    images_path = video_folder + "/images"
    print("images_path : ", images_path)
    if os.path.exists(images_path):
        image_path = glob.glob(video_folder + "/images/*.jpg")
        print("\nnb images before: ", len(image_path))
        ## If images in directory, delete images for create new images
        for file in os.listdir(images_path):
            if file.endswith('.jpg'):
                print(file)
                os.remove(os.path.join(images_path, file))


    video_paths = glob.glob(video_folder + os.sep + '/*.avi')
    print("\nvideo_paths : ",video_paths)

    # if video_path:  
    for video_path in video_paths:        
        extract_movie_and_crop(video_path, seq_ID=video_path[:-4])






