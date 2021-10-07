import glob
import os 
import pandas as pd
import numpy as np



def create_raw_dataframe(video_folders, final_save_path):

    raw_dataframe = pd.DataFrame(columns=['group', 'video_name', 'label', 'path', 'complete_video_name'])

    for video_folder in video_folders:
        label = video_folder[47:]
        image_paths = sorted(glob.glob(video_folder + '/*.jpg'))
        for image_path in image_paths:
            video_name_1 = image_path[48+len(label):-17]
            video_name_and_label = image_path[:-8]
            group = image_path[47 +len(label) + len(video_name_1) + 2:-13]
            video_name_2 = image_path[48+len(label) + len(video_name_1) + len(group) + 2:-9]
            video_name = video_name_1 + "_" + video_name_2
            complete_video_name = image_path[48+len(label):-9]
            print(image_path)
            print(label)
            print(group)
            print(video_name)
            print(complete_video_name)
            raw_dataframe = raw_dataframe.append({'group': str(group),
                                    'video_name': str(video_name),
                                    'label': str(label),
                                    'path': image_path,
                                    'complete_video_name': str(complete_video_name),
                                    }, ignore_index=True) 

    print(raw_dataframe)
    raw_dataframe.to_pickle(final_save_path)
    return raw_dataframe

BASE_DIR = '/home/lenny/Bureau/Données_UCF/'
DATABASE_DIR = 'UCF50-4-classes/'
DATAFRAME_DIR = 'dataframe_UCF50_4_classes/'

# get label and image_paths in dataframe 
video_folders = sorted(glob.glob(BASE_DIR + DATABASE_DIR + "*"))
final_save_path = os.path.join(BASE_DIR + DATAFRAME_DIR,"frames_dataframe_UCF50-4-Classes.pkl")

if os.path.exists(final_save_path):
    print('dataframe always exist !')
    raw_dataframe = pd.read_pickle(final_save_path)
else:
    # take a coffee...
    print('creation of dataframe ...')
    raw_dataframe = create_raw_dataframe(video_folders, final_save_path)