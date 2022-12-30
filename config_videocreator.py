"""
Helper module 
to create training and testing samples (videos) 
based on the available dataset (a collection of time-series images)
Author: Monica Rotulo
"""

import os, random, linecache

from torch import equal


def count_lines_in_file(filepath):
    """ 
        Count number of lines in a file

        Arguments:
        ----------
        filepath: path of the file
    """
    file = open(filepath,"r")
    counter = 0
    content = file.read()
    content_list = content.split("\n")
    for i in content_list:
        if i:
            counter +=1
    
    file.close()
    return counter


def parse_config(root, init_path, n_total_frames):
    """ 
        Read the config file (containing all the names of the folders of the dataset "images")
        and construct new files for training and testing 
        with the list of video dataset up to "n_total_frames" frames

        Arguments:
        ----------
        root: root directory
        init_path:
        n_total_frames: how many frames I want to consider for my video
    """

    train_path = os.path.join(root, "config_file_train")     # set new config file
    test_path = os.path.join(root, "config_file_test")

    print("creating the config train and test files... ")
    with open(init_path) as f:
        for line in f:      #parse lines in original config 
            line = line.strip()   
            sub_path = os.path.join(root, line)
            if os.path.exists(sub_path) and line != "":
                tot_images_in_subfolder = len(next(os.walk(sub_path))[2]) #dir is your directory path as string
                if n_total_frames > tot_images_in_subfolder-1:      
                    with open(test_path, "a") as newfile:     
                        newfile.write(line+" "+str(1)+" "+str(tot_images_in_subfolder-1)+"\t\n")
                else: 
                    start = 1
                    while start != (tot_images_in_subfolder-n_total_frames+1):
                        with open(train_path, "a") as newfile:     #'a' open for writing, appending to the end of the file if it exists
                            newfile.write(line+" "+str(start)+" "+str(start+n_total_frames-1)+"\t\n")
                            start += 1
                
    print("config files created")
    return train_path, test_path



if __name__ == '__main__':

    videos_root = os.path.join(os.getcwd(), 'code\data\dataset_I')
    init_path = os.path.join(videos_root, 'config_file')

    train_path, test_path = parse_config(videos_root, init_path, n_total_frames=20)
