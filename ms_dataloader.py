"""
Dataloder for the microstructure dataset
Author: Monica Rotulo
"""

import os, glob
from os.path import exists
import numpy as np
import skimage
import random
from PIL import Image  
from torchvision import transforms
import torch
from torch.utils.data.dataset import Dataset

class Video(object):
    """
    This class represents a sample (video)
    Args:
        root_datapath: the system path to the root folder
                       of the videos.
        row: single row from config file: A list with four or more elements where 
             1) The first element is the path to the video sample's frames excluding the root_datapath prefix 
             2) The  second element is the starting frame id of the video
             3) The third element is the inclusive ending frame id of the video
    """
    def __init__(self, row, root_datapath):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])


    @property
    def path(self): #-> str:
        return self._path

    @property
    def start_frame(self): #-> int:
        return int(self._data[1])

    @property
    def end_frame(self): #-> int:
        return int(self._data[2])
    
    @property
    def num_frames(self): #-> int:
        return self.end_frame - self.start_frame + 1  # +1 because end frame is inclusive



class MicroS_Dataset(Dataset):

    def __init__(self, root, config_path, n_frames_input, n_frames_output, imagefile_template, transform, is_train):
      super(MicroS_Dataset, self).__init__()
        
      self.root_path = root
      self.config_path = config_path
      self.is_train = is_train
      self.n_frames_input = n_frames_input
      self.n_frames_output = n_frames_output
      self.n_frames_total = self.n_frames_input + self.n_frames_output
      self.transform = transform
      self.imagefile_template = imagefile_template

      self._parse_config_file()


    def _parse_config_file(self):
        self.video_list = [Video(x.strip().split(), self.root_path) for x in open(self.config_path)]



    def __getitem__(self, idx):
        """
          Loads the frames of a video at the corresponding indices.
          Args:
              idx: Video sample index. 
          Returns:
              A tuple of (video, label). 
              Video is either 1) a list of PIL images if no transform is used
              2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
              if the transform "ImglistToTensor" is used
              Aim to return video = (frames_in, frames_out), where frames_out are my label
          """
        sample: Video = self.video_list[idx]
  
        # load input frames
        tot_frames = []
        input_frames = []
        output_frames = []
        frame_idx = int(sample.start_frame)

        for _ in range(self.n_frames_total):
          file_exists = exists(os.path.join(sample.path, self.imagefile_template.format(frame_idx)))  
          if not file_exists:
              print("file do not exist!", os.path.join(sample.path, self.imagefile_template.format(frame_idx)))
          else:
            image = Image.open(os.path.join(sample.path, self.imagefile_template.format(frame_idx))).convert('RGB')
            # settings based on the used dataset:
            if image.size[0] == 1520:
                image = image.crop((477, 113, 1043, 680))
                half = 0.319
                image = image.resize( [int(half * s) for s in image.size] )
            elif image.size[0] == 500:
                image = image.crop((70, 70, 430, 430))
                half = 0.5
                image = image.resize( [int(half * s) for s in image.size] )
            else:
                print('Size anomaly, file should be skipped:', self.imagefile_template.format(frame_idx))

            tot_frames.append(image)

            if frame_idx < sample.end_frame:
                frame_idx += 1


        if self.transform is not None:
            tot_frames = self.transform(tot_frames)
        
        # now we split between in and out frames
        input_frames = tot_frames[:self.n_frames_input]
        if self.n_frames_output > 0:
            output_frames= tot_frames[self.n_frames_input:self.n_frames_total]
        else:
            output_frames = []

        return input_frames, output_frames


    def __len__(self):
        return len(self.video_list)

class ImglistToTensor(torch.nn.Module):
    """
    Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
    Can be used as first transform for ``MS_Dataset``.
    """
    @staticmethod
    def forward(img_list):
        """
        Converts each PIL image in a list to
        a torch Tensor and stacks them into
        a single tensor.
        Args:
            img_list: list of PIL images.  
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH`` or ``NUM_IMAGES x HEIGHT x WIDTH x CHANNELS`` if permute is used 
        """
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])
