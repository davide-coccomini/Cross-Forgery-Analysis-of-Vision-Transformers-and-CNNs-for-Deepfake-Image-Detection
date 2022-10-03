import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List


import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
from torch.utils.data import Dataset


class VideoFaceDetector(ABC):

    def __init__(self, **kwargs) -> None:
        super().__init__()

    @property
    @abstractmethod
    def _batch_size(self) -> int:
        pass

    @abstractmethod
    def _detect_faces(self, frames) -> List:
        pass


class FacenetDetector(VideoFaceDetector):

    def __init__(self, device="cuda:0") -> None:
        super().__init__()
        self.detector = MTCNN(margin=0,thresholds=[0.90, 0.90, 0.90], device=device)

    def _detect_faces(self, frames) -> List:
        batch_boxes, *_ = self.detector.detect(frames, landmarks=False)
        if batch_boxes is None:
            return []
        return [b.tolist() if b is not None else None for b in batch_boxes]

    @property
    def _batch_size(self):
        return 32


class VideoDataset(Dataset):

    def __init__(self, videos) -> None:
        super().__init__()
        self.videos = videos

    def __getitem__(self, index: int):
        video = self.videos[index]
        frames_names = os.listdir(video)
        
        frames_num = len(frames_names)
        
        frames = OrderedDict()
        for i in range(frames_num):
            frame_path = os.path.join(video, frames_names[i])
            frame = Image.open(frame_path)
            frame = frame.resize(size=[s // 2 for s in frame.size])
            frames[frames_names[i]] = frame
        return video, list(frames.keys()), list(frames.values())

    def __len__(self) -> int:
        return len(self.videos)
