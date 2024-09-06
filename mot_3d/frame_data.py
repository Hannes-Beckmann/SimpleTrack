""" input form of the data in each frame
"""
from .data_protos import BBox
import numpy as np, mot_3d.utils as utils
from numpy import ndarray
from typing import Dict, List, Optional


class FrameData:
    def __init__(self, dets: List[ndarray], ego: ndarray, time_stamp: Optional[float]=None, pc: Optional[ndarray]=None, det_types: Optional[List[int]]=None, aux_info: Optional[Dict[str, bool]]=None) -> None:
        self.dets = dets         # detections for each frame
        self.ego = ego           # ego matrix information
        self.pc = pc
        self.det_types = det_types
        self.time_stamp = time_stamp
        self.aux_info = aux_info

        for i, det in enumerate(self.dets):
            self.dets[i] = BBox.array2bbox(det)
        
        # if not aux_info['is_key_frame']:
        #     self.dets = [d for d in self.dets if d.s >= 0.5]