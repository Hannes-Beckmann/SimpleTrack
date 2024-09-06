""" a general interface for aranging the things inside a single tracklet
    data structure for updating the life cycles and states of a tracklet 
"""
from .data_protos import BBox
from . import utils
import numpy as np
from numpy import ndarray
from typing import Dict, List, Optional


class UpdateInfoData:
    def __init__(self, mode: int, bbox: BBox, frame_index: int, ego: ndarray, dets: Optional[List[BBox]]=None, pc: Optional[ndarray]=None, aux_info: Optional[Dict[str, bool]]=None) -> None:
        self.mode = mode   # association state
        self.bbox = bbox
        self.ego = ego    
        self.frame_index = frame_index
        self.pc = pc
        self.dets = dets
        self.aux_info = aux_info
