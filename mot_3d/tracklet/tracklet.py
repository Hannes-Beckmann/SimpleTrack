from typing import Any, Dict, Optional, Union

import numpy as np

from mot_3d.update_info_data import UpdateInfoData

from .. import life as life_manager
from .. import motion_model
from ..data_protos import BBox
from ..frame_data import FrameData
from ..update_info_data import UpdateInfoData


class Tracklet:
    def __init__(self, configs: Dict[str, Any], id: int, bbox: BBox, det_type: int, frame_index: int, time_stamp: Optional[float]=None, aux_info=None) -> None:
        self.id = id
        self.time_stamp = time_stamp
        self.asso = configs['running']['asso']
        
        self.configs = configs
        self.det_type = det_type
        self.aux_info = aux_info
        
        # initialize different types of motion model
        self.motion_model_type = configs['running']['motion_model']
        # simple kalman filter
        if self.motion_model_type == 'kf':
            self.motion_model = motion_model.KalmanFilterMotionModel(
                bbox=bbox, inst_type=self.det_type, time_stamp=time_stamp, covariance=configs['running']['covariance'])

        # life and death management
        self.life_manager = life_manager.HitManager(configs, frame_index)
        # store the score for the latest bbox
        self.latest_score = bbox.s
    
    def predict(self, time_stamp: Optional[float]=None, is_key_frame: bool=True) -> BBox:
        """ in the prediction step, the motion model predicts the state of bbox
            the other components have to be synced
            the result is a BBox

            the ussage of time_stamp is optional, only if you use velocities
        """
        result = self.motion_model.get_prediction(time_stamp=time_stamp)
        self.life_manager.predict(is_key_frame=is_key_frame)
        self.latest_score = self.latest_score * 0.01
        result.s = self.latest_score
        return result

    def update(self, update_info: UpdateInfoData) -> None:
        """ update the state of the tracklet
        """
        self.latest_score = update_info.bbox.s
        is_key_frame = update_info.aux_info['is_key_frame']
        
        # only the direct association update the motion model
        if update_info.mode == 1 or update_info.mode == 3:
            self.motion_model.update(update_info.bbox, update_info.aux_info)
        else:
            pass
        self.life_manager.update(update_info, is_key_frame)
        return

    def get_state(self) -> BBox:
        """ current state of the tracklet
        """
        result = self.motion_model.get_state()
        result.s = self.latest_score
        return result
    
    def valid_output(self, frame_index):
        return self.life_manager.valid_output(frame_index)
    
    def death(self, frame_index: int) -> bool:
        return self.life_manager.death(frame_index)
    
    def state_string(self, frame_index: int) -> str:
        """ the string describes how we get the bbox (e.g. by detection or motion model prediction)
        """
        return self.life_manager.state_string(frame_index)
    
    def compute_innovation_matrix(self):
        """ compute the innovation matrix for association with mahalonobis distance
        """
        return self.motion_model.compute_innovation_matrix()
    
    def sync_time_stamp(self, time_stamp: float) -> None:
        """ sync the time stamp for motion model
        """
        self.motion_model.sync_time_stamp(time_stamp)
        return
