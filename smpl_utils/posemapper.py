'''

About this file:
================
Based on smpl_webuser/posemapper.py
This module defines the mapping of joint-angles to pose-blendshapes. 

Modules included:
- posemap:
  computes the joint-to-pose blend shape mapping given a mapping type as input
'''

import chumpy as ch 
import numpy as np 
import cv2 

class Rodrigues(ch.Ch):
    # subclassing Ch
    # http://files.is.tue.mpg.de/black/papers/chumpy_tutorial.pdf
    # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    dterms = 'rt' # self data name; 
    def compute_r(self):
        # forward 
        return cv2.Rodrigues(self.rt.r)[0] # dst
    
    def compute_dr_wrt(self, wrt):
        # backward; jacobian = 
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T # jacobian

def lrotmin(p): 
    if isinstance(p, np.ndarray):
        p = p.ravel()[3:]
        return np.concatenate([(cv2.Rodrigues(np.array(pp))[0]-np.eye(3)).ravel() for pp in p.reshape((-1,3))]).ravel()        
    if p.ndim != 2 or p.shape[1] != 3:
        p = p.reshape((-1,3))
    p = p[1:] # joint[0] & joint[1] has the same position; 
    return ch.concatenate([(Rodrigues(pp)-ch.eye(3)).ravel() for pp in p]).ravel() # .ravel() == .view(); streach to a line;

def posemap(s):
    if s == 'lrotmin':
        return lrotmin
    else:
        raise Exception('Unknown posemapping: %s' % (str(s),))