# SMPL and SMAL deployment based on py37

## Overview

Based on the numpy and open3d(0.8.0) in python3.7, we practice the morphing both on their betas and poses. This repo can help you learn and debug these two model more efficiently. 

## Teaser
|    Name    | Usage |
| dB01       |  load rest model from the pkl, and observe the beta para.
| dB02       |  load typical poses from pkl, and observe the pose para.

Here is some screen-shot of the two demo.
## File Tree
|    Folder Name    | Usage |
| ----------        | --- |
| com_utils         |  the utilities based on open3d |
| smpl_utils        |  the class definition of SMPL(also for SMAL) |
| template_pkl      | the parameters of rest shapes(diff identity) and typical poses(diff posed) |


