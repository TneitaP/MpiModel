# SMPL and SMAL deployment based on py37

## Overview

Based on the numpy and open3d(0.8.0) in python3.7, we practice the morphing both on their betas and poses. This repo can help you learn and debug these two model more efficiently. 

## Teaser
|    Name    | Usage |
| ----------        | --- |
| dB01       |  load rest model from the pkl, and observe the beta para.
| dB02       |  load typical poses from pkl, and observe the pose para.

Here is some screen-shot of the two demo.
- *dB01*: View how the first 3 **beta[]** parameters identify the person and animal.
    Notice that when increasing **beta[0]** , the female become taller while the male become shorter.
![image](https://github.com/TneitaP/SMPL_py37/tree/np_pure/illus/illu_shape_3.png)
- *dB02*: View how the first 3 **pose[:3]** parameters(in 1-D scope) influence the model.
    They all control the whole body to rotate w.r.t. **point J[0]**
![image](https://github.com/TneitaP/SMPL_py37/tree/np_pure/illus/illu_pose_rigid.png)
- *dB02*: For the combined influence of the rest parameters in **pose[3:]** cannot be told explicitly, we directly load the **pose[3:]** from the pkl and can clearly distinguish different poses as following:
![image](https://github.com/TneitaP/SMPL_py37/tree/np_pure/illus/illu_pose_nonrigid.png)
## File Tree
|    Folder Name    | Usage |
| ----------        | --- |
| com_utils         |  the utilities based on open3d |
| smpl_utils        |  the class definition of SMPL(also for SMAL) |
| template_pkl      | the parameters of rest shapes(diff identity) and typical poses(diff posed) |


