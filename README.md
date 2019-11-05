# SMPL and SMAL based on py37

## Overview

Based on the numpy and open3d(0.8.0) in python3.7, we deploy the morphing betas and poses of SMPL and SMAL model. This repo can help you learn and debug these two model more efficiently. 

## Teaser
|    Name    | Usage |
| ----------        | --- |
| dB01       |  load rest model from the pkl, and observe the beta para.
| dB02       |  load typical poses from pkl, and observe the pose para.

Here is some screen-shot of the two demo.
- *dB01*: View how the first 3 **beta[:3]** parameters identify the person and animal.
    Notice that when increasing **beta[0]** , the female become taller while the male become shorter.

![image](illus/illu_shape_3_s.png)
- *dB02*: View how the first 3 **pose[:3]** parameters(in 1-D scope) influence the model.
    They all control the whole body to rotate w.r.t. **point J[0]**, called " root orientation". 
![image](illus/illu_pose_rigid_s.png)
- *dB02*: For the combined influence of the rest parameters in **pose[3:]** cannot be controlled easily, we directly load the **pose[3:]** from the .pkl and can clearly distinguish different poses as following:
![image](illus/illu_pose_nonrigid_s.png)
## File Tree
|    Folder Name    | Usage |
| ----------        | --- |
| com_utils         |  the utilities based on open3d |
| smpl_utils        |  the class definition of SMPL(also for SMAL) |
| template_pkl      | the parameters of rest shapes(diff identity) and typical poses(diff posed) |

## Installation
Environment: python3.7; 
Dependency: numpy, chumpy(only for read old .pkl data format), open3d(Visualization & 3D Operation); 

To create the environment, you can:
```
conda create -n smpl37 python=3.7
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
(Nevertheless, chumpy is also needed for some original pikle loading process for **Shape blendshapes** (params['shapedirs'] in the code), but we don't use it when computing. )

