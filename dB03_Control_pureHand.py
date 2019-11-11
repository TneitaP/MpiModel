# Load the rest pose
# API for both SMPL & SMAL
import os 
import pickle 
import numpy as np 
np.random.seed(5)
import open3d as o3d 
import copy 

import smpl_utils.smpl_np as smplnp_load 
import com_uitils.operate3d as operate3d 


if __name__ == "__main__":
    gm_switch = "pure_hand"
    gm_coord_mode = "xyz"
    gm_side = "right"
    if gm_side == "right": 
        rest_model_path = "template_pkl/hand_std_model/MANO_RIGHT.pkl"
    elif gm_side == "left": 
        rest_model_path = "template_pkl/hand_std_model/MANO_LEFT.pkl"

    gm_rest_model = smplnp_load.MANOModel(rest_model_path) # flat_hand_mean=True, , flat_hand_mean=True
    rest_mesh = operate3d.catch_model2o3dmesh(gm_rest_model, coord_mode = gm_coord_mode, model_format = "np")
    rest_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(gm_rest_model, coord_mode = gm_coord_mode, pRadius=0.015)
    gm_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    operate3d.draw_Obj_Visible([rest_mesh, gm_mesh_frame, rest_joint_sphere_Lst], window_name = "Template mesh")
    
    print( np.zeros(6)) 
    # change the rot and observe the result:
    for i in range(4):
        pose_Arr = np.hstack(
                (
                    np.array([-np.pi/2, 0, 0]),
                    #np.random.rand(6)*3-1.5
                    np.zeros(6)
                )
                        )
        pose_Arr[5] = -1*i
        print(pose_Arr[3:])
        gm_rest_model.set_params(pose_coeffs= pose_Arr) 
        posed_mesh = operate3d.catch_model2o3dmesh(gm_rest_model, coord_mode = gm_coord_mode, model_format = "np")
        posed_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(gm_rest_model, coord_mode = gm_coord_mode, pRadius=0.015)
        operate3d.draw_Obj_Visible([posed_mesh, posed_joint_sphere_Lst, gm_mesh_frame], window_name = "Template mesh") 

        
