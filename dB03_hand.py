# Load the rest pose
# API for both SMPL & SMAL
import os 
import pickle 
import numpy as np 
import open3d as o3d 
import copy 

import smpl_utils.smpl_np as smplnp_load 
import com_uitils.operate3d as operate3d 


if __name__ == "__main__":
    gm_switch = "pure_hand"
    rest_model_path = "template_pkl/hand_std_model/MANO_RIGHT.pkl"
    gm_coord_mode = "xyz"

    gm_rest_model = smplnp_load.MANOModel(rest_model_path) # flat_hand_mean=True
    rest_mesh = operate3d.catch_model2o3dmesh(gm_rest_model, coord_mode = gm_coord_mode, model_format = "np")
    # rest_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(gm_rest_model, coord_mode = gm_coord_mode, pRadius=0.08)
    gm_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # operate3d.draw_Obj_Visible([rest_mesh, gm_mesh_frame], window_name = "Template mesh")
    

    # change the rot and observe the result:
    pose_Arr = np.hstack(
            (
                np.array([np.pi/2, 0, 0]),
                np.random.rand(6)
            )
    )

    gm_rest_model.set_params(pose_coeffs= pose_Arr) 
    gm_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]) 
    operate3d.draw_Obj_Visible([rest_mesh, gm_mesh_frame], window_name = "Template mesh") 

    
