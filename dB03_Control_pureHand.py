import os 
import pickle 
import numpy as np 
np.random.seed(5)
import open3d as o3d 
import copy 

import smpl_utils.smpl_np as smplnp_load 
import com_uitils.operate3d as operate3d 

# global varible
g_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])


def observe_shape_change(pModel, pObserved_dim = 0, pCoord_mode= "xyz"):
    pModel = copy.deepcopy(pModel)

    mesh_Lst = [operate3d.catch_model2o3dmesh(pModel, coord_mode = pCoord_mode, model_format = "np"),
                operate3d.creat_joint_as_sphereLst(pModel, coord_mode = pCoord_mode, pRadius=0.012), 
                g_mesh_frame]
    mesh_interval_Arr = np.array([0, -0.1, 0])

    Bete_Arr = pModel.beta
    for i in range(4):
        Bete_Arr[pObserved_dim] +=2
        print(Bete_Arr)
        cur_tran_Arr = pModel.trans # (3, )
        cur_tran_Arr += mesh_interval_Arr
        pModel.set_params(beta = Bete_Arr, trans=cur_tran_Arr)
        cur_mesh = operate3d.catch_model2o3dmesh(pModel, coord_mode = pCoord_mode, model_format = "np")
        cur_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(pModel, coord_mode = pCoord_mode, pRadius=0.012)
        mesh_Lst.append(cur_mesh)
        mesh_Lst.append(cur_joint_sphere_Lst)
    operate3d.draw_Obj_Visible(mesh_Lst, window_name = "Shape Observation") 



def observe_pose_change_wCoeffs(pModel, pObserved_dim = 3, pCoord_mode= "xyz"):
    pModel = copy.deepcopy(pModel)
    # first 3 for globel theta of the root joint;
    # others control the PC coefficients.(MANO_wSMPL use 6, atmolst use 45);
    pose_Arr = np.hstack(
                (np.array([-np.pi/2, 0, 0]),
                    #np.random.rand(6)*3-1.5
                    np.zeros(6),
                ))
    for i in range(4):
        pose_Arr[pObserved_dim] = -1*i
        print(pose_Arr[3:])
        pModel.set_params_wCoeffs(pose_coeffs= pose_Arr) 
        cur_posed_mesh = operate3d.catch_model2o3dmesh(pModel, coord_mode = pCoord_mode, model_format = "np")
        cur_posed_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(pModel, coord_mode = pCoord_mode, pRadius=0.012)
        operate3d.draw_Obj_Visible([cur_posed_mesh, cur_posed_joint_sphere_Lst, g_mesh_frame], window_name = "Pose Observation") 


def observe_pose_change(pModel, pObserved_dim, pCoord_mode= "xyz"):
    pModel = copy.deepcopy(pModel)

    dim_y = pObserved_dim //3
    dim_x = pObserved_dim % 3 
    cur_pose_Arr = np.hstack(
                (np.array([-np.pi/2, 0, 0]),
                    #np.random.rand(6)*3-1.5
                    np.zeros(45),
                )).reshape([-1, 3]) # (16,3)
    for i in range(10):
        cur_pose_Arr[dim_y][dim_x] += np.pi/4
        print("cur pose key:" ,cur_pose_Arr[dim_y][dim_x])
        pModel.set_params(pose = cur_pose_Arr)
        print("cur model pose:" ,pModel.pose)
        cur_posed_mesh = operate3d.catch_model2o3dmesh(pModel, coord_mode = pCoord_mode, model_format = "np")
        cur_posed_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(pModel, coord_mode = pCoord_mode, pRadius=0.005)
        operate3d.draw_Obj_Visible([cur_posed_joint_sphere_Lst, g_mesh_frame], window_name = "Pose Observation") 



if __name__ == "__main__":
    gm_switch = "pure_hand"
    gm_coord_mode = "xyz"
    gm_side = "right"
    if gm_side == "right": 
        rest_model_path = "template_pkl/hand_std_model/MANO_RIGHT.pkl"
    elif gm_side == "left": 
        rest_model_path = "template_pkl/hand_std_model/MANO_LEFT.pkl"

    gm_rest_model = smplnp_load.MANO_Model(rest_model_path) # flat_hand_mean=True, , flat_hand_mean=True
    rest_mesh = operate3d.catch_model2o3dmesh(gm_rest_model, coord_mode = gm_coord_mode, model_format = "np")
    rest_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(gm_rest_model, coord_mode = gm_coord_mode, pRadius=0.025)
    
    # operate3d.draw_Obj_Visible([rest_mesh, g_mesh_frame], window_name = "Template mesh") # rest_joint_sphere_Lst[15]
    # rest_mesh, 
    # print(np.zeros(6)) 
    # change the rot and observe the result:
    # observe_shape_change(gm_rest_model, 1)
    observe_pose_change(gm_rest_model, 4) # first 3 dim is global oritation
    # observe_pose_change_wCoeffs(gm_rest_model, 3) # first 3 dim is global oritation
