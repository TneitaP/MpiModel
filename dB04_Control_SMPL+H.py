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


g_person_pose_path_Lst = [
                    
                    'template_pkl/person_pose_pkl/0463_walking.pkl',
                    'template_pkl/person_pose_pkl/0024_lookdown.pkl',
                    'template_pkl/person_pose_pkl/0037_squat.pkl',
                    'template_pkl/person_pose_pkl/0047_bow.pkl',
                    'template_pkl/person_pose_pkl/0175_sitdown.pkl',
                    'template_pkl/person_pose_pkl/0191_hello.pkl',
                    'template_pkl/person_pose_pkl/0659_grasp.pkl',
                    ]


def observe_pose_change_fromArr(pModel, pPose_Arr, pObserved_dim = 3, pCoord_mode= "xyz"):
    pModel = copy.deepcopy(pModel)
    pModel.set_params(pose_coeffs= pPose_Arr) 
    cur_posed_mesh = operate3d.catch_model2o3dmesh(pModel, coord_mode = pCoord_mode, model_format = "np")
    # cur_posed_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(pModel, coord_mode = pCoord_mode, pRadius=0.012)
    operate3d.draw_Obj_Visible([cur_posed_mesh, g_mesh_frame], window_name = "Pose Observation") 

def observe_pose_change_fromFile(pModel, pObserved_dim = 3, pCoord_mode= "xyz"):
    pModel = copy.deepcopy(pModel)
    for pose_path_i in g_person_pose_path_Lst:
        #pModel.set_params(pose_coeffs= pPose_Arr) 
        pModel.set_pose_from_smpl_file(pose_path_i)
        cur_posed_mesh = operate3d.catch_model2o3dmesh(pModel, coord_mode = pCoord_mode, model_format = "np")
        cur_posed_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(pModel, coord_mode = pCoord_mode, pRadius=0.05, smplwH= True)
        operate3d.draw_Obj_Visible([cur_posed_mesh, g_mesh_frame, cur_posed_joint_sphere_Lst], window_name = "Pose Observation") # 
    
if __name__ == "__main__":
    gm_coord_mode = "xyz"
    gm_side = "right"
    rest_body_model_path = "template_pkl/hand_std_model/SMPLH_male.pkl" # SMPLH_female
    rest_lhand_model_path = "template_pkl/hand_std_model/MANO_LEFT.pkl"
    rest_rhand_model_path = "template_pkl/hand_std_model/MANO_RIGHT.pkl"

    gm_rest_model = smplnp_load.SMPLwH_Model(
        smplwH_model_path= rest_body_model_path, 
        left_hand_path = rest_lhand_model_path, 
        right_hand_path = rest_rhand_model_path, 
        ) 
    rest_mesh = operate3d.catch_model2o3dmesh(gm_rest_model, coord_mode = gm_coord_mode, model_format = "np")
    rest_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(gm_rest_model, coord_mode = gm_coord_mode, pRadius=0.05, smplwH= True)
    
    operate3d.draw_Obj_Visible([rest_mesh, rest_joint_sphere_Lst, g_mesh_frame], window_name = "Template mesh")
    
    
    body_pose_Arr = np.array([-0.17192541, +0.36310464, +0.05572387, -0.42836206, -0.00707548, +0.03556427,
             +0.18696896, -0.22704364, -0.39019834, +0.20273526, +0.07125099, +0.07105988,
             +0.71328310, -0.29426986, -0.18284189, +0.72134655, +0.07865227, +0.08342645,
             +0.00934835, +0.12881420, -0.02610217, -0.15579594, +0.25352553, -0.26097519,
             -0.04529948, -0.14718626, +0.52724564, -0.07638319, +0.03324086, +0.05886086,
             -0.05683995, -0.04069042, +0.68593617, -0.75870686, -0.08579930, -0.55086359,
             -0.02401033, -0.46217096, -0.03665799, +0.12397343, +0.10974685, -0.41607569,
             -0.26874970, +0.40249335, +0.21223768, +0.03365140, -0.05243080, +0.16074013,
             +0.13433811, +0.10414972, -0.98688595, -0.17270103, +0.29374368, +0.61868383,
             +0.00458329, -0.15357027, +0.09531648, -0.10624117, +0.94679869, -0.26851003,
             +0.58547889, -0.13735695, -0.39952280, -0.16598853, -0.14982575, -0.27937399,])
    
    left_hand_pose_Arr = np.random.rand(6) #*3-1.5
    right_hand_pose_Arr = np.array([-0.51730816, -1.0675071, -1.00316142, 1.39179159, 1.38068015, -0.93475603, ])
    pose_Arr = np.hstack(
        (body_pose_Arr, left_hand_pose_Arr, right_hand_pose_Arr)
    )
    
    # observe_pose_change(gm_rest_model, pPose_Arr= pose_Arr) 
    gm_rest_model.set_params(pose_coeffs= pose_Arr) 
    observe_pose_change_fromFile(gm_rest_model)

    

