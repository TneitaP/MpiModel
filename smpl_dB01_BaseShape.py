'''
Skinned Multi-Person Linear(SMPL) model , API offered by MPI for IS, 2015.
'''
import pickle
import os
import open3d as o3d 
import numpy as np 
import com_utils.serialization as pkl_loader
import com_utils.operate3d as operate3d


if __name__ == "__main__":

    rest_model_path = "template_pkl/person_std_model/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
    assert os.path.isfile(rest_model_path), "illegal dir"
    gm_rest_model = pkl_loader.load_model(rest_model_path)

    rest_person_mesh = operate3d.catch_model2o3dmesh(gm_rest_model, coord_mode = "xyz")
    rest_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(gm_rest_model, coord_mode = "xyz", pRadius=0.08)

    gm_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    operate3d.draw_Obj_Visible([rest_person_mesh, gm_mesh_frame, rest_joint_sphere_Lst], window_name = "Template mesh")

    # smpl_model.J[24]          vs. smpl_model.J[33]
    # smpl_model.betas[10]      vs. smal_model.betas[41]
    # smpl_model.pose[72](24*3) vs. smal_model.pose[99](33*3)