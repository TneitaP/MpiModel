import pickle
import os
import open3d as o3d 
import numpy as np 
import com_utils.serialization as pkl_loader
import com_utils.operate3d as operate3d


if __name__ == "__main__":

    avg_model_path = "template_pkl/person_std_model/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
    assert os.path.isfile(avg_model_path), "illegal dir"
    gm_avg_model = pkl_loader.load_model(avg_model_path)

    avg_ani_mesh = operate3d.catch_model2o3dmesh(gm_avg_model, coord_mode = "xyz")
    avg_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(gm_avg_model, coord_mode = "xyz", pRadius=0.08)

    gm_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    # smpl: J[24]; smal: J[33]
    operate3d.draw_Obj_Visible([avg_ani_mesh, gm_mesh_frame, avg_joint_sphere_Lst], window_name = "Template mesh")

    for i in range(3):
        gm_avg_model.pose[:] = np.random.rand(gm_avg_model.pose.size) * .2
        gm_avg_model.betas[:] = np.random.rand(gm_avg_model.betas.size) * .03 
        cur_ani_mesh = operate3d.catch_model2o3dmesh(gm_avg_model, coord_mode = "xyz")
        operate3d.draw_Obj_Visible([cur_ani_mesh, gm_mesh_frame], window_name = "Template mesh")
