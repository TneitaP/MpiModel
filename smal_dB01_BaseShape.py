import pickle
import os
import sys 
sys.path.append(r"D:\Documents\Git_Hub\TneitaP_repo\SMPL_py37")
import open3d as o3d
import numpy as np 
import com_utils.serialization as pkl_loader
import com_utils.operate3d as operate3d

if __name__ == "__main__":
    # load average model
    rest_model_path = "template_pkl/animal_std_model/smal_CVPR2017.pkl"
    assert os.path.isfile(rest_model_path), "illegal dir"
    gm_rest_model = pkl_loader.load_model(rest_model_path)
    rest_ani_mesh = operate3d.catch_model2o3dmesh(gm_rest_model)
    gm_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    rest_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(gm_rest_model)
    operate3d.draw_Obj_Visible([rest_ani_mesh, rest_joint_sphere_Lst, gm_mesh_frame], window_name = "Template mesh")


    # load typical model
    gm_ani_name_Lst = ["cats", "dogs", "horses", "cows", "hippos"]
    model_data_path = "template_pkl/animal_std_model/smal_CVPR2017_data.pkl"
    # keys = ['toys_betas'(41), 'cluster_cov'(5), 'cluster_means'(5)]
    typical_shape_Dic = pickle.load(open(model_data_path,'rb'),encoding='latin1')
    for i, betas in enumerate(typical_shape_Dic['cluster_means']):
        gm_rest_model.betas[:] = betas # the shape of animal, dimension = 41
        # gm_rest_model.J = np.dot(gm_rest_model.J_regressor.toarray(), gm_rest_model.r) #np.zeros((33,3))
        print("cur beta shape:", betas.shape)
        cur_ani_mesh = operate3d.catch_model2o3dmesh_Tpose(gm_rest_model, True)
        cur_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(gm_rest_model)
        print('cur mean per-family shape = ' + gm_ani_name_Lst[i])
        operate3d.draw_Obj_Visible([cur_ani_mesh, cur_joint_sphere_Lst, gm_mesh_frame], window_name = "animal_"+ gm_ani_name_Lst[i])
    

    
