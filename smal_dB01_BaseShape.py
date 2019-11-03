import pickle
import os
import open3d as o3d 
import numpy as np 
import com_utils.serialization as pkl_loader
import com_utils.operate3d as operate3d

if __name__ == "__main__":
    # load average model
    avg_model_path = "template_pkl/animal_std_model/smal_CVPR2017.pkl"
    assert os.path.isfile(avg_model_path), "illegal dir"
    gm_avg_model = pkl_loader.load_model(avg_model_path)
    avg_ani_mesh = operate3d.catch_model2o3dmesh(gm_avg_model)
    gm_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # load typical model
    gm_ani_name_Lst = ["cats", "dogs", "horses", "cows", "hippos"]
    model_data_path = "template_pkl/animal_std_model/smal_CVPR2017_data.pkl"
    # keys = ['toys_betas'(41), 'cluster_cov'(5), 'cluster_means'(5)]
    typical_shape_Dic = pickle.load(open(model_data_path,'rb'),encoding='latin1')
    for i, betas in enumerate(typical_shape_Dic['cluster_means']):
        gm_avg_model.betas[:] = betas # the shape of animal, dimension = 41
        # gm_avg_model.J = np.dot(gm_avg_model.J_regressor.toarray(), gm_avg_model.r) #np.zeros((33,3))
        print("cur beta shape:", betas.shape)
        cur_ani_mesh = operate3d.catch_model2o3dmesh_Tpose(gm_avg_model, True)
        cur_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(gm_avg_model)
        print('cur mean per-family shape = ' + gm_ani_name_Lst[i])
        operate3d.draw_Obj_Visible([cur_ani_mesh, cur_joint_sphere_Lst, gm_mesh_frame], window_name = "animal_"+ gm_ani_name_Lst[i])
    

    
