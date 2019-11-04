import pickle
import os
import open3d as o3d 
import numpy as np 
import com_utils.serialization as pkl_loader
import com_utils.operate3d as operate3d 


def observe_posed_change(pModel, pObserved_dim):
    t_deform_model = pModel.copy()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    mesh_Lst = [operate3d.catch_model2o3dmesh(t_deform_model), mesh_frame]
    for i in range(3):
        cur_pose_Arr = np.array(t_deform_model.pose[:]) # (33, )
        cur_pose_Arr[pObserved_dim] +=np.pi/6
        t_deform_model.pose[:] = cur_pose_Arr
        cur_ani_mesh = operate3d.catch_model2o3dmesh(t_deform_model, True)
        #cur_joint_sphere_Lst = base_utils.creat_joint_as_sphereLst(t_deform_model)
        #base_utils.draw_Obj_Visible([rest_ani_mesh, cur_ani_mesh, cur_joint_sphere_Lst, gm_mesh_frame], window_name = "rest_deform_"+str(i))
        mesh_Lst.append(cur_ani_mesh)
    operate3d.draw_Obj_Visible(mesh_Lst, window_name = "rest_deform"+str(pObserved_dim))
    del mesh_Lst

def observe_position_change(pModel, pTransArr, coord_mode = "xyz"):
    print("translate step:",  pTransArr)
    t_deform_model = pModel.copy()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    origin_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(t_deform_model)

    mesh_Lst = [operate3d.catch_model2o3dmesh(t_deform_model), origin_joint_sphere_Lst, mesh_frame]
    if coord_mode == "xyz":
        pTransArr = pTransArr[[0,2,1]]
    t_deform_model.trans[:] = pTransArr
    
    cur_ani_mesh = operate3d.catch_model2o3dmesh(t_deform_model, True)
    
    t_deform_model.J = np.dot(t_deform_model.J_regressor.toarray(), t_deform_model.r) # recompute the lbs
    trans_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(t_deform_model)
    
    mesh_Lst.append(cur_ani_mesh)
    mesh_Lst.append(trans_joint_sphere_Lst)
    
    operate3d.draw_Obj_Visible(mesh_Lst, window_name = "trans_demo")
    del mesh_Lst
    
if __name__ == "__main__":

    
    # step1. load average model
    rest_model_path = "template_pkl/animal_std_model/smal_CVPR2017.pkl"
    assert os.path.isfile(rest_model_path), "illegal dir"
    gm_rest_model = pkl_loader.load_model(rest_model_path)
    rest_ani_mesh = operate3d.catch_model2o3dmesh(gm_rest_model)
    
    # rest_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(gm_rest_model)
    # operate3d.draw_Obj_Visible([rest_ani_mesh, rest_joint_sphere_Lst], window_name = "Original Template mesh")

    # len(pose[:]) = 99 = 33* 3
    # > Demonstrate the rigid rotate; (w.r.t to join[0])
    # for observed_dim_idx in range(3):
    #     # i choose the pose dim want to observe 
    #     observe_posed_change(gm_rest_model, observed_dim_idx)


    # > Demonstrate the translate; (w.r.t to world coord)
    observe_position_change(gm_rest_model, np.array([0,0,1]))
    