# Load the rest pose
# API for both SMPL & SMAL
import os 
import pickle 
import numpy as np 
import open3d as o3d 
import copy 

import smpl_utils.smpl_np as smplnp_load 
import com_uitils.operate3d as operate3d



def switch_animal_identity(pAnimal_model, pIdenCollection_path, view_collection = "cluster"):
    # only for animal
    iden_enum_Lst = ["cats", "dogs", "horses", "cows", "hippos"]
    toys_name_Lst = ["cat"] + ["cheetahs"+str(i) for i in range(5)] + \
                    ["lions"+str(i) for i in range(8)] + ["tigers"+str(i) for i in range(7)] + \
                    ["dogs"+str(i) for i in range(2)] + \
                    ["fox"] + ["wolf"] + ["hyena"] + ["deer"] + ["horse"] + ["zebras"+str(i) for i in range(6)] + \
                    ["cows"+str(i) for i in range(4)] + ["hippos"+str(i) for i in range(3)]
    
    iden_Dic = pickle.load(open(pIdenCollection_path,'rb'),encoding='latin1')

    if view_collection == "cluster":
        used_key = 'cluster_means'
        name_Lst = iden_enum_Lst
    elif view_collection == "toys_betas":
        used_key = 'toys_betas'
        name_Lst = toys_name_Lst
    # animal iden_Dic
    # toys_betas is 41 scan animals;
    # cluster_means is 4 mean identitys;
    for i, beta_i in enumerate(iden_Dic[used_key]):
        print('cur mean per-family shape = ' + name_Lst[i])
        pAnimal_model.set_params(beta = beta_i)
        cur_ani_mesh = operate3d.catch_model2o3dmesh(pAnimal_model, paint_color_Arr = [1, 0.706, 0], model_format = "np")
        cur_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(pAnimal_model)
        # gm_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        operate3d.draw_Obj_Visible([cur_ani_mesh, cur_joint_sphere_Lst, gm_mesh_frame], window_name = "anima-identity"+ name_Lst[i])
        # cur_class_path = os.path.join(r"D:\Documents\Git_Hub\TneitaP_repo\SMPL_pure37\template_pkl\animal_std_model", "smal_rest_"+ name_Lst[i]+ ".pkl")
        # pAnimal_model.save_to_pkl_model(cur_class_path)

def observe_shape_change(pModel, pObserved_dim, pCoord_mode):
    # observe shape both for person and animal 
    pModel = copy.deepcopy(pModel)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    mesh_Lst = [operate3d.catch_model2o3dmesh(pModel, coord_mode = pCoord_mode, model_format = "np"),
                operate3d.creat_joint_as_sphereLst(pModel, coord_mode = pCoord_mode), 
                mesh_frame]
    mesh_interval_Arr = np.array([0, 0, -1.5])
    color_Lst_Arr = np.array([[255,182,193], [255,105,180], [220,20,60]])/255 # [128,0,128],
    for i in range(3):
        #cur_pose_Arr = np.array(pModel.pose[:]) # (33, )
        #cur_pose_Arr[pObserved_dim] +=np.pi/6
        #pModel.pose[:] = cur_pose_Arr
        cur_beta_Arr = pModel.beta # (33, )
        cur_tran_Arr = pModel.trans # (3, )
        cur_beta_Arr[pObserved_dim] += 1.5
        cur_tran_Arr += mesh_interval_Arr
        pModel.set_params(beta = cur_beta_Arr, trans= cur_tran_Arr)
        cur_ani_mesh = operate3d.catch_model2o3dmesh(pModel, coord_mode = pCoord_mode, paint_color_Arr= color_Lst_Arr[i], model_format = "np")
        cur_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(pModel, coord_mode= pCoord_mode)
        #base_utils.draw_Obj_Visible([rest_ani_mesh, cur_ani_mesh, cur_joint_sphere_Lst, gm_mesh_frame], window_name = "rest_deform_"+str(i))
        mesh_Lst.append(cur_ani_mesh)
        mesh_Lst.append(cur_joint_sphere_Lst)
    operate3d.draw_Obj_Visible(mesh_Lst, window_name = "rest_deform"+str(pObserved_dim))
    del mesh_Lst


if __name__ == "__main__":

    gm_switch = "person"
    if gm_switch == "animal":
        rest_model_path = "template_pkl/animal_std_model/smal_CVPR2017.pkl"
        iden_collection_path = "template_pkl/animal_std_model/smal_CVPR2017_data.pkl"
        gm_coord_mode = "xzy"
    elif gm_switch == "person":
        rest_model_path = "template_pkl/person_std_model/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
        gm_coord_mode = "xyz"
    gm_rest_model = smplnp_load.SMPL_Model(rest_model_path, class_name= gm_switch)
    rest_mesh = operate3d.catch_model2o3dmesh(gm_rest_model, coord_mode = gm_coord_mode, model_format = "np")
    rest_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(gm_rest_model, coord_mode = gm_coord_mode, pRadius=0.1)

    gm_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    operate3d.draw_Obj_Visible([rest_mesh, gm_mesh_frame, rest_joint_sphere_Lst], window_name = "Template mesh")

    # switch_animal_identity(gm_rest_model, iden_collection_path, view_collection = "cluster") # "cluster"
    
    # for dim_select_i in range(3):
    #     observe_shape_change(gm_rest_model, dim_select_i, gm_coord_mode)






