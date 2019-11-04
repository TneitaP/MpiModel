import pickle 
import os 
import open3d as o3d 
import numpy as np 
import com_utils.serialization as pkl_loader 
import com_utils.operate3d as operate3d 

def check_pose_path_legal(p_path_Lst): 
    for path_i in p_path_Lst: 
        assert os.path.isfile(path_i) and os.path.splitext(path_i)[-1] == ".pkl", "have illegal pkl:" + path_i 
    print("[info] check all pose pkl legal...", len(p_path_Lst)) 
    print("[info] expecting mesh number : 41*len = ", len(p_path_Lst)*41) 

def clear_model_toT(pmodel):
    pmodel.pose[:] = 0 # T pose
    pmodel.trans[:] = 0 # origin position
    # update the Joint: 
    pmodel.J = np.dot(pmodel.J_regressor.toarray(), pmodel.r) #np.zeros((33,3))
    return pmodel

def load_std_pose_from_posDic(pmodel, pPose_Dic):
    # [0,1,2] referes to the rigid transform 
    pmodel.pose[3:] = np.array(pPose_Dic['pose'])[3:] #0 # pose_j
    # update the Joint: 
    pmodel.J = np.dot(pmodel.J_regressor.toarray(), pmodel.r)
    return pmodel

if __name__ == "__main__":
    # step1. load average model
    rest_model_path = "template_pkl/animal_std_model/smal_CVPR2017.pkl"
    assert os.path.isfile(rest_model_path), "illegal dir"
    gm_rest_model = pkl_loader.load_model(rest_model_path)
    rest_ani_mesh = operate3d.catch_model2o3dmesh(gm_rest_model)
    gm_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    # step2. load typical model
    gm_ani_name_Lst = ["cats", "dogs", "horses", "cows", "hippos"]
    gm_toys_name_Lst = ["cat"] + ["cheetahs"+str(i) for i in range(5)] + \
                    ["lions"+str(i) for i in range(8)] + ["tigers"+str(i) for i in range(7)] + \
                    ["dogs"+str(i) for i in range(2)] + \
                    ["fox"] + ["wolf"] + ["hyena"] + ["deer"] + ["horse"] + ["zebras"+str(i) for i in range(6)] + \
                    ["cows"+str(i) for i in range(4)] + ["hippos"+str(i) for i in range(3)]
    model_data_path = "template_pkl/animal_std_model/smal_CVPR2017_data.pkl"
    # keys = ['toys_betas', 'cluster_cov', 'cluster_means']
    typical_data_Dic = pickle.load(open(model_data_path,'rb'),encoding='latin1')

    # step3. load std pose template
    gm_source_pose_path_Lst = [
                    # 6 cats'
                    'template_pkl/animal_pose_pkl/big_cats/450-122410176-lions-natural-habitat.pkl', 
                    'template_pkl/animal_pose_pkl/big_cats/00211799_ferrari.pkl', 
                    'template_pkl/animal_pose_pkl/big_cats/cougar.pkl', 
                    'template_pkl/animal_pose_pkl/big_cats/lion_yawn.pkl', 
                    'template_pkl/animal_pose_pkl/big_cats/lion3.pkl', 
                    'template_pkl/animal_pose_pkl/big_cats/lion6.pkl', 
                    'template_pkl/animal_pose_pkl/big_cats/MaleLion800.pkl', 
                    'template_pkl/animal_pose_pkl/big_cats/muybridge_132_133_07.pkl', 
                    'template_pkl/animal_pose_pkl/big_cats/muybridge_107_110_03.pkl', 
                    # 7 cows'
                    'template_pkl/animal_pose_pkl/cows/cow2.pkl', 
                    'template_pkl/animal_pose_pkl/cows/cow_alph.pkl', 
                    'template_pkl/animal_pose_pkl/cows/cow_alph4.pkl', 
                    'template_pkl/animal_pose_pkl/cows/cow_alph5.pkl', 
                    'template_pkl/animal_pose_pkl/cows/Davis_cow_00000.pkl', 
                    'template_pkl/animal_pose_pkl/cows/muybridge_076_04.pkl', 
                    'template_pkl/animal_pose_pkl/cows/muybridge_087_04.pkl', 
                    'template_pkl/animal_pose_pkl/cows/timthumb.pkl', 
                    # 6 horses'
                    'template_pkl/animal_pose_pkl/horses/muybridge_071_04.pkl', 
                    'template_pkl/animal_pose_pkl/horses/grazing.pkl', 
                    'template_pkl/animal_pose_pkl/horses/00049424_ferrari.pkl', 
                    'template_pkl/animal_pose_pkl/horses/00057894_ferrari.pkl', 
                    'template_pkl/animal_pose_pkl/horses/muybridge_014_01.pkl', 
                    'template_pkl/animal_pose_pkl/horses/muybridge_075_04.pkl', 
                    # 7 dogs'
                    'template_pkl/animal_pose_pkl/dogs/Brown-And-White-Akita-Dog_alph.pkl', 
                    'template_pkl/animal_pose_pkl/dogs/dog_alph.pkl', 
                    'template_pkl/animal_pose_pkl/dogs/dog2.pkl', 
                    'template_pkl/animal_pose_pkl/dogs/fox_alph.pkl', 
                    'template_pkl/animal_pose_pkl/dogs/fox.pkl', 
                    'template_pkl/animal_pose_pkl/dogs/fox-05.pkl', 
                    'template_pkl/animal_pose_pkl/dogs/fox-06.pkl', 
                    'template_pkl/animal_pose_pkl/dogs/muybridge_097_01.pkl', 
                    'template_pkl/animal_pose_pkl/dogs/muybridge_097_02.pkl', 
                    'template_pkl/animal_pose_pkl/dogs/muybridge_101_03.pkl',
                    'template_pkl/animal_pose_pkl/dogs/muybridge_102_03.pkl',
                    'template_pkl/animal_pose_pkl/dogs/muybridge_104_04.pkl',
                    'template_pkl/animal_pose_pkl/dogs/NORTHERN-INUIT-DOG-3.pkl', 
                    'template_pkl/animal_pose_pkl/dogs/stalking_wolf_cub_by_nieme.pkl',
                    'template_pkl/animal_pose_pkl/dogs/wolf_alph2.pkl',
                    'template_pkl/animal_pose_pkl/dogs/wolf_alph3.pkl',
                    # 3 hippos
                    'template_pkl/animal_pose_pkl/hippos/hippo5.pkl',
                    'template_pkl/animal_pose_pkl/hippos/hippo_alpha_mouthopen2.pkl',
                    'template_pkl/animal_pose_pkl/hippos/Hippo_for_Nat.pkl',
                    ] 
    check_pose_path_legal(gm_source_pose_path_Lst) 

    for idx_j, pose_Dic_j_path in enumerate(gm_source_pose_path_Lst):
        print("cur_pose name = " + pose_Dic_j_path) 
        # pose_Dic_j = pickle.load(open(pose_Dic_j_path,'rb'),encoding='latin1') # gm_source_pose_path_Lst[idx_j]
        pose_Dic_j = pickle.load(open(pose_Dic_j_path,'rb'),encoding='latin1') # pkl_loader.load_model(pose_Dic_j_path)
        # 在指定的 pose 下， 遍历各个Family/ toys
        for idx_i, betas_i in enumerate(typical_data_Dic['toys_betas']):
            print('\t cur family = '+ gm_toys_name_Lst[idx_i] )
            # step41 : load the T pose of different family            
            gm_rest_model.betas[:] = betas_i[:] #np.array(pose_Dic_j['beta']) full = 41;  PCA(4)
            gm_rest_model = clear_model_toT(gm_rest_model)
            cur_ani_mesh = operate3d.catch_model2o3dmesh(gm_rest_model, True)
            cur_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(gm_rest_model)
            operate3d.draw_Obj_Visible([cur_ani_mesh, cur_joint_sphere_Lst, 
                                         gm_mesh_frame, ], window_name = "animal-Origin_"+gm_toys_name_Lst[idx_i] +"_shape-" + str(0))

            # step42: trans shape using loading pose_Dic_j['pose'] && (J_regressor * Vertices)
            cur_NewPose_ani_model = load_std_pose_from_posDic(gm_rest_model, pose_Dic_j)
            cur_NewPose_ani_mesh = operate3d.catch_model2o3dmesh(cur_NewPose_ani_model, True)
            cur_NewPose_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(cur_NewPose_ani_model)
            operate3d.draw_Obj_Visible([ cur_NewPose_ani_mesh, cur_NewPose_joint_sphere_Lst, 
                                         gm_mesh_frame, ], window_name = "animal-Posed(%d)_"%(idx_j)+gm_toys_name_Lst[idx_i] +"_shape-" + str(0))