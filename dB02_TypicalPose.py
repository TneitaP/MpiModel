

import pickle 
import numpy as np 
import open3d as o3d 
import copy 

import smpl_utils.smpl_np as smplnp_load 
import com_uitils.operate3d as operate3d



g_animal_pose_path_Lst = [
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


g_person_pose_path_Lst = [
                    
                    'template_pkl/person_pose_pkl/0463_walking.pkl',
                    'template_pkl/person_pose_pkl/0024_lookdown.pkl',
                    'template_pkl/person_pose_pkl/0037_squat.pkl',
                    'template_pkl/person_pose_pkl/0047_bow.pkl',
                    'template_pkl/person_pose_pkl/0175_sitdown.pkl',
                    'template_pkl/person_pose_pkl/0191_hello.pkl',
                    'template_pkl/person_pose_pkl/0659_grasp.pkl',
                    ]


def get_pose_para_from_file_Dic(para_Dic):
    # for animal, pose[] is ch.Ch; and the shape is (n_J*3, )
    pose_in_Arr = para_Dic['pose']
    if not isinstance(pose_in_Arr, np.ndarray):
        pose_in_Arr = np.array(pose_in_Arr)
    if pose_in_Arr.ndim == 1:
        pose_in_Arr = pose_in_Arr.reshape((-1, 3))
    # std pose para shape = (n_J, 3), type = np.narray 
    # pose_in_Arr[0] = 0 # the global rotation should set to 0;
    # pose_in_Arr *= 0
    pose_in_Arr[0] = np.zeros(pose_in_Arr[0].shape)
    return pose_in_Arr


def switch_pose(pModel, pose_path_Lst, pCoord_mode):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    for idx_j, pose_Dic_j_path in enumerate(pose_path_Lst):
        print ("cur_pose name = "+ pose_Dic_j_path)
        pose_Dic_j = pickle.load(open(pose_Dic_j_path,'rb'),encoding='latin1')
        print(pose_Dic_j.keys())
        needed_pose_Arr = get_pose_para_from_file_Dic(pose_Dic_j)
        pModel.set_params(pose= needed_pose_Arr)
        posed_mesh = operate3d.catch_model2o3dmesh(pModel, coord_mode = pCoord_mode, model_format = "np")
        posed_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(pModel, coord_mode = pCoord_mode)
        operate3d.draw_Obj_Visible([posed_mesh, posed_joint_sphere_Lst, mesh_frame], window_name = "posed mesh")



def align_to_self_rot_center(pModel):
    # for animal, when edit the first 3 pose_para, 
    # it will rotate w.r.t. # 0
    pModel.set_params(trans = -pModel.J[0])
    return pModel

def observe_pose_change(pModel, pObserved_dim, pCoord_mode):
    pModel = copy.deepcopy(pModel)
    align_to_self_rot_center(pModel)
    # observe shape both for person and animal 
    # for pose is 2-dim, firstly we should trans pObserved_dim -> [x][y]
    dim_y = pObserved_dim //3
    dim_x = pObserved_dim % 3 
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    mesh_Lst = [operate3d.catch_model2o3dmesh(pModel, coord_mode = pCoord_mode, model_format = "np"), mesh_frame]
    operate3d.draw_Obj_Visible(mesh_Lst, window_name = "Template mesh")
    mesh_interval_Arr = np.array([0, 0, -0.5])
    color_Lst_Arr = np.array([[255,182,193], [255,105,180], [220,20,60]])/255 # [128,0,128],
    for i in range(3):
        #cur_pose_Arr = np.array(pModel.pose[:]) # (33, )
        #cur_pose_Arr[pObserved_dim] +=np.pi/6
        #pModel.pose[:] = cur_pose_Arr
        cur_pose_Arr = pModel.pose # (33, )
        cur_tran_Arr = pModel.trans # (3, )
        cur_pose_Arr[dim_y][dim_x] +=np.pi/6
        # cur_tran_Arr += mesh_interval_Arr
        pModel.set_params(pose = cur_pose_Arr) # , trans= cur_tran_Arr
        cur_ani_mesh = operate3d.catch_model2o3dmesh(pModel, coord_mode = pCoord_mode, paint_color_Arr= color_Lst_Arr[i], model_format = "np")
        cur_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(pModel, coord_mode= pCoord_mode)
        #base_utils.draw_Obj_Visible([rest_ani_mesh, cur_ani_mesh, cur_joint_sphere_Lst, gm_mesh_frame], window_name = "rest_deform_"+str(i))
        mesh_Lst.append(cur_ani_mesh)
        mesh_Lst.append(cur_joint_sphere_Lst)
    operate3d.draw_Obj_Visible(mesh_Lst, window_name = "rest_deform"+str(pObserved_dim))
    del mesh_Lst

if __name__ == "__main__":

    # load shape template. 
    gm_switch = "animal"
    if gm_switch == "animal":
        rest_model_path = "template_pkl/animal_std_model/rest_animals/smal_rest_dogs.pkl"
        gm_coord_mode = "xzy"
        gm_pose_path_Lst = g_animal_pose_path_Lst
    elif gm_switch == "person":
        rest_model_path = "template_pkl/person_std_model/basicmodel_m_lbs_10_207_0_v1.0.0.pkl"
        gm_coord_mode = "xyz"
        gm_pose_path_Lst = g_person_pose_path_Lst
    gm_rest_model = smplnp_load.SMPL_Model(rest_model_path, class_name= gm_switch)
    print("oringin pose para:", gm_rest_model.pose.shape, gm_rest_model.pose[0])
    gm_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    rest_mesh = operate3d.catch_model2o3dmesh(gm_rest_model, coord_mode = gm_coord_mode, model_format = "np")
    rest_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(gm_rest_model, coord_mode = gm_coord_mode)
    operate3d.draw_Obj_Visible([rest_mesh, rest_joint_sphere_Lst, gm_mesh_frame], window_name = "Template mesh")

    # step2. load typical pose (pose[3:])
    switch_pose(gm_rest_model, gm_pose_path_Lst, gm_coord_mode)
    
    # step3. observe the rigid rotate (pose[:3])
    # for dim_select_i in range(3):
    #     observe_pose_change(gm_rest_model,dim_select_i, gm_coord_mode)


    
