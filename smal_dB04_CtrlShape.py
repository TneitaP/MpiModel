# Resize all shape in a unit_BBx; 
# 
import os 
import pickle 
import numpy as np 
import open3d as o3d 
import com_utils.serialization as pkl_loader 
import com_utils.operate3d as operate3d 


def check_pose_path_legal(p_path_Lst): 
    for path_i in p_path_Lst: 
        assert os.path.isfile(path_i) and os.path.splitext(path_i)[-1] == ".pkl", "have illegal pkl:" + path_i 
    print("[info] check all pose pkl legal...", len(p_path_Lst)) 
    print("[info] expecting mesh number : 41*len = ", len(p_path_Lst)*41) 

def align_model_n4(pmodel):
    # joint_pcd = operate3d.catch_model_Joint2o3dpcd(pmodel) 
    align_no4_point_Arr = pmodel.J[4]
    # print("[info]align_no4_point_Arr: ",  align_no4_point_Arr)
    
    pmodel.trans[:] = -align_no4_point_Arr
    # update the Joint: 
    pmodel.J = np.dot(pmodel.J_regressor.toarray(), pmodel.r) #np.zeros((33,3))
    return pmodel


def load_std_pose_from_posDic(pmodel, pPose_Dic):
    # [0,1,2] referes to the rigid transform
    pmodel.pose[3:] = np.array(pPose_Dic['pose'])[3:] #0 # pose_j
    # update the Joint: 
    pmodel.J = np.dot(pmodel.J_regressor.toarray(), pmodel.r)
    return pmodel 

def clear_model_toT(pmodel):
    pmodel.pose[:] = 0 # T pose
    pmodel.trans[:] = 0 # origin position
    # update the Joint: 
    pmodel.J = np.dot(pmodel.J_regressor.toarray(), pmodel.r) #np.zeros((33,3))
    return pmodel

####################################
#  Add to resize model to identical size
def compute_model_range(pmodel):
    ani_mesh = operate3d.catch_model2o3dmesh(pmodel)
    x_max, y_max, z_max = ani_mesh.get_max_bound()
    x_min, y_min, z_min = ani_mesh.get_min_bound()

    # x_delta, y_delta, z_delta = round(x_max - x_min, 2), round(y_max - y_min, 2), round(z_max - z_min, 2)
    x_delta, y_delta, z_delta = (x_max - x_min), (y_max - y_min), (z_max - z_min)
    return x_delta, y_delta, z_delta

def resize_shape_2normal(pmodel, plog_file):
    # print ("Before Shape Normal:",  compute_model_range(pmodel))
    x_delta, y_delta, z_delta = compute_model_range(pmodel)
    
    cude_edge = 1.8
    max_delta = max(x_delta, y_delta, z_delta)
    if max_delta == x_delta:
        shrink_ruler = 0.06
        
    elif max_delta == y_delta:
        shrink_ruler = 0.03
    else:
        shrink_ruler = 0.01
    
    beta0_delta = -(max_delta - cude_edge) / shrink_ruler * 0.1

    cur_beta_Arr = np.array(pmodel.betas[:]) # (33, )
    cur_beta_Arr[0] += beta0_delta
    pmodel.betas[:] = cur_beta_Arr
    pmodel.J = np.dot(pmodel.J_regressor.toarray(), pmodel.r) #np.zeros((33,3))

    # print ("After Shape Normal:",  compute_model_range(pmodel))
    return pmodel
####################################

def create_unit_BBx(pUnit= 0.9):
    [minX, minY, minZ] = -pUnit,-pUnit,-pUnit
    [maxX, maxY, maxZ] = pUnit,pUnit,pUnit
    points = [[minX, minY, minZ], [maxX, minY, minZ], [minX, maxY, minZ], [maxX, maxY, minZ], 
    [minX, minY, maxZ], [maxX, minY, maxZ], [minX, maxY, maxZ], [maxX, maxY, maxZ]]
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    q_line_set = o3d.geometry.LineSet()
    q_line_set.points = o3d.utility.Vector3dVector(points)
    q_line_set.lines = o3d.utility.Vector2iVector(lines)
    q_line_set.colors = o3d.utility.Vector3dVector(colors)
    return q_line_set 

# g_dataset_save_dir = r"G:\BaiduNetdiskDownload\norigid_SMAL\mesh2"
g_dataset_save_dir = r"F:\norigid_STD_SMAL"

gm_point_unit = o3d.geometry.PointCloud()
gm_point_unit.points = o3d.utility.Vector3dVector(np.array([[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0]])*0.8)
gm_point_unit.paint_uniform_color([1,1,1])
gm_log_txt_path = r"F:\norigid_STD_SMAL\size_log.txt"


if __name__ == "__main__":

    # step1. load and trans avg_model
    avg_model_path = "template_pkl/animal_std_model/smal_CVPR2017.pkl"
    assert os.path.isfile(avg_model_path), "illegal dir"
    gm_avg_model = pkl_loader.load_model(avg_model_path)
    # <chumpy.ch.Ch> gm_avg_model.betas[(41L,)] ; gm_avg_model.pose[(99L,)]; gm_avg_model.trans[(3L, )]; 
    # <chumpy.reordering.transpose>gm_avg_model.J[(33L, 3L)]; <chumpy.reordering.transpose>gm_avg_model.T[(3L, 3889L)];
    # <numpy.ndarray> gm_avg_model.r(vertices)[(3889L, 3L)]; gm_avg_model.f(faces)[(7774L, 3L)]; 
    # <scipy.sparse.csc.csc_matrix> gm_avg_model.J_regressor | <numpy.ndarray>(gm_avg_model.J_regressor.toarray())[(33L, 3889L)]
    avg_ani_mesh = operate3d.catch_model2o3dmesh(gm_avg_model)
    gm_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[-0.4, -0.1, 0.3])
    avg_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(gm_avg_model)
    operate3d.draw_Obj_Visible([avg_ani_mesh, avg_joint_sphere_Lst], window_name = "Original Template mesh")

    # step2. load std pose template
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
    
    # step3. load family_model 
    
    model_data_path = "template_pkl/animal_std_model/smal_CVPR2017_data.pkl"
    data_Dic = pickle.load(open(model_data_path, 'rb'), encoding='latin1') # keys = ['toys_betas', 'cluster_cov', 'cluster_means']

    # gm_family_name_Lst = ["cats", "dogs", "horses", "cows", "hippos"] # 'cluster_means'
    gm_toys_name_Lst = ["cat"] + ["cheetahs"+str(i) for i in range(5)] + \
                    ["lions"+str(i) for i in range(8)] + ["tigers"+str(i) for i in range(7)] + \
                    ["dogs"+str(i) for i in range(2)] + \
                    ["fox"] + ["wolf"] + ["hyena"] + ["deer"] + ["horse"] + ["zebras"+str(i) for i in range(6)] + \
                    ["cows"+str(i) for i in range(4)] + ["hippos"+str(i) for i in range(3)]


    gm_log_file = open(gm_log_txt_path, "a")
    for idx_j, pose_Dic_j_path in enumerate(gm_source_pose_path_Lst):
        print ("cur_pose name = "+ pose_Dic_j_path)
        pose_Dic_j = pickle.load(open(pose_Dic_j_path,'rb'),encoding='latin1')

        # 在指定的 pose 下， 遍历各个Family：
        for idx_i, betas_i in enumerate(data_Dic['toys_betas']):
            # step1 : load the T pose of different family
            print('\t cur family = '+ gm_toys_name_Lst[idx_i] )
            
            gm_avg_model.betas[:] = betas_i[:] #np.array(pose_Dic_j['beta']) full = 41;  PCA(4)
            gm_avg_model = clear_model_toT(gm_avg_model)
            cur_ani_mesh = operate3d.catch_model2o3dmesh(gm_avg_model, True)
            cur_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(gm_avg_model)
            # operate3d.draw_Obj_Visible([cur_ani_mesh, cur_joint_sphere_Lst, 
            #                             gm_mesh_frame, ], window_name = "animal-Origin_"+gm_toys_name_Lst[idx_i] +"_shape-" + str(0))
            
            # step2: trans shape using loading pose_Dic_j['pose'] && (J_regressor * Vertices)
            # pose 迁移过程 [no-rigid process] 

            cur_NewPose_ani_model = load_std_pose_from_posDic(gm_avg_model, pose_Dic_j)
            cur_NewPose_ani_mesh = operate3d.catch_model2o3dmesh(cur_NewPose_ani_model, True)
            cur_NewPose_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(cur_NewPose_ani_model)
            # operate3d.draw_Obj_Visible([ cur_NewPose_ani_mesh, cur_NewPose_joint_sphere_Lst, 
            #                             gm_mesh_frame, ], window_name = "animal-newPose_"+gm_toys_name_Lst[idx_i] +"_shape-" + str(0))
            cur_NewPose_ani_mesh.paint_uniform_color([0.5, 0.5, 0.5])


            # step3-- : resize the model to cube (1,1,1)
            cur_shape_normal_model = resize_shape_2normal(cur_NewPose_ani_model, gm_log_file)

            # step3: Aligned No.4 Joint to (0,0,0)
            cur_Aligned_ani_model = align_model_n4(cur_shape_normal_model)
            cur_Aligned_ani_mesh = operate3d.catch_model2o3dmesh(cur_Aligned_ani_model, True)
            cur_Aligned_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(cur_Aligned_ani_model)

            cur_Aligned_joint_pcd = operate3d.catch_model_Joint2o3dpcd(cur_Aligned_ani_model)
            cur_id = gm_toys_name_Lst[idx_i]  + "_pose-"+ str(idx_j).zfill(2)# + pose_Dic_j_path "animal-Align_"+

            
            operate3d.draw_Obj_Visible([ # gm_mesh_frame,
                                        create_unit_BBx(), 
                                        # compute_model_range(cur_Aligned_ani_model), 
                                        gm_point_unit, # cur_NewPose_ani_mesh, 
                                        cur_Aligned_ani_mesh, #cur_Aligned_joint_pcd, cur_Aligned_joint_sphere_Lst, #cur_Aligned_joint_pcd, 
                                        cur_Aligned_ani_mesh.get_oriented_bounding_box()
                                        ], window_name = cur_id)
            # o3d.io.write_triangle_mesh(filename = os.path.join(g_dataset_save_dir, "mesh", "mesh_"+cur_id + ".ply"), 
            #                             mesh = cur_Aligned_ani_mesh ,
            #                             )

            # o3d.io.write_point_cloud(filename = os.path.join(g_dataset_save_dir, "joint", "joint_"+cur_id + ".ply"),
            #                     pointcloud  = cur_Aligned_joint_pcd, 
            #                     # write_ascii  = True
            #                     )