'''
About this file:
================
Based on smpl_webuser/serialization.py
'''
import pickle 
import numpy as np 
import chumpy as ch 
import os
import sys
sys.path.append(r"D:\3rdPartyLib\Self_Hub\nonRigid\SMAL_py37")
import smpl_utils.posemapper as posemapper
import smpl_utils.verts as verts
import com_utils.operate3d as operate3d

def backwards_compatibility_replacements(pData_Dic):
    # compatibility and unity in diff format.
    # ['v_template'] <- {['default_v'] / ['template_v']}; 
    # ['J_regressor'] <- ['joint_regressor']
    # ['posedirs'] <- ['blendshapes']
    # ['J'] <- ['joints']
    # ['bs_style'] = 'lbs'
    if 'default_v' in pData_Dic:
        pData_Dic['v_template'] = pData_Dic['default_v']
        del pData_Dic['default_v']
    if 'template_v' in pData_Dic:
        pData_Dic['v_template'] = pData_Dic['template_v']
        del pData_Dic['template_v']
    if 'joint_regressor' in pData_Dic:
        pData_Dic['J_regressor'] = pData_Dic['joint_regressor']
        del pData_Dic['joint_regressor']
    if 'blendshapes' in pData_Dic:
        pData_Dic['posedirs'] = pData_Dic['blendshapes']
        del pData_Dic['blendshapes']
    if 'J' not in pData_Dic:
        pData_Dic['J'] = pData_Dic['joints']
        del pData_Dic['joints']

    # defaults
    if 'bs_style' not in pData_Dic:
        pData_Dic['bs_style'] = 'lbs'

def ready_arguments(pData_Dic):
    if 'trans' not in pData_Dic:
        pData_Dic['trans'] = np.zeros(3)
    if 'pose' not in pData_Dic:
        pData_Dic['pose'] = np.zeros(pData_Dic['kintree_table'].shape[1]*3) # (2,33),  |-> (33*3)
    if 'shapedirs' in pData_Dic and 'betas' not in pData_Dic:
        pData_Dic['betas'] = np.zeros(pData_Dic['shapedirs'].shape[-1]) # (3889, 3, 41) |-> (41)

    for ToChumypy_key_i in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
        if (ToChumypy_key_i in pData_Dic) and not hasattr(pData_Dic[ToChumypy_key_i], 'dterms'):
            pData_Dic[ToChumypy_key_i] = ch.array(pData_Dic[ToChumypy_key_i])


    posemap_Rodrigues_func_ptr = posemapper.posemap(pData_Dic['bs_type'])
    pose_map_ChArr = posemap_Rodrigues_func_ptr(pData_Dic['pose'])

    if 'shapedirs' in pData_Dic:
        # v_shaped = shapedirs * betas(betas ctrl part) + v_template
        pData_Dic['v_shaped'] = pData_Dic['v_template'] + pData_Dic['shapedirs'].dot(pData_Dic['betas'])# (3889, 3)
        # pData_Dic['J'] = ch.dot(pData_Dic['J_regressor'].toarray(), pData_Dic['v_shaped'])# (33, 3889) * (3889, 1)
        # pData_Dic['v_posed'] = pData_Dic['v_shaped'] + pData_Dic['posedirs'].dot(pose_map_ChArr)
        pData_Dic['v_posed'] = pData_Dic['v_shaped'] + ch.dot(pData_Dic['posedirs'], (pose_map_ChArr))
    else:
        # don't have shape, directly aserial_Dic pose
        pData_Dic['v_posed'] = pData_Dic['v_template'] + ch.dot(pData_Dic['posedirs'], (pose_map_ChArr))
    
    pData_Dic['J'] = ch.dot(pData_Dic['J_regressor'].toarray(), pData_Dic['v_posed'])
    return pData_Dic

def load_model(pFname):
    
    serial_Dic = pickle.load(open(pFname,'rb'),encoding='latin1')
    # .keys() = 'f', 'J_regressor', 'kintree_table', 'J', 'bs_style', 'weights', 'posedirs', 'v_template', 'shapedirs', 'bs_type'

    # 1) <np.narray> f (7774,3)
    # 8) <np.narray> v_template (3889, 3)
    # 6)                                              <np.narray> weights(3889, 33)
    # 3)                                              <np.narray>kintree_table (2, 33)
    # 9) <chumpy.ch.Ch>'shapedirs'(3889, 3, 41)
    # 7) <np.narray> 'posedirs' (3889, 3, 288)
    # 2) <scipy.csc_matrix> |-> <np.narray> "J_regressor".toarray() (33, 3889)
    # 4) <np.narray> J (33,3)
    # 5)                                              ['bs_style'] = 'lbs'; 
    # A)                                              ['bs_type'] ='lrotmin'

    backwards_compatibility_replacements(serial_Dic)
    serial_Dic = ready_arguments(serial_Dic)

    args_Dic = {
        'pose': serial_Dic['pose'],
        'v': serial_Dic['v_posed'],
        'J': serial_Dic['J'],
        'weights': serial_Dic['weights'],
        'kintree_table': serial_Dic['kintree_table'],
        'xp': ch,
        'want_Jtr': True,
        'bs_style': serial_Dic['bs_style']
    }

    result, Jtr = verts.verts_core(**args_Dic)
    result = result + serial_Dic['trans'].reshape((1,3))
    result.J_transformed = Jtr + serial_Dic['trans'].reshape((1,3))

    for k, v in serial_Dic.items():
        setattr(result, k, v)
    
    return result

def save_model(model, fname):
    m0 = model
    trainer_dict = {'v_template': np.asarray(m0.v_template),'J': np.asarray(m0.J),'weights': np.asarray(m0.weights),'kintree_table': m0.kintree_table,'f': m0.f, 'bs_type': m0.bs_type, 'posedirs': np.asarray(m0.posedirs)}    
    if hasattr(model, 'J_regressor'):
        trainer_dict['J_regressor'] = m0.J_regressor
    if hasattr(model, 'J_regressor_prior'):
        trainer_dict['J_regressor_prior'] = m0.J_regressor_prior
    if hasattr(model, 'weights_prior'):
        trainer_dict['weights_prior'] = m0.weights_prior
    if hasattr(model, 'shapedirs'):
        trainer_dict['shapedirs'] = m0.shapedirs
    if hasattr(model, 'vert_sym_idxs'):
        trainer_dict['vert_sym_idxs'] = m0.vert_sym_idxs
    if hasattr(model, 'bs_style'):
        trainer_dict['bs_style'] = model.bs_style
    else:
        trainer_dict['bs_style'] = 'lbs'
    pickle.dump(trainer_dict, open(fname, 'wb'), -1)

if __name__ == "__main__": 

    import open3d as o3d 
    gm_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    # step1. load average model
    rest_model_path = "template_pkl/animal_std_model/smal_CVPR2017.pkl"
    assert os.path.isfile(rest_model_path), "illegal dir"
    gm_rest_model = load_model(rest_model_path)
    rest_ani_mesh = operate3d.catch_model2o3dmesh(gm_rest_model)
    
    rest_joint_sphere_Lst = operate3d.creat_joint_as_sphereLst(gm_rest_model)
    operate3d.draw_Obj_Visible([rest_ani_mesh, rest_joint_sphere_Lst, gm_mesh_frame], window_name = "Template mesh")


    # saving test ...
    save_model(gm_rest_model, "template_pkl/animal_std_model/smal_VCL2020.pkl")