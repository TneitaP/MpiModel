import numpy as np 
import open3d as o3d 


g_spl_color_Arr = np.array(
            [[0.44700521163871676, 0.9149047770645528, 0.2364668865826297], # light green
            [0.546715506420034, 0.2843135945154448, 0.59660967178703],
            [0.6866160772196026, 0.1582440738213644, 0.5352219736082486], # purple
            [0.16737440970029605, 0.7473829817440123, 0.6408770034894606], # qing
            [0.4036741764457761, 0.649254495458317, 0.30266114940417066], 
            [0.5850198414349911, 0.5572068031392857, 0.7897370178925692], # grey
            [0.6061187523676671, 0.02352376025211733, 0.125743623104707], # tiexiuhong
            [0.0983177267033869, 0.45167074814306607, 0.7772092130185153], # blue
            [0.5362068522201038, 0.4823186296727855, 0.2849664954418937], # zong
            [0.24243293467583527, 0.23770070635091722, 0.19855487487190848], # black
            [0.9327154191154875, 0.49505414043035667, 0.28626034672961975], # orange
            [0.10304475744738528, 0.5273551207376438, 0.22594274822347715], # mo lv
            [0.8515360053202375, 0.7461464262302233, 0.12688968916092613], # gold
            [0.6320228352842366, 0.8645799601845567, 0.7505235788900761], # dan lv
            [0.798475028594087, 0.3421010336796366, 0.37228899317607], # rou se
            [0.9798089671888133, 0.9192081421284964, 0.5962808711958855], # dan huang

            [0.44700521163871676, 0.9149047770645528, 0.2364668865826297], # light green
            [0.546715506420034, 0.2843135945154448, 0.59660967178703],
            [0.6866160772196026, 0.1582440738213644, 0.5352219736082486], # purple
            [0.16737440970029605, 0.7473829817440123, 0.6408770034894606], # qing
            [0.4036741764457761, 0.649254495458317, 0.30266114940417066], 
            [0.5850198414349911, 0.5572068031392857, 0.7897370178925692], # grey
            [0.6061187523676671, 0.02352376025211733, 0.125743623104707], # tiexiuhong
            [0.0983177267033869, 0.45167074814306607, 0.7772092130185153], # blue
            [0.5362068522201038, 0.4823186296727855, 0.2849664954418937], # zong
            [0.24243293467583527, 0.23770070635091722, 0.19855487487190848], # black
            [0.9327154191154875, 0.49505414043035667, 0.28626034672961975], # orange
            [0.10304475744738528, 0.5273551207376438, 0.22594274822347715], # mo lv
            [0.8515360053202375, 0.7461464262302233, 0.12688968916092613], # gold
            [0.6320228352842366, 0.8645799601845567, 0.7505235788900761], # dan lv
            [0.798475028594087, 0.3421010336796366, 0.37228899317607], # rou se
            [0.9798089671888133, 0.9192081421284964, 0.5962808711958855], # dan huang

            [0.44700521163871676, 0.9149047770645528, 0.2364668865826297], # light green
            ])


def catch_model2o3dmesh_Tpose(pModel, paint_flag = False, coord_mode = "xzy"):
    # load shape as Tpose
    q_ani_mesh = o3d.geometry.TriangleMesh()
    v_Arr = np.transpose(np.array(pModel.T)) # T is the T-pose of the model
    f_Arr = np.array(pModel.f)
    if coord_mode == "xzy": 
        q_ani_mesh.vertices = o3d.utility.Vector3dVector(v_Arr[:,[ 0,2,1]]) # v_Arr[:,[ 0,2,1]]
    elif coord_mode == "xyz": 
        q_ani_mesh.vertices = o3d.utility.Vector3dVector(v_Arr) # v_Arr[:,[ 0,2,1]]
    q_ani_mesh.triangles = o3d.utility.Vector3iVector(f_Arr)
    q_ani_mesh.compute_vertex_normals()
    if paint_flag:
        q_ani_mesh.paint_uniform_color([1, 0.706, 0])
    return q_ani_mesh

def catch_model2o3dmesh(pModel, paint_flag = False, coord_mode = "xzy"):
    q_ani_mesh = o3d.geometry.TriangleMesh()
    v_Arr = np.array(pModel.r)
    f_Arr = np.array(pModel.f)
    if coord_mode == "xzy": 
        q_ani_mesh.vertices = o3d.utility.Vector3dVector(v_Arr[:,[ 0,2,1]]) # v_Arr[:,[ 0,2,1]]
    elif coord_mode == "xyz": 
        q_ani_mesh.vertices = o3d.utility.Vector3dVector(v_Arr) # v_Arr[:,[ 0,2,1]]
    q_ani_mesh.triangles = o3d.utility.Vector3iVector(f_Arr)
    q_ani_mesh.compute_vertex_normals()
    if paint_flag:
        q_ani_mesh.paint_uniform_color([1, 0.706, 0])
    return q_ani_mesh

def catch_model_Joint2o3dpcd(pModel, coord_mode = "xzy"):
    q_ani_joint_pcd = o3d.geometry.PointCloud()
    j_Arr = np.array(pModel.J)
    if coord_mode == "xzy": 
        q_ani_joint_pcd.points = o3d.utility.Vector3dVector(j_Arr[:,[ 0,2,1]]) # v_Arr[[0,2,1]:]
    elif coord_mode == "xyz": 
        q_ani_joint_pcd.points = o3d.utility.Vector3dVector(j_Arr) # v_Arr[[0,2,1]:]
    # print("lens:", len(q_ani_joint_pcd.points))
    q_ani_joint_pcd.paint_uniform_color([1, 0, 0])
    return q_ani_joint_pcd

def creat_joint_as_sphereLst(pModel, coord_mode = "xzy", pRadius= 0.05):
    # shape (n_points, 3)
    q_ani_joint_pcd = o3d.geometry.PointCloud()
    j_Arr = np.array(pModel.J)
    if coord_mode == "xzy": 
        j_Arr_xzy =j_Arr[:,[ 0,2,1]]
    elif coord_mode == "xyz": 
        j_Arr_xzy =j_Arr
    q_sphere_Lst = []
    for idx, point_i_Arr in enumerate(j_Arr_xzy):
        sphere_i = o3d.geometry.TriangleMesh.create_sphere(radius= pRadius, resolution= 10)
        sphere_i.translate(point_i_Arr)
        sphere_i.compute_vertex_normals()
        sphere_i.paint_uniform_color(g_spl_color_Arr[idx])
        q_sphere_Lst.append(sphere_i)
    return q_sphere_Lst


def draw_Obj_Visible(obj3d_Lst, window_name, addFrame = False):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name= window_name, width = 800, height= 600, 
                    visible= True)
    for obj_i in obj3d_Lst:
        if type(obj_i) == list:
            for indj, obj_sub_pt_j in enumerate(obj_i):
                vis.add_geometry(obj_sub_pt_j)
        else:
            vis.add_geometry(obj_i)
    if addFrame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[-0.4, -0.1, 0.3])
        vis.add_geometry(mesh_frame)
    # change render mode
    rdr = vis.get_render_option()
    # rdr.mesh_show_back_face = True #  advised by Wang 08.18, 2019
    rdr.point_size = 11.0 # for dB6
    # rdr.mesh_color_option = o3d.visualization.MeshColorOption.ZCoordinate

    ctr = vis.get_view_control()
    ctr.scale(scale = 17)

    vis.run()
    vis.destroy_window()

