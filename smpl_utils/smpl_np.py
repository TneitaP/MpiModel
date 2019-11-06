# adapted from https://github.com/Lotayou/SMPL/blob/master/smpl_np.py
# 2019/11/04, ZimengZhao
'''
Comparison btw SMPL & SMAL

.verts(.r)    (6890, 3)   |  (3889, 3)
.faces(.f)   (13776, 3)   |  (7774,3)
.J               (24,3)   |  (33,3)
.beta             (10,)   |  (41,)
.pose            (24*3)   |  (33*3)

# Relation of (f, v, J)
mddel.faces are fixed;
model.J   =  model.J_regressor * mdoel.verts;     J_regressor.shape =  (n_J, n_vert)
mdoel.verts = model.weights * model.J;             weights.shape = (n_vert, n_J)

# Relation of v = F(T, beta, pose)
step0. vert = v_template; 

step1. add beta term(identity-dependent shape, varible<bete>)
        verts += shapedirs * beta   (called v_shaped)
        J = J_regressor * verts

step2. add theta term(non-rigid pose-dependent shape,  varible<pose>)
        
        vert += posedirs * lrotmin;  lrotmin <- R <-(rodrigues)--pose_cube <- pose



'''
import numpy as np
import pickle


class SMPLModel(object):
    def __init__(self, model_path, class_name):
        '''
        SMPL model.

        Parameter:
        ---------
        model_path: Path to the SMPL model parameters, pre-processed by
        `preprocess.py`.
        '''
        with open(model_path, 'rb') as f:
            params = pickle.load(f, encoding='latin1')
            # when calling the pkl load, there is a ch.Ch in the file, 
            # so the shumpy is still needed when file loading. 
            self.J_regressor = params['J_regressor'] # default <scipy.sparse.csc.csc_matrix>
            self.weights = params['weights'] 
            self.posedirs = params['posedirs']
            self.v_template = params['v_template']
            self.shapedirs = params['shapedirs'] # default is a <chumpy.ch.Ch>
            self.faces = params['f']
            self.kintree_table = params['kintree_table']
            

            # currently useless:
            if 'bs_type' in params:
                self.bs_type = params['bs_type'] # 'lrotmin'
            else:
                self.bs_type = 'lrotmin'
            if 'bs_type' in params:
                self.bs_style = params['bs_style'] # 'lbs'
            else:
                self.bs_style = 'lbs'

        id_to_col = {
            self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
        }
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }
        if class_name == "person":
            self.pose_shape = [24, 3]
            self.beta_shape = [10]
            self.trans_shape = [3]
        elif class_name == "animal":
            self.pose_shape = [33, 3]
            self.beta_shape = [41]
            self.trans_shape = [3]


        self.pose = np.zeros(self.pose_shape)
        self.beta = np.zeros(self.beta_shape)
        self.trans = np.zeros(self.trans_shape)

        self.verts = None
        self.J = None
        self.R = None

        self.update()

    def set_params(self, pose=None, beta=None, trans=None):
        '''
        Set pose, shape, and/or translation parameters of SMPL model. Verices of the
        model will be updated and returned.

        Prameters:
        ---------
        pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
        relative to parent joint. For root joint it's global orientation.
        Represented in a axis-angle format.

        beta: Parameter for model shape. A vector of shape [10]. Coefficients for
        PCA component. Only 10 components were released by MPI.

        trans: Global translation of shape [3].

        Return:
        ------
        Updated vertices.
        '''
        if pose is not None:
            self.pose = pose
        if beta is not None:
            self.beta = beta
        if trans is not None:
            self.trans = trans
        self.update()
        return self.verts
    
    def update(self):
        '''
        Called automatically when parameters are updated.
        '''
        # how beta affect body shape
        v_shaped = self.shapedirs.dot(self.beta) + self.v_template
        # joints location
        self.J = self.J_regressor.dot(v_shaped)
        pose_cube = self.pose.reshape((-1, 1, 3))
        # rotation matrix for each joint
        self.R = self.rodrigues(pose_cube) # (n_J, 3, 3)
        I_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            (self.R.shape[0]-1, 3, 3)
        ) # (n_J, ([[1,0], [0,1]]))
        lrotmin = (self.R[1:] - I_cube).ravel() # ((n_J-1)*3*3, )
        # how pose affect body shape in zero pose
        v_posed = v_shaped + self.posedirs.dot(lrotmin)
        # world transformation of each joint
        # formula(2)(3)(4) in SMPL paper
        G = np.empty((self.kintree_table.shape[1], 4, 4)) # (num_joint,4,4)
        G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
        for i in range(1, self.kintree_table.shape[1]):
            G[i] = G[self.parent[i]].dot(
                self.with_zeros(
                    np.hstack(
                        [self.R[i],((self.J[i, :]-self.J[self.parent[i],:]).reshape([3,1]))]
                    )
                )
            )
        # remove the transformation due to the rest pose
        G = G - self.pack(
            np.matmul(
                G,
                np.hstack([self.J, np.zeros([self.pose_shape[0], 1])]).reshape([self.pose_shape[0], 4, 1])
                )
            )
        # transformation of each vertex
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]

        self.verts = v + self.trans.reshape([1, 3])
        self.J = self.J_regressor.dot(self.verts) # add by Zimeng Zhao    
    def rodrigues(self, r):
        '''
        formula(1) in SMPL paper
        Rodrigues' rotation formula that turns axis-angle vector into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation vector of shape [batch_size, 1, 3].

        Return:
        -------
        Rotation matrix of shape [batch_size, 3, 3].

        '''
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(theta.dtype).tiny) # 'theta.dtype' edit by Zimeng Zhao    
        r_hat = r / theta
        cosTheta = np.cos(theta)
        sinTheta = np.sin(theta)

        z_stick = np.zeros(theta.shape[0])
        # m is the skew symmetric of r_hat
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
        ).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            [theta.shape[0], 3, 3]
        )
        # mm = m.dot(m)
        # A = np.transpose(r_hat, axes=[0, 2, 1])
        # B = r_hat
        # dot = np.matmul(A, B)
        # R = cosTheta * i_cube + (1 - cosTheta) * dot + sinTheta * m

        R = i_cube + (1 - cosTheta) * np.matmul(m,m) + sinTheta * m
        return R

    def with_zeros(self, x):
        '''
        Append a [0, 0, 0, 1] vector to a [3, 4] matrix.
        [R  | t]
        [ 0  |1]
        Parameter:
        ---------
        x: Matrix to be appended.

        Return:
        ------
        Matrix after appending of shape [4,4]

        '''
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

    def pack(self, x):
        '''
        Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
        manner.

        Parameter:
        ----------
        x: Matrices to be appended of shape [batch_size, 4, 1]

        Return:
        ------
        Matrix of shape [batch_size, 4, 4] after appending.

        '''
        return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

    def save_to_obj(self, path):
        '''
        Save the SMPL model into .obj file.

        Parameter:
        ---------
        path: Path to save.

        '''
        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def save_to_pkl_model(self, path):
        '''
        Save the SMPL model into .pkl file.

        Parameter:
        ---------
        path: Path to save.

        '''
        trainer_dict = {
            'v_template': np.asarray(self.verts), # make current self.verts as template.
            'J': np.asarray(self.J),
            'weights': np.asarray(self.weights),
            'kintree_table': self.kintree_table,
            'f': self.faces, 
            'bs_type': self.bs_type, 
            'posedirs': np.asarray(self.posedirs), 
            'shapedirs': np.asarray(self.shapedirs), # may lost, save to Np rather than Ch.
            'J_regressor': self.J_regressor,  # may lost
            'bs_style': self.bs_style # may lost
        }   
        pickle.dump(trainer_dict, open(path, 'wb'), -1)

if __name__ == '__main__':
    smpl = SMPLModel('./model.pkl')
    np.random.seed(9608)
    pose = (np.random.rand(*smpl.pose_shape) - 0.5) * 0.4
    beta = (np.random.rand(*smpl.beta_shape) - 0.5) * 0.06
    trans = np.zeros(smpl.trans_shape)
    smpl.set_params(beta=beta, pose=pose, trans=trans)
    smpl.save_to_obj('./smpl_np.obj')

