'''
About this file:
================
Based on smpl_webuser/verts.py

This file defines the basic skinning modules for the SMPL loader which 
defines the effect of bones and blendshapes on the vertices of the template mesh.

Modules included:
- verts_decorated: 
  creates an instance of the SMPL model which inherits model attributes from another 
  SMPL model.
- verts_core: [overloaded function inherited by lbs.verts_core]
  computes the blending of joint-influences for each vertex based on type of skinning

'''

import chumpy
import scipy.sparse as sp
from chumpy.ch import MatVecMult

try:
    import smpl_utils.lbs as lbs
    import smpl_utils.posemapper as posemapper
except ImportError:
    import lbs
    import posemapper

def ischumpy(x): return hasattr(x, 'dterms')

def verts_decorated(trans, pose, 
    v_template, J, weights, kintree_table, bs_style, f,
    bs_type=None, posedirs=None, betas=None, shapedirs=None, want_Jtr=False):

    for which in [trans, pose, v_template, weights, posedirs, betas, shapedirs]:
        if which is not None:
            assert ischumpy(which)

    v = v_template

    if shapedirs is not None:
        if betas is None:
            betas = chumpy.zeros(shapedirs.shape[-1])
        v_shaped = v + shapedirs.dot(betas)
    else:
        v_shaped = v
        
    if posedirs is not None:
        v_posed = v_shaped + posedirs.dot(posemapper.posemap(bs_type)(pose))
    else:
        v_posed = v_shaped
        
    v = v_posed
        
    if sp.issparse(J):
        regressor = J
        # J_tmpx = MatVecMult(regressor, v_shaped[:,0])        
        # J_tmpy = MatVecMult(regressor, v_shaped[:,1])        
        # J_tmpz = MatVecMult(regressor, v_shaped[:,2])        
        # J = chumpy.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        # J = chumpy.array(regressor.dot(v_shaped))
        if not isinstance(regressor, chumpy.Ch):
            J = chumpy.dot(regressor.toarray(), v_shaped)
        else:
            J = chumpy.dot(regressor, v_shaped)
        # pData_Dic['J'] = ch.dot(pData_Dic['J_regressor'].toarray(), pData_Dic['v_shaped'])
        
        # pData_Dic['J'] = pData_Dic['J_regressor'].dot(pData_Dic['v_shaped']) # (33, 3889) * (3889, 1)
        # pData_Dic['J'] = ch.array(pData_Dic['J'])            
    else:    
        assert(ischumpy(J))
        
    assert(bs_style=='lbs')
    result, Jtr = lbs.verts_core(pose, v, J, weights, kintree_table, want_Jtr=True, xp=chumpy)
     
    tr = trans.reshape((1,3))
    result = result + tr
    Jtr = Jtr + tr

    result.trans = trans
    result.f = f
    result.pose = pose
    result.v_template = v_template
    result.J = J
    result.weights = weights
    result.kintree_table = kintree_table
    result.bs_style = bs_style
    result.bs_type =bs_type
    if posedirs is not None:
        result.posedirs = posedirs
        result.v_posed = v_posed
    if shapedirs is not None:
        result.shapedirs = shapedirs
        result.betas = betas
        result.v_shaped = v_shaped
    if want_Jtr:
        result.J_transformed = Jtr
    return result

def verts_core(pose, v, J, weights, kintree_table, bs_style, want_Jtr=False, xp=chumpy):
    
    if xp == chumpy:
        assert(hasattr(pose, 'dterms'))
        assert(hasattr(v, 'dterms'))
        assert(hasattr(J, 'dterms'))
        assert(hasattr(weights, 'dterms'))
     
    assert(bs_style=='lbs')
    result = lbs.verts_core(pose, v, J, weights, kintree_table, want_Jtr, xp)

    return result
