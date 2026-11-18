from types import SimpleNamespace
import unittest
import numpy as np
from lidar_point_time import decode_cloud,deskew_to_header

class PointTimeTests(unittest.TestCase):
    def test_padded_rows_and_aux_fields_preserved(self):
        types={'x':(0,7),'y':(4,7),'z':(8,7),'intensity':(12,7),'ring':(16,4),'timestamp':(24,8)}
        dtype=np.dtype({'names':list(types),'formats':['<f4']*4+['<u2','<f8'],'offsets':[v[0] for v in types.values()],'itemsize':32})
        raw=bytearray(160);a=np.ndarray((2,2),dtype=dtype,buffer=raw,strides=(80,32))
        a['x']=[[1,2],[0,4]];a['intensity']=[[.1,.2],[.3,.4]];a['ring']=[[1,2],[3,4]];a['timestamp']=[[1.,1.01],[1.02,1.03]]
        m=SimpleNamespace(is_bigendian=False,fields=[SimpleNamespace(name=n,offset=o,datatype=t,count=1) for n,(o,t) in types.items()],point_step=32,height=2,width=2,row_step=80,data=raw)
        result=decode_cloud(m);np.testing.assert_array_equal(result['source_point_index'],[0,1,3]);np.testing.assert_array_equal(result['raw_ring'],[1,2,4]);self.assertEqual(result['raw_timestamp'][-1],1.03)
        np.testing.assert_array_equal(result['xyz'],result['xyz_raw'])
    def test_stationary_and_constant_velocity(self):
        t=np.array([0,1_000_000_000,2_000_000_000],dtype=np.int64);poses=np.zeros((3,7));poses[:,6]=1
        xyz=np.array([[2.,0,0],[3.,0,0]],dtype=np.float32)
        result,_=deskew_to_header(xyz,[.9,1.],1_000_000_000,0,t,poses,np.eye(4));np.testing.assert_allclose(result,xyz)
        poses[:,0]=[0,1,2];result,_=deskew_to_header(xyz,[.9,1.],1_000_000_000,0,t,poses,np.eye(4));np.testing.assert_allclose(result[:,0],[1.9,3.],atol=1e-6)
    def test_wrong_clock_and_missing_pose_not_clamped(self):
        t=np.array([0,1_000_000_000]);p=np.zeros((2,7));p[:,6]=1
        with self.assertRaises(ValueError):deskew_to_header([[2.,0,0]],[100.],1_000_000_000,0,t,p,np.eye(4))
        with self.assertRaises(ValueError):deskew_to_header([[2.,0,0]],[-.1],0,0,t,p,np.eye(4))

if __name__=='__main__':unittest.main()
