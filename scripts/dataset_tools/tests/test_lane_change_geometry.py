#!/usr/bin/env python3
import math,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from lane_change_geometry import *

def straight(y=0): return [{'x':x,'y':y} for x in range(41)]
def curved(y=0): return [{'x':x,'y':0.008*x*x+y} for x in range(41)]
class Tests(unittest.TestCase):
 def test_multiscale_curved_parallel(self):
  result=compare_parallel_corridors(curved(0),curved(3.5)); self.assertTrue(result.same_direction_parallel,result.reasons)
 def test_downstream_divergence(self):
  source=straight(0); target=[]
  for x in range(41): target.append({'x':x,'y':0 if x<8 else 0.05*(x-8)**2})
  result=compare_parallel_corridors(source,target); self.assertFalse(result.same_direction_parallel); self.assertTrue(result.diverging)
 def test_preference_switch(self):
  traj=[{'x':x,'y':3.5*x/20} for x in range(21)]
  result=distance_preference(traj,straight(0),straight(3.5)); self.assertTrue(result.confirmed_switch,result.reasons)
 def test_wiggle_no_switch(self):
  traj=[{'x':x,'y':0.1*math.sin(x)} for x in range(21)]
  result=distance_preference(traj,straight(0),straight(3.5)); self.assertFalse(result.confirmed_switch)
 def test_in_progress_left(self):
  traj=[{'x':x,'y':1.5*(x/20)**2} for x in range(21)]; yaws=[math.atan2(3*x/400,1) for x in range(21)]
  result=in_progress_lane_change_evidence(traj,yaws,straight(0),straight(3.5),'left'); self.assertTrue(result.candidate,result.reasons)
 def test_slice_projection(self):
  result=slice_polyline_from_projection(straight(0),{'x':15.5,'y':1},10); self.assertAlmostEqual(result[0].x,15.5); self.assertAlmostEqual(polyline_length(result),10)
if __name__=='__main__': unittest.main(verbosity=2)
