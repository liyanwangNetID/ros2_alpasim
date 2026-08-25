#!/usr/bin/env python3
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from profile_lane_change_geometry_features import adjacent_transition_stable,lane_start_indices,transition_direction,turn_direction_from_yaw
class Tests(unittest.TestCase):
 def test_persistence(self):
  record={'lane_matching':{'adjacent_transition_evidence':[{'corridor_point_count':8,'return_to_source':False}]}}
  stable,reasons=adjacent_transition_stable(record); self.assertTrue(stable,reasons)
 def test_lane_start_indices(self):
  rows=[{'target_lane_id':'B','target_point_index':7},{'target_lane_id':'C','target_point_index':15}]
  self.assertEqual(lane_start_indices(['A','B','C'],rows),{'A':0,'B':7,'C':15})
 def test_turn_direction_uses_actual_yaw(self):
  self.assertEqual(turn_direction_from_yaw([0,0.2],0),'left')
  self.assertEqual(turn_direction_from_yaw([0,-0.2],0),'right')
 def test_transition_direction(self):
  self.assertEqual(transition_direction('right_adjacent'),'right'); self.assertIsNone(transition_direction('successor'))
if __name__=='__main__': unittest.main(verbosity=2)
