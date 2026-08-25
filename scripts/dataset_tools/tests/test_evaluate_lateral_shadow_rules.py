#!/usr/bin/env python3
import math
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from evaluate_lateral_shadow_rules import observed_proposal,propose_lateral


def frozen(action="change_lane_right",quality="usable"):
 return {"lateral":{"action":action,"quality_status":quality}}

class Tests(unittest.TestCase):
 def test_turn_candidate_normalized(self):
  action,reasons=observed_proposal([{"interpretation":"turn_left_candidate","interpretation_reason":"diverging"}])
  self.assertEqual(action,"turn_left"); self.assertEqual(reasons,["diverging"])
 def test_conflicting_observed_is_unknown(self):
  action,_=observed_proposal([{"interpretation":"turn_left_candidate"},{"interpretation":"keep_direction"}])
  self.assertEqual(action,"unknown")
 def test_in_progress_changes_keep(self):
  geometry={
   "observed_adjacent_transitions":[],
   "in_progress_candidates":[{
    "candidate":True,
    "direction":"left",
    "final_target_advantage_m":-1.0,
    "directional_heading_progress_deg":5.0,
   }],
   "inferred_in_progress_action":"change_lane_left",
  }
  lateral={
   "lateral":{
    "ego_total_yaw_change_rad":math.radians(5.0),
    "map_corridor_heading_change_rad":math.radians(1.0),
   }
  }
  proposal=propose_lateral(
   frozen("keep_direction"),
   geometry,
   lateral,
  )
  self.assertEqual(proposal["action"],"change_lane_left")
 def test_unknown_frozen_preserved(self):
  proposal=propose_lateral(frozen("unknown","unknown"),{"observed_adjacent_transitions":[{"interpretation":"turn_right_candidate"}]})
  self.assertEqual(proposal["action"],"unknown")
 def test_no_new_evidence_preserves(self):
  proposal=propose_lateral(frozen("turn_left"),{"observed_adjacent_transitions":[],"inferred_in_progress_action":None})
  self.assertEqual(proposal["action"],"turn_left")
if __name__=="__main__": unittest.main(verbosity=2)
