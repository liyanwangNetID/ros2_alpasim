#!/usr/bin/env python3
"""Corridor geometry for lane-change and turn arbitration.

The functions are independent of dataset I/O. Callers provide downstream
source/target corridor polylines already aligned to the same decision point.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

EPS = 1e-9

@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

@dataclass(frozen=True)
class Projection:
    x: float
    y: float
    distance_m: float
    signed_offset_m: float
    heading_rad: float
    arc_length_m: float

@dataclass(frozen=True)
class ParallelCorridorConfig:
    scales_m: tuple[float, ...] = (10.0, 20.0, 30.0)
    sample_spacing_m: float = 1.0
    minimum_required_scale_m: float = 10.0
    maximum_p90_heading_difference_deg: float = 10.0
    maximum_heading_difference_deg: float = 20.0
    minimum_median_separation_m: float = 1.5
    maximum_median_separation_m: float = 6.5
    maximum_separation_mad_m: float = 0.9
    maximum_endpoint_separation_change_m: float = 1.25

@dataclass(frozen=True)
class PreferenceConfig:
    minimum_initial_source_advantage_m: float = 0.25
    minimum_final_target_advantage_m: float = 0.25
    minimum_preference_change_m: float = 0.9
    minimum_target_preference_fraction_last_third: float = 0.6

@dataclass(frozen=True)
class InProgressLaneChangeConfig:
    minimum_trajectory_length_m: float = 8.0
    minimum_source_offset_progress_m: float = 0.35
    minimum_target_distance_reduction_m: float = 0.35
    minimum_monotonic_progress_fraction: float = 0.6
    minimum_directional_heading_progress_deg: float = 3.0
    maximum_directional_heading_progress_deg: float = 18.0
    minimum_final_target_advantage_m: float = -2.25

@dataclass(frozen=True)
class ScaleParallelEvidence:
    scale_m: float
    available: bool
    p90_heading_difference_deg: float | None
    maximum_heading_difference_deg: float | None
    median_separation_m: float | None
    separation_mad_m: float | None
    endpoint_separation_change_m: float | None
    parallel: bool
    reasons: tuple[str, ...]

@dataclass(frozen=True)
class ParallelCorridorEvidence:
    source_length_m: float
    target_length_m: float
    scales: tuple[ScaleParallelEvidence, ...]
    same_direction_parallel: bool
    diverging: bool
    reasons: tuple[str, ...]
    def to_dict(self) -> dict[str, Any]: return asdict(self)

@dataclass(frozen=True)
class PreferenceEvidence:
    initial_source_advantage_m: float
    final_target_advantage_m: float
    preference_change_m: float
    target_preference_fraction_last_third: float
    confirmed_switch: bool
    reasons: tuple[str, ...]
    source_distances_m: tuple[float, ...]
    target_distances_m: tuple[float, ...]
    def to_dict(self) -> dict[str, Any]: return asdict(self)

@dataclass(frozen=True)
class InProgressEvidence:
    direction: str
    trajectory_length_m: float
    source_offset_progress_m: float
    target_distance_reduction_m: float
    final_target_advantage_m: float
    monotonic_progress_fraction: float
    directional_heading_progress_deg: float
    candidate: bool
    reasons: tuple[str, ...]
    def to_dict(self) -> dict[str, Any]: return asdict(self)


def _point(value: Any) -> Point2D:
    if isinstance(value, Point2D): return value
    if isinstance(value, Mapping): return Point2D(float(value['x']), float(value['y']))
    return Point2D(float(value.x), float(value.y))

def points(values: Iterable[Any]) -> tuple[Point2D, ...]:
    result=tuple(_point(v) for v in values)
    if len(result)<2: raise ValueError('polyline requires at least two points')
    return result

def normalize_angle(v: float)->float: return (v+math.pi)%(2*math.pi)-math.pi
def angle_difference(a: float,b: float)->float: return normalize_angle(b-a)

def cumulative_lengths(values: Sequence[Any])->tuple[float,...]:
    p=points(values); out=[0.0]
    for a,b in zip(p,p[1:]): out.append(out[-1]+math.hypot(b.x-a.x,b.y-a.y))
    if out[-1]<=EPS: raise ValueError('zero-length polyline')
    return tuple(out)

def polyline_length(values: Sequence[Any])->float: return cumulative_lengths(values)[-1]

def sample_polyline(values: Sequence[Any], s: float)->tuple[Point2D,float]:
    p=points(values); cumulative=cumulative_lengths(p); q=min(max(float(s),0.0),cumulative[-1])
    for i,(a_s,b_s) in enumerate(zip(cumulative,cumulative[1:])):
        if q<=b_s+EPS:
            a,b=p[i],p[i+1]; length=b_s-a_s; ratio=0 if length<=EPS else (q-a_s)/length
            return Point2D(a.x+ratio*(b.x-a.x),a.y+ratio*(b.y-a.y)), math.atan2(b.y-a.y,b.x-a.x)
    a,b=p[-2],p[-1]; return b,math.atan2(b.y-a.y,b.x-a.x)

def project_to_polyline(values: Sequence[Any], q_value: Any)->Projection:
    p=points(values); q=_point(q_value); cumulative=cumulative_lengths(p); best=None
    for i,(a,b) in enumerate(zip(p,p[1:])):
        dx=b.x-a.x; dy=b.y-a.y; sq=dx*dx+dy*dy
        if sq<=EPS: continue
        r=min(1.0,max(0.0,((q.x-a.x)*dx+(q.y-a.y)*dy)/sq))
        x=a.x+r*dx; y=a.y+r*dy; ex=q.x-x; ey=q.y-y; heading=math.atan2(dy,dx)
        candidate=Projection(x,y,math.hypot(ex,ey),-math.sin(heading)*ex+math.cos(heading)*ey,heading,cumulative[i]+r*math.sqrt(sq))
        if best is None or candidate.distance_m<best.distance_m: best=candidate
    if best is None: raise ValueError('no valid segment')
    return best

def slice_polyline_from_projection(values: Sequence[Any], q_value: Any, maximum_length_m: float)->tuple[Point2D,...]:
    p=points(values); projection=project_to_polyline(p,q_value); cumulative=cumulative_lengths(p)
    result=[Point2D(projection.x,projection.y)]
    for i,s in enumerate(cumulative):
        if s>projection.arc_length_m+EPS: result.append(p[i])
    if len(result)<2: return tuple(result)
    return truncate_polyline(result,maximum_length_m)

def truncate_polyline(values: Sequence[Any], length_m: float)->tuple[Point2D,...]:
    p=points(values); cumulative=cumulative_lengths(p)
    if cumulative[-1]<=length_m+EPS: return p
    result=[]
    for point,s in zip(p,cumulative):
        if s<length_m-EPS: result.append(point)
        else: break
    endpoint,_=sample_polyline(p,length_m); result.append(endpoint)
    return tuple(result)

def append_polyline(base: Sequence[Any], extra: Sequence[Any])->tuple[Point2D,...]:
    a=list(points(base)); b=list(points(extra))
    if math.hypot(a[-1].x-b[0].x,a[-1].y-b[0].y)<0.5: b=b[1:]
    return tuple(a+b)

def _percentile(values: Sequence[float], q: float)->float:
    ordered=sorted(values); idx=max(0,min(len(ordered)-1,math.ceil(q*len(ordered))-1)); return ordered[idx]
def _mad(values: Sequence[float])->float:
    center=median(values); return median(abs(v-center) for v in values)

def compare_parallel_corridors(source: Sequence[Any], target: Sequence[Any], config: ParallelCorridorConfig=ParallelCorridorConfig())->ParallelCorridorEvidence:
    source_length=polyline_length(source); target_length=polyline_length(target); scales=[]
    for scale in config.scales_m:
        available=min(source_length,target_length)>=scale-0.5
        reasons=[]; p90=max_heading=med_sep=mad=endpoint=None
        if available:
            sample_count=max(5,int(math.ceil(scale/config.sample_spacing_m))+1)
            headings=[]; separations=[]
            for i in range(sample_count):
                s=scale*i/(sample_count-1); sp,sh=sample_polyline(source,s); tp,th=sample_polyline(target,s)
                headings.append(abs(math.degrees(angle_difference(sh,th))))
                separations.append(math.hypot(tp.x-sp.x,tp.y-sp.y))
            p90=_percentile(headings,0.9); max_heading=max(headings); med_sep=median(separations); mad=_mad(separations); endpoint=abs(separations[-1]-separations[0])
            if p90>config.maximum_p90_heading_difference_deg: reasons.append('heading_p90_too_large')
            if max_heading>config.maximum_heading_difference_deg: reasons.append('maximum_heading_too_large')
            if not config.minimum_median_separation_m<=med_sep<=config.maximum_median_separation_m: reasons.append('median_separation_out_of_range')
            if mad>config.maximum_separation_mad_m: reasons.append('separation_not_stable')
            if endpoint>config.maximum_endpoint_separation_change_m: reasons.append('endpoint_separation_change_too_large')
        else: reasons.append('scale_unavailable')
        scales.append(ScaleParallelEvidence(scale,available,p90,max_heading,med_sep,mad,endpoint,available and not reasons,tuple(reasons)))
    usable=[item for item in scales if item.available and item.scale_m>=config.minimum_required_scale_m]
    same=bool(usable) and all(item.parallel for item in usable)
    diverging=any(item.available and not item.parallel and ('heading_p90_too_large' in item.reasons or 'maximum_heading_too_large' in item.reasons or 'separation_not_stable' in item.reasons or 'endpoint_separation_change_too_large' in item.reasons) for item in scales)
    reasons=tuple(sorted({reason for item in scales for reason in item.reasons if reason!='scale_unavailable'}))
    return ParallelCorridorEvidence(source_length,target_length,tuple(scales),same,diverging,reasons)

def distance_preference(trajectory: Sequence[Any], source: Sequence[Any], target: Sequence[Any], config: PreferenceConfig=PreferenceConfig())->PreferenceEvidence:
    traj=points(trajectory); source_d=tuple(project_to_polyline(source,p).distance_m for p in traj); target_d=tuple(project_to_polyline(target,p).distance_m for p in traj)
    initial=target_d[0]-source_d[0]; final=source_d[-1]-target_d[-1]; change=(source_d[-1]-target_d[-1])-(source_d[0]-target_d[0])
    tail=max(1,len(traj)//3); fraction=sum(1 for sd,td in zip(source_d[-tail:],target_d[-tail:]) if td+0.05<sd)/tail
    reasons=[]
    if initial<config.minimum_initial_source_advantage_m: reasons.append('initial_source_advantage_too_small')
    if final<config.minimum_final_target_advantage_m: reasons.append('final_target_advantage_too_small')
    if change<config.minimum_preference_change_m: reasons.append('preference_change_too_small')
    if fraction<config.minimum_target_preference_fraction_last_third: reasons.append('target_preference_not_persistent')
    return PreferenceEvidence(initial,final,change,fraction,not reasons,tuple(reasons),source_d,target_d)

def in_progress_lane_change_evidence(trajectory: Sequence[Any], yaws: Sequence[float], source: Sequence[Any], target: Sequence[Any], direction: str, config: InProgressLaneChangeConfig=InProgressLaneChangeConfig())->InProgressEvidence:
    traj=points(trajectory)
    if len(traj)!=len(yaws): raise ValueError('trajectory/yaw length mismatch')
    sign=1.0 if direction=='left' else -1.0
    source_proj=[project_to_polyline(source,p) for p in traj]; target_proj=[project_to_polyline(target,p) for p in traj]
    offsets=[sign*p.signed_offset_m for p in source_proj]; target_d=[p.distance_m for p in target_proj]
    changes=[b-a for a,b in zip(offsets,offsets[1:])]; monotonic=sum(1 for c in changes if c>=-0.05)/len(changes)
    progress=offsets[-1]-offsets[0]; reduction=target_d[0]-target_d[-1]; advantage=source_proj[-1].distance_m-target_proj[-1].distance_m
    heading=sign*math.degrees(angle_difference(float(yaws[0]),float(yaws[-1])))
    length=sum(math.hypot(b.x-a.x,b.y-a.y) for a,b in zip(traj,traj[1:])); reasons=[]
    if length<config.minimum_trajectory_length_m: reasons.append('trajectory_too_short')
    if progress<config.minimum_source_offset_progress_m: reasons.append('source_offset_progress_too_small')
    if reduction<config.minimum_target_distance_reduction_m: reasons.append('target_distance_reduction_too_small')
    if advantage<config.minimum_final_target_advantage_m: reasons.append('target_still_too_far')
    if monotonic<config.minimum_monotonic_progress_fraction: reasons.append('progress_not_persistent')
    if heading<config.minimum_directional_heading_progress_deg: reasons.append('heading_progress_too_small')
    if heading>config.maximum_directional_heading_progress_deg: reasons.append('heading_progress_too_large_for_lane_change')
    return InProgressEvidence(direction,length,progress,reduction,advantage,monotonic,heading,not reasons,tuple(reasons))
