import torch
import torch.nn.functional as F
from conceptgraph.slam.instance_guard import enabled as instance_guard_enabled, compatible

from conceptgraph.slam.slam_classes import MapObjectList, DetectionList
from conceptgraph.utils.general_utils import Timer
from conceptgraph.utils.ious import (
    compute_iou_batch, 
    compute_giou_batch, 
    compute_3d_iou_accuracte_batch, 
    compute_3d_giou_accurate_batch,
)
from conceptgraph.slam.utils import (
    merge_obj2_into_obj1, 
    compute_overlap_matrix_2set
)

def compute_spatial_similarities(cfg, detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    Compute the spatial similarities between the detections and the objects
    
    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of spatial similarities
    '''
    det_bboxes = detection_list.get_stacked_values_torch('bbox')
    obj_bboxes = objects.get_stacked_values_torch('bbox')

    if cfg.spatial_sim_type == "iou":
        spatial_sim = compute_iou_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "giou":
        spatial_sim = compute_giou_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "iou_accurate":
        spatial_sim = compute_3d_iou_accuracte_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "giou_accurate":
        spatial_sim = compute_3d_giou_accurate_batch(det_bboxes, obj_bboxes)
    elif cfg.spatial_sim_type == "overlap":
        spatial_sim = compute_overlap_matrix_2set(cfg, objects, detection_list)
        spatial_sim = torch.from_numpy(spatial_sim).T
    else:
        raise ValueError(f"Invalid spatial similarity type: {cfg.spatial_sim_type}")
    
    return spatial_sim

def compute_visual_similarities(cfg, detection_list: DetectionList, objects: MapObjectList) -> torch.Tensor:
    '''
    Compute the visual similarities between the detections and the objects
    
    Args:
        detection_list: a list of M detections
        objects: a list of N objects in the map
    Returns:
        A MxN tensor of visual similarities
    '''
    det_fts = detection_list.get_stacked_values_torch('clip_ft') # (M, D)
    obj_fts = objects.get_stacked_values_torch('clip_ft') # (N, D)

    det_fts = det_fts.unsqueeze(-1) # (M, D, 1)
    obj_fts = obj_fts.T.unsqueeze(0) # (1, D, N)
    
    visual_sim = F.cosine_similarity(det_fts, obj_fts, dim=1) # (M, N)
    
    return visual_sim

def aggregate_similarities(cfg, spatial_sim: torch.Tensor, visual_sim: torch.Tensor) -> torch.Tensor:
    '''
    Aggregate spatial and visual similarities into a single similarity score
    
    Args:
        spatial_sim: a MxN tensor of spatial similarities
        visual_sim: a MxN tensor of visual similarities
    Returns:
        A MxN tensor of aggregated similarities
    '''
    if cfg.match_method == "sim_sum":
        sims = (1 + cfg.phys_bias) * spatial_sim + (1 - cfg.phys_bias) * visual_sim # (M, N)
    else:
        raise ValueError(f"Unknown matching method: {cfg.match_method}")

    if instance_guard_enabled(cfg):
        minimum = float(cfg.get('instance_min_spatial_overlap', 0.01))
        if not 0 < minimum <= 1:
            raise ValueError('instance_min_spatial_overlap must be in (0,1]')
        # Appearance alone cannot bridge disjoint geometry.
        sims = sims.masked_fill(~torch.isfinite(spatial_sim) | (spatial_sim < minimum), float('-inf'))
    
    return sims

def merge_detections_to_objects(
    cfg, 
    detection_list: DetectionList, 
    objects: MapObjectList, 
    agg_sim: torch.Tensor
) -> MapObjectList:
    if instance_guard_enabled(cfg):
        # Strongest association first. Later co-observed detections may use the
        # next compatible object, but never an object containing a distinct box
        # from this very frame. Newly appended objects are not in agg_sim.
        order = torch.argsort(agg_sim.max(dim=1).values, descending=True, stable=True)
        for i in order.tolist():
            target = None
            for j in torch.argsort(agg_sim[i], descending=True, stable=True).tolist():
                if not torch.isfinite(agg_sim[i,j]) or agg_sim[i,j] < float(cfg.get('sim_threshold', 0.0)):
                    break
                if compatible(cfg, objects[j], detection_list[i]):
                    target = j
                    break
            if target is None:
                objects.append(detection_list[i])
            else:
                objects[target] = merge_obj2_into_obj1(cfg, objects[target], detection_list[i], run_dbscan=False)
        return objects

    # Iterate through all detections and merge them into objects
    for i in range(agg_sim.shape[0]):
        # If not matched to any object, add it as a new object
        if agg_sim[i].max() == float('-inf'):
            objects.append(detection_list[i])
        # Merge with most similar existing object
        else:
            j = agg_sim[i].argmax()
            matched_det = detection_list[i]
            matched_obj = objects[j]
            merged_obj = merge_obj2_into_obj1(cfg, matched_obj, matched_det, run_dbscan=False)
            objects[j] = merged_obj
            
    return objects
