from pager import ImageSegment
import numpy as np


def graph_creat(segments):
    def fun_dist_bottom(seg1: ImageSegment, seg: ImageSegment):
        DIST = 3
        r1 = seg1.x_bottom_right
        r = seg.x_bottom_right
        l1 = seg1.x_top_left
        l = seg.x_top_left

        x1c, y1c = seg1.get_center()
        xc, yc = seg.get_center()
        if y1c > yc: # Только в одном направление
            return np.inf
        
        if abs(x1c-xc)+abs(y1c-yc) < DIST: # Если совпали
            return np.inf
        
        xd = min(abs(r1-r), abs(l1-l), abs(xc-x1c))
        yd = abs(y1c-yc) 
        
        if abs(r1-r) < DIST or abs(l1-l) < DIST or abs(xc-x1c) < DIST :
            return yd

        
        return xd+yd

    def fun_dist_right(seg1: ImageSegment, seg: ImageSegment):
        DIST = 3
        r1 = seg1.x_bottom_right
        r = seg.x_bottom_right
        l1 = seg1.x_top_left
        l = seg.x_top_left

        x1c, y1c = seg1.get_center()
        xc, yc = seg.get_center()
        if x1c > xc: # Только в одном направление
            return np.inf

        
        if abs(x1c-xc)+abs(y1c-yc) < DIST: # Если совпали
            return np.inf
        
        xd = min(abs(r1-l), abs(l1-r))
        yd = abs(y1c-yc) 

        h = (seg.height + seg1.height)/2
        if yd > 2*h:
            return np.inf
            
        
        return xd+yd

    dists_bottom = []
    for j, seg1 in enumerate(segments):
        dist_bottom = [fun_dist_bottom(seg1, seg) for seg in segments]
        if min(dist_bottom) == np.inf:
            continue
        k = int(np.argmin(dist_bottom))
        dists_bottom.append((min(j, k), max(j, k)))

    # dists_top = [(k, j) for j, k in dists_bottom]

    dists_right = []
    for j, seg1 in enumerate(segments):
        dist_right = [fun_dist_right(seg1, seg) for seg in segments]
        if min(dist_right) == np.inf:
            continue
        k = int(np.argmin(dist_right))
        dists_right.append((min(j, k), max(j, k)))

    # dists_left = [(k, j) for j, k in dists_right]

    all_edges = dists_bottom + dists_right
    all_edges = list(set(all_edges))
    return all_edges