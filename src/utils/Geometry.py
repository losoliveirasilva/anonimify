def rectangle_area(rect):
    rect_w, rect_h = rect[1][0] - rect[0][0], rect[1][1] - rect[0][1]
    return rect_w * rect_h

def intersection_over_union(rect_a, rect_b):
    x = max(rect_a[0][0], rect_b[0][0]) - min(rect_a[1][0], rect_b[1][0])
    y = max(rect_a[0][1], rect_b[0][1]) - min(rect_a[1][1], rect_b[1][1])

    overlap_area = x * y

    return overlap_area / (rectangle_area(rect_a) + rectangle_area(rect_b) - overlap_area)

def intersection_area(rect_a, rect_b):
	xA = max(rect_a[0][0], rect_b[0][0])
	yA = max(rect_a[0][1], rect_b[0][1])
	xB = min(rect_a[1][0], rect_b[1][0])
	yB = min(rect_a[1][1], rect_b[1][1])

	return max(0, xB - xA + 1) * max(0, yB - yA + 1)

# def rectangle_area(rect):
#     rect_w, rect_h = rect[2] - rect[0], rect[3] - rect[1]
#     return rect_w * rect_h

# def intersection_over_union(rect_a, rect_b):
#     x = max(rec1[0], rec2[0]) - min(rec1[2], rec2[2])
#     y = max(rec1[1], rec2[1]) - min(rec1[3], rec2[3])

#     overlap_area = x * y

#     return overlap_area / (rectangle_area(rec1) + rectangle_area(rec2) - overlap_area)
