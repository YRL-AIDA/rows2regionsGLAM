import json

category_mapping = {
    1: 5,
    2: 1,
    3: 0,
    4: 1,
    5: 0,
    6: 0,
    7: 5,
    8: 2,
    9: 4,
    10: 1,
    11: 2,
    0: 0
}

# def merge_list_items(annotations, image_id, y_threshold=50, x_tolerance=0.1):
#     list_items = [anno for anno in annotations if anno['image_id'] == image_id and anno['category_id'] == 3]
#
#     if len(list_items) < 2:
#         return [], set()
#
#     list_items_sorted = sorted(list_items, key=lambda x: (x['bbox'][1], x['bbox'][0]))
#
#     merged_list_items = []
#     current_group = []
#     deleted_annotations = set()
#
#     prev_bbox = None
#     for item in list_items_sorted:
#         if prev_bbox is None:
#             current_group.append(item)
#         else:
#             y_distance = item['bbox'][1] - (prev_bbox[1] + prev_bbox[3])
#             x_distance = abs(round(item['bbox'][0], 1) - round(prev_bbox[0], 1))
#
#             if y_distance <= y_threshold and x_distance <= x_tolerance:
#                 current_group.append(item)
#             else:
#                 merged_list_items.append(current_group)
#                 current_group = [item]
#
#         prev_bbox = item['bbox']
#
#     if current_group:
#         merged_list_items.append(current_group)
#
#     merged_annotations = []
#     for group in merged_list_items:
#         merged_bbox = [
#             min([item['bbox'][0] for item in group]),
#             min([item['bbox'][1] for item in group]),
#             max([item['bbox'][0] + item['bbox'][2] for item in group]) - min([item['bbox'][0] for item in group]),
#             max([item['bbox'][1] + item['bbox'][3] for item in group]) - min([item['bbox'][1] for item in group])
#         ]
#
#         merged_annotations.append({
#             'image_id': image_id,
#             'category_id': 3,  # List
#             'bbox': merged_bbox,
#             'segmentation': merged_bbox,
#             'area': sum([item['area'] for item in group]),
#             'iscrowd': 0,
#             'precedence': 0
#         })
#
#         for item in group:
#             deleted_annotations.add(item['id'])
#
#     return merged_annotations, deleted_annotations


def transform_annotations(data, y_threshold=50, x_tolerance=0.1):
    transformed_annotations = []
    # merged_annotations = []
    # deleted_annotations = set()

    for anno in data['annotations']:
        category_id = anno['category_id']
        anno['category_id'] = category_mapping.get(category_id, 0)
        transformed_annotations.append(anno)

    # images = {image['id']: image for image in data['images']}
    # for image_id in images:
    #     new_merged, deleted = merge_list_items(transformed_annotations, image_id, y_threshold, x_tolerance)
    #     merged_annotations.extend(new_merged)
    #     deleted_annotations.update(deleted)
    #
    # transformed_annotations = [anno for anno in transformed_annotations if anno['category_id'] != 3 or anno['id'] not in deleted_annotations]
    #
    # transformed_annotations.extend(merged_annotations)

    data['annotations'] = transformed_annotations
    return data


with open('E:/Work/DocLayNet/train (old).json', 'r', encoding='utf-8') as f:
    data = json.load(f)

transformed_data = transform_annotations(data, y_threshold=50, x_tolerance=0.1)

with open('E:/Work/DocLayNet/train.json', 'w', encoding='utf-8') as f:
    json.dump(transformed_data, f, ensure_ascii=False, indent=4)

