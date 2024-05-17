def check_area(midpoint_x, midpoint_y, area_pointA, area_pointD, array_ids, id, label):
    line_y_min = min(area_pointA[1], area_pointD[1])
    line_y_max = max(area_pointA[1], area_pointD[1])

    # 자동차가 선을 통과하는지 확인
    if line_y_min <= midpoint_y <= line_y_max:
        midpoint_color = (0, 0, 255)
        print('Kategori : ' + str(label))

        # Add vehicles counting
        if len(array_ids) > 0:
            if label not in array_ids:
                array_ids.append(label)
        else:
            array_ids.append(label)
    return array_ids

midpoint_x = 342.0
midpoint_y = 697.5
area_pointA = (422, 407)
area_pointD = (382, 437)
array_ids = []
id=32
label='32:truck'
print(check_area(midpoint_x, midpoint_y, area_pointA, area_pointD, array_ids, id, label))