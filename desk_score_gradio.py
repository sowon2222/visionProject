import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from pydantic import ConfigDict

model_config = ConfigDict(arbitrary_types_allowed=True)

NAMES = ['NOTEBOOK', 'paper', 'pen', 'post-it', 'bottle', 'cup', 'laptop', 'mouse', 'keyboard']

organizer_classes = {"NOTEBOOK", "book"}
unnecessary_classes = {"paper", "post-it"}
device_sets = [{"laptop", "keyboard", "mouse"}]
messy_classes = {"paper", "post-it", "bottle"}

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def extract_visual_features(image_gray, image):
    h, w = image_gray.shape
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    edge_strength = np.mean(np.abs(laplacian))

    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256]*3)
    hist_norm = hist / hist.sum()
    color_entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-6))

    block_size = 50
    block_var = []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image_gray[y:y+block_size, x:x+block_size]
            if block.size > 0:
                block_var.append(np.var(block))
    mean_variance = np.mean(block_var)

    edges = cv2.Canny(image_gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)

    blurred = cv2.GaussianBlur(image_gray, (9, 9), 0)
    texture_diff = np.mean(np.abs(image_gray - blurred))

    return edge_strength, color_entropy, mean_variance, contour_count, texture_diff

def compute_combined_overlap_score(iou_scores, edge_map, detections, image_shape):
    n = len(detections)
    total_pairs = n * (n - 1) / 2 if n > 1 else 0
    if total_pairs == 0:
        return 0.0

    iou_overlap_count = np.sum(iou_scores >= 0.5) / 2
    iou_overlap_ratio = iou_overlap_count / total_pairs

    edge_overlap_score = 0
    for i in range(n):
        for j in range(i + 1, n):
            if iou_scores[i, j] >= 0.5:
                box1 = detections[i]['box']
                box2 = detections[j]['box']
                x1 = int(max(box1[0], box2[0]))
                y1 = int(max(box1[1], box2[1]))
                x2 = int(min(box1[2], box2[2]))
                y2 = int(min(box1[3], box2[3]))
                if x2 > x1 and y2 > y1:
                    overlap_region = edge_map[y1:y2, x1:x2]
                    edge_density = np.mean(overlap_region) / 255
                    edge_overlap_score += edge_density

    edge_overlap_score /= total_pairs
    combined_score = 0.7 * iou_overlap_ratio + 0.3 * edge_overlap_score
    return combined_score

def is_aligned(points, tolerance_deg=15):
    # points: [(x1, y1), (x2, y2), (x3, y3)]
    if len(points) < 3:
        return False
    a, b, c = np.array(points[0]), np.array(points[1]), np.array(points[2])
    v1 = b - a
    v2 = c - b
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return False
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
    return abs(angle) < tolerance_deg or abs(angle - 180) < tolerance_deg

def score_desk(detections, image_gray, image):
    score = 80
    feedback = []
    total_pairs = len(detections) * (len(detections) - 1) / 2 if len(detections) > 1 else 0

    # 중복 물건 체크
    item_counts = {}
    for d in detections:
        item_counts[d['name']] = item_counts.get(d['name'], 0) + 1
    
    duplicate_items = {item: count for item, count in item_counts.items() if count > 1}
    if duplicate_items:
        score -= 10
        feedback.append(f"📌 같은 물건이 여러 개 있습니다: {', '.join([f'{item}({count}개)' for item, count in duplicate_items.items()])} (감점 -10)")
        feedback.append("💡 정리 팁: 같은 종류의 물건은 하나만 책상 위에 두고, 나머지는 정리하세요.")

    # 지저분한 물건 체크
    messy_items = [d['name'] for d in detections if d['name'] in messy_classes]
    if messy_items:
        score -= 15
        feedback.append(f"📌 지저분한 물건이 감지되었습니다: {', '.join(messy_items)} (감점 -15)")
        feedback.append("💡 정리 팁: 불필요한 종이, 포스트잇, 물병을 정리하거나 파일에 보관하세요.")

    # 물건 간 겹침 체크 (새로운 방식)
    n = len(detections)
    iou_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                iou_matrix[i, j] = compute_iou(detections[i]['box'], detections[j]['box'])

    edge_map = cv2.Canny(image_gray, 100, 200)
    combined_overlap_score = compute_combined_overlap_score(iou_matrix, edge_map, detections, image_gray.shape)

    # 복합 겹침 점수 및 시각적 복잡도 보정
    if combined_overlap_score > 0.15:
        score -= 10
        feedback.append("📌 물건 간 겹침이 심하고 어수선해 보입니다. (감점 -10)")
        feedback.append("💡 정리 팁: 겹쳐진 물건들을 분리하고 여백을 확보하세요.")
    else:
        edge_strength, color_entropy, mean_variance, contour_count, texture_diff = extract_visual_features(image_gray, image)
        if edge_strength > 10 or contour_count > 1000:
            score -= 7
            feedback.append("📌 겹침은 적지만, 책상이 시각적으로 어수선합니다. (감점 -7)")
            feedback.append("💡 정리 팁: 물건을 정렬하거나, 불필요한 물건을 치워보세요.")

    # 전자기기 정렬 체크
    centers = {d['name']: ((d['box'][0]+d['box'][2])/2, (d['box'][1]+d['box'][3])/2) for d in detections}

    # 키보드-마우스 정렬 (아주 관대하게, 더 널널하게)
    if 'keyboard' in centers and 'mouse' in centers:
        k, m = centers['keyboard'], centers['mouse']
        dx = abs(k[0] - m[0])
        dy = abs(k[1] - m[1])
        if dy < 120 or dx < 200:
            score += 5
            feedback.append('✅ 키보드와 마우스가 잘 정렬되어 있습니다. (가점 +5)')
        else:
            score -= 2  # 감점도 줄여줌
            feedback.append('📌 키보드와 마우스가 다소 떨어져 있습니다. (감점 -2)')
            feedback.append('💡 정리 팁: 키보드와 마우스를 좀 더 가까이 놓아보세요.')

    # 주요 전자기기 정렬 체크 (관대한 버전)
    main_items = []
    for name in ['laptop', 'keyboard', 'mouse']:
        if name in centers:
            main_items.append((name, centers[name]))

    if len(main_items) == 3:
        points = [item[1] for item in main_items]

        def loose_alignment(p):
            x_coords, y_coords = zip(*p)
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            # 가로 또는 세로로 대략 비슷한 선상에 있으면 OK
            return x_range < 400 or y_range < 150

        if loose_alignment(points):
            score += 5
            feedback.append('✅ 전자기기 배치가 대체로 잘 정렬되어 있습니다. (가점 +5)')
        else:
            score -= 2
            feedback.append('📌 전자기기 위치가 조금 어수선합니다. (감점 -2)')
            feedback.append('💡 정리 팁: 키보드와 마우스를 노트북과 비슷한 선상에 배치해보세요.')

    # 물건 수 체크
    if len(detections) >= 8:
        score -= 5
        feedback.append("📌 탐지 객체 수가 많습니다. (감점 -5)")
        feedback.append("💡 정리 팁: 현재 사용하지 않는 물건들은 책상에서 치우고, 필요한 것만 남겨두세요.")

    # 전자기기 세트 체크
    detected_names = set([d['name'] for d in detections])
    for device in device_sets:
        if device.issubset(detected_names):
            score += 5
            feedback.append("✅ 전자기기 세트 감지됨 (가점 +5)")
            break

    # 물건 간 겹침 체크 (중간)
    iou_mid = sum(1 for i in range(len(detections)) for j in range(i+1, len(detections))
                  if compute_iou(detections[i]['box'], detections[j]['box']) >= 0.3)
    if total_pairs > 0 and (iou_mid / total_pairs) <= 0.1:
        score += 5
        feedback.append("✅ 탐지된된 물건 간 겹침이 거의 없습니다. (가점 +5)")

    # 시각적 복잡도 기반 점수 계산
    edge_score = np.clip((15 - edge_strength) / 15, 0, 1)
    entropy_score = np.clip((4.0 - color_entropy) / 4.0, 0, 1)
    variance_score = np.clip((3000 - mean_variance) / 3000, 0, 1)
    contour_score = np.clip((1600 - contour_count) / 1600, 0, 1)
    texture_score = np.clip((90 - texture_diff) / 90, 0, 1)

    vision_score = (
        0.2 * edge_score +
        0.2 * entropy_score +
        0.2 * variance_score +
        0.2 * contour_score +
        0.2 * texture_score
    )
    vision_score_scaled = (vision_score - 0.5) * 20  # -10~+10점
    score += int(vision_score_scaled)

    print(f"vision_score_scaled: {vision_score_scaled}")
    if vision_score_scaled > 3:
        feedback.append(f"✅ 전반적으로 깔끔한 이미지입니다 (가점 +{int(vision_score_scaled)}점)")
    elif vision_score_scaled < -1:
        abs_score = int(-vision_score_scaled)
        if abs_score <= 3:
            msg = "조금 어수선합니다"
        elif abs_score <= 6:
            msg = "꽤 어수선합니다"
        elif abs_score <= 9:
            msg = "매우 어수선합니다"
        else:
            msg = "심각하게 어수선합니다"
        feedback.append(f"📌 이미지 전반이 {msg} (감점 {abs_score}점)")
        feedback.append("💡 정리 팁: 물건들을 카테고리별로 분류하고, 비슷한 물건끼리 모아두세요. (배경도 평가에 포함될 수 있습니다. 책상만 보이게 찍어주세요.)")
    else:
        feedback.append("🟰 이미지 복잡도는 평균 수준입니다 (±0점)")

    # 물건 차지 비율
    if len(detections) > 0:
        img_h, img_w = image_gray.shape
        object_area = 0
        for d in detections:
            x1, y1, x2, y2 = map(int, d['box'])
            x1 = np.clip(x1, 0, img_w)
            x2 = np.clip(x2, 0, img_w)
            y1 = np.clip(y1, 0, img_h)
            y2 = np.clip(y2, 0, img_h)
            area = max(0, x2 - x1) * max(0, y2 - y1)
            object_area += area
        
        area_ratio = object_area / (img_h * img_w)
        if area_ratio > 0.40:
            score -= 10
            feedback.append(f"📌 물건이 너무 많은 자리를 차지하고 있어요! (감점 -10, 비율: {area_ratio:.1%})")
            feedback.append("💡 정리 팁: 책상의 40% 이상을 물건이 차지하고 있습니다. 불필요한 물건을 치우고 여백을 확보하세요.")
        elif area_ratio <= 0.3:
            score += 5
            feedback.append(f"✅ 물건이 적당한 공간만 차지하고 있어요! (가점 +5, 비율: {area_ratio:.1%})")
        else:
            feedback.append(f"🟰 물건 차지 비율이 평범합니다 (비율: {area_ratio:.1%})")

    return max(0, min(100, score)), feedback

def make_feedback(score, num_objects):
    msg = ""
    if score >= 90:
        msg += "🟢 책상이 매우 깔끔합니다.\n"
    elif score >= 70:
        msg += "🟡 정돈은 되었으나 개선이 필요합니다.\n"
    else:
        msg += "🔴 책상이 어지럽고 정리가 필요합니다.\n"
    if num_objects < 5:
        msg += "⚠️ 객체 수가 적어 신뢰도가 낮을 수 있습니다.\n"
    return msg

model = YOLO("last.pt")

def analyze_desk(image):
    try:
        results = model(image, conf=0.25, iou=0.1)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        detections = []
        img = image.copy()
        for box, cls, conf in zip(boxes, classes, confs):
            label = NAMES[cls] if cls < len(NAMES) else str(cls)
            detections.append({'name': label, 'box': box, 'conf': conf})
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        score, feedback_list = score_desk(detections, gray, image)
        feedback_msg = make_feedback(score, len(detections))
        feedback_detail = "\n".join(feedback_list)
        final_feedback = f"📊 점수: {score}점\n\n{feedback_msg}\n📋 상세 피드백:\n{feedback_detail}"
        return img, final_feedback
    except Exception as e:
        error_msg = f"처리 중 오류가 발생했습니다: {str(e)}"
        return image, error_msg

iface = gr.Interface(
    fn=analyze_desk,
    inputs=gr.Image(type="numpy", label="책상 사진 업로드"),
    outputs=[
        gr.Image(type="numpy", label="탐지 결과"),
        gr.Textbox(label="피드백 및 점수", lines=10)
    ],
    title="📏 책상 정돈 상태 평가",
    description="책상 사진을 업로드하면 YOLO + 시각적 복잡도 기반으로 정돈 점수와 피드백을 제공합니다.",
    examples=[],
    cache_examples=False
)

if __name__ == "__main__":
    iface.launch(share=True)
