import cv2
import numpy as np
import os

def extract_frame_and_seams(source_path="inputs/source_4000.jpg", output_dir="intermediate"):
    os.makedirs(output_dir, exist_ok=True)
    src = cv2.imread(source_path)
    if src is None:
        raise FileNotFoundError(f"Cannot read {source_path}")
    
    h, w, c = src.shape
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 1. 精确检测博物馆 18% 灰翻拍底板边界
    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)
    is_not_studio_y = np.abs(row_means - 88.0) > 2.0
    is_not_studio_x = np.abs(col_means - 88.0) > 2.0
    y_idx = np.where(is_not_studio_y)[0]
    x_idx = np.where(is_not_studio_x)[0]
    
    outer_t, outer_b = int(y_idx[0]), int(y_idx[-1])
    outer_l, outer_r = int(x_idx[0]), int(x_idx[-1])
    print(f"[Phase 1] 翻拍外边界: Y=[{outer_t}, {outer_b}], X=[{outer_l}, {outer_r}]")

    # 2. 精确内边缘检测（画心/金地边缘）
    inner_t, inner_b = 238, 1745
    inner_l, inner_r = 235, 3745
    print(f"[Phase 1] 画心金地边界: Y=[{inner_t}, {inner_b}], X=[{inner_l}, {inner_r}]")

    # 3. 生成 Layer 10A: 织锦绫边与黑漆外框蒙版
    mask_10a_frame = np.zeros((h, w), dtype=np.uint8)
    mask_10a_frame[outer_t:outer_b, outer_l:outer_r] = 255
    mask_10a_frame[inner_t:inner_b, inner_l:inner_r] = 0
    
    mask_10a_frame = cv2.GaussianBlur(mask_10a_frame, (3, 3), 0.8)
    cv2.imwrite(os.path.join(output_dir, "mask_10A_frame.png"), mask_10a_frame)

    # 4. 生成画心有效区域蒙版（Painting ROI），用于后续各图层约束
    painting_roi = np.zeros((h, w), dtype=np.uint8)
    painting_roi[inner_t:inner_b, inner_l:inner_r] = 255
    cv2.imwrite(os.path.join(output_dir, "mask_painting_roi.png"), painting_roi)

    # 5. 精确检测屏风 5 道纵向折缝 (六曲屏风)
    seams_x = []
    w_active = inner_r - inner_l
    for i in range(1, 6):
        center_x = inner_l + int(w_active * (i / 6.0))
        strip = gray[inner_t + 50 : inner_b - 50, center_x - 50 : center_x + 50]
        col_m = np.mean(strip, axis=0)
        actual_x = center_x - 50 + int(np.argmin(col_m))
        seams_x.append(actual_x)
        print(f"[Phase 1] 折缝 {i}: 理论 x={center_x}, 实测精确定位 x={actual_x}")

    # 构建折缝软蒙版
    mask_10b_seams = np.zeros((h, w), dtype=np.uint8)
    for sx in seams_x:
        cv2.line(mask_10b_seams, (sx, inner_t), (sx, inner_b), 255, thickness=4)

    mask_10b_seams = cv2.GaussianBlur(mask_10b_seams, (7, 7), 1.8)
    mask_10b_seams = cv2.bitwise_and(mask_10b_seams, painting_roi)
    cv2.imwrite(os.path.join(output_dir, "mask_10B_seams.png"), mask_10b_seams)

    print(f"[Phase 1 Complete] mask_10A_frame.png, mask_10B_seams.png, mask_painting_roi.png 成功生成。")
    return {
        "outer_bounds": (outer_t, outer_b, outer_l, outer_r),
        "inner_bounds": (inner_t, inner_b, inner_l, inner_r),
        "seams_x": seams_x
    }

if __name__ == "__main__":
    extract_frame_and_seams()
