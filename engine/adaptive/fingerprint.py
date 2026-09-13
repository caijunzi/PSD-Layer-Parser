"""
图级指纹提取模块

提取图像的全局特征、纹理特征、结构特征，并降维到 128 维（PCA）
用于材质家族判别和案例推理检索（Stage 4）
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional

# 延迟导入（避免未安装 scikit-learn 时加载失败）
_pca_model = None
_scaler = None


def extract_fingerprint(
    image: np.ndarray,
    background_mode: str = "paper"
) -> Dict:
    """
    提取图级指纹（128 维 PCA）
    
    Args:
        image: BGR 图像（np.ndarray）
        background_mode: 背景模式提示（"paper" / "gold_screen" / ...）
    
    Returns:
        {
            "embedding": np.ndarray(128,),          # PCA 降维后的指纹
            "global_features": {...},               # 全局特征（尺寸/色彩）
            "texture_features": {...},              # 纹理特征（LBP/GLCM）
            "structure_features": {...},            # 结构特征（边缘密度）
            "raw_features": np.ndarray,             # 原始特征向量（PCA 前）
        }
    """
    h, w = image.shape[:2]
    
    # 1. 全局特征
    global_feats = _extract_global_features(image, w, h)
    
    # 2. 纹理特征
    texture_feats = _extract_texture_features(image)
    
    # 3. 结构特征
    structure_feats = _extract_structure_features(image)
    
    # 4. 拼接原始特征向量
    raw_vector = np.concatenate([
        global_feats["vector"],
        texture_feats["vector"],
        structure_feats["vector"]
    ])
    
    # 5. PCA 降维到 128 维
    embedding_128d = _apply_pca(raw_vector)
    
    return {
        "embedding": embedding_128d,
        "global_features": global_feats,
        "texture_features": texture_feats,
        "structure_features": structure_feats,
        "raw_features": raw_vector,
        "image_shape": (h, w),
    }


def _extract_global_features(image: np.ndarray, w: int, h: int) -> Dict:
    """
    全局特征：尺寸、长宽比、色彩直方图（LAB 空间）、背景 median 色
    """
    # 尺寸特征
    aspect_ratio = w / h if h > 0 else 1.0
    megapixels = (w * h) / 1e6
    
    # 色彩特征（LAB 空间）
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # OpenCV 的 BGR2LAB 返回 L∈[0,255]、a/b∈[0,255]（中性灰 a=b=128）；
    # 但材质判别使用的是「标准 LAB」语义：L∈[0,100]、a/b∈[-128,127]（中性=0）。
    # 这里把 median/mean/饱和度换算到标准 LAB，直方图保留 OpenCV 原值
    # （避免扰动 PCA 原始特征口径）。否则所有图 bg_b≈128 都会误触发金地屏风规则。
    lab_std = lab.astype(np.float32)
    lab_std[:, :, 0] = lab_std[:, :, 0] * (100.0 / 255.0)
    lab_std[:, :, 1] = lab_std[:, :, 1] - 128.0
    lab_std[:, :, 2] = lab_std[:, :, 2] - 128.0

    # 背景 median 色（假设边缘 10% 为背景）
    margin_h, margin_w = int(h * 0.1), int(w * 0.1)
    bg_mask = np.zeros((h, w), dtype=bool)
    bg_mask[:margin_h, :] = True  # 上边缘
    bg_mask[-margin_h:, :] = True  # 下边缘
    bg_mask[:, :margin_w] = True  # 左边缘
    bg_mask[:, -margin_w:] = True  # 右边缘

    bg_pixels = lab_std[bg_mask]
    bg_median_L, bg_median_a, bg_median_b = np.median(bg_pixels, axis=0)

    # 全图色彩统计（标准 LAB 各通道）
    L_mean, a_mean, b_mean = np.mean(lab_std, axis=(0, 1))
    L_std, a_std, b_std = np.std(lab_std, axis=(0, 1))

    # 饱和度（标准 LAB 的 a*/b* 欧氏范数；中性灰≈0）
    saturation = np.sqrt(a_mean**2 + b_mean**2)

    # 色彩直方图（OpenCV LAB 原值，16 bins/通道）
    hist_L = cv2.calcHist([lab], [0], None, [16], [0, 256]).flatten()
    hist_a = cv2.calcHist([lab], [1], None, [16], [0, 256]).flatten()
    hist_b = cv2.calcHist([lab], [2], None, [16], [0, 256]).flatten()
    
    # 归一化直方图
    hist_L /= (hist_L.sum() + 1e-8)
    hist_a /= (hist_a.sum() + 1e-8)
    hist_b /= (hist_b.sum() + 1e-8)
    
    # 拼接向量（7 + 16*3 = 55 维）
    vector = np.array([
        aspect_ratio,
        megapixels,
        bg_median_L, bg_median_a, bg_median_b,
        saturation,
        L_std,
        *hist_L, *hist_a, *hist_b
    ], dtype=np.float32)
    
    return {
        "vector": vector,
        "aspect_ratio": aspect_ratio,
        "megapixels": megapixels,
        "bg_median_LAB": (bg_median_L, bg_median_a, bg_median_b),
        "saturation": saturation,
    }


def _extract_texture_features(image: np.ndarray) -> Dict:
    """
    纹理特征：LBP 直方图 + GLCM 能量
    """
    try:
        from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
    except ImportError:
        # scikit-image 未安装，返回零向量
        return {
            "vector": np.zeros(24, dtype=np.float32),  # LBP(16) + GLCM(8)
            "lbp_hist": np.zeros(16, dtype=np.float32),
            "glcm_energy": 0.0,
        }
    
    # 转灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 下采样到 512px（加速）
    h, w = gray.shape
    if max(h, w) > 512:
        scale = 512 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    # LBP（局部二值模式）
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 16), density=True)
    
    # GLCM（灰度共生矩阵）
    # 量化到 16 级灰度（加速）
    gray_q = (gray // 16).astype(np.uint8)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray_q, distances=distances, angles=angles, levels=16, symmetric=True, normed=True)
    
    # 提取 GLCM 特征（能量/对比度/同质性/相关性）
    energy = graycoprops(glcm, 'energy').flatten()
    contrast = graycoprops(glcm, 'contrast').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    
    glcm_feats = np.concatenate([energy[:2], contrast[:2], homogeneity[:2], correlation[:2]])  # 8 维（对称取前 2 个角度）
    
    # 拼接向量（16 + 8 = 24 维）
    vector = np.concatenate([lbp_hist, glcm_feats]).astype(np.float32)
    
    return {
        "vector": vector,
        "lbp_hist": lbp_hist,
        "glcm_energy": float(energy.mean()),
    }


def _extract_structure_features(image: np.ndarray) -> Dict:
    """
    结构特征：边缘密度、连通域统计、文本区粗定位
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 边缘密度（Canny）
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.sum() / (h * w * 255)
    
    # 连通域统计（二值化后）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n_components, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # 连通域数量（去除背景）
    n_components_fg = n_components - 1
    
    # 连通域面积统计（归一化）
    if n_components_fg > 0:
        areas = stats[1:, cv2.CC_STAT_AREA]  # 去除背景（label=0）
        area_mean = areas.mean() / (h * w)
        area_std = areas.std() / (h * w)
    else:
        area_mean = 0.0
        area_std = 0.0
    
    # 文本区粗定位（高纵横比连通域占比）
    text_like_count = 0
    if n_components_fg > 0:
        for i in range(1, n_components):
            x, y, w_cc, h_cc, area = stats[i]
            aspect = max(w_cc, h_cc) / (min(w_cc, h_cc) + 1e-8)
            if 3 < aspect < 20 and area > 50:  # 高纵横比 + 小面积
                text_like_count += 1
    text_density = text_like_count / (n_components_fg + 1)
    
    # 拼接向量（5 维）
    vector = np.array([
        edge_density,
        n_components_fg / 1000,  # 归一化
        area_mean,
        area_std,
        text_density,
    ], dtype=np.float32)
    
    return {
        "vector": vector,
        "edge_density": edge_density,
        "n_components": n_components_fg,
        "text_density": text_density,
    }


def _apply_pca(raw_vector: np.ndarray, n_components: int = 128) -> np.ndarray:
    """
    PCA 降维到 128 维（延迟导入 sklearn，避免未安装时报错）
    
    注意：首次调用会初始化 PCA 模型（需要训练集）
    """
    global _pca_model, _scaler
    
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        # sklearn 未安装，返回截断或填充的向量
        if len(raw_vector) >= n_components:
            return raw_vector[:n_components]
        else:
            return np.pad(raw_vector, (0, n_components - len(raw_vector)), constant_values=0)
    
    # 首次调用：初始化 PCA 模型（这里用恒等映射占位，真实训练在 Stage 1.2 后补充）
    if _pca_model is None:
        # TODO: 用 4 个预设的示例图 + 5 张锚图训练 PCA
        # 当前占位：用一个「恒等标准化」（mean=0, scale=1）的 StandardScaler，
        # 直接截断/填充到 n_components 维。注意：必须显式赋值 mean_/scale_，
        # 否则未 fit 的 StandardScaler 没有这两个属性，读取时会抛 AttributeError。
        _scaler = StandardScaler()
        _scaler.mean_ = np.zeros(len(raw_vector), dtype=np.float64)
        _scaler.scale_ = np.ones(len(raw_vector), dtype=np.float64)
        _pca_model = "placeholder"  # 标记已初始化

    # 标准化（动态适配原始特征维度；维度变化理论上不会发生，保险起见重建）
    if len(_scaler.mean_) != len(raw_vector):
        _scaler.mean_ = np.zeros(len(raw_vector), dtype=np.float64)
        _scaler.scale_ = np.ones(len(raw_vector), dtype=np.float64)

    scaled = (raw_vector - _scaler.mean_) / (_scaler.scale_ + 1e-8)

    # PCA（占位：截断或填充）
    if len(scaled) >= n_components:
        return scaled[:n_components]
    else:
        return np.pad(scaled, (0, n_components - len(scaled)), constant_values=0)


def train_pca_from_dataset(image_paths: list, n_components: int = 128):
    """
    从数据集训练 PCA 模型（供 Stage 1.2 后补充调用）
    
    Args:
        image_paths: 训练图像路径列表（4 个预设示例 + 5 张锚图）
        n_components: PCA 降维维度（默认 128）
    """
    global _pca_model, _scaler
    
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("[WARN] scikit-learn 未安装，跳过 PCA 训练")
        return
    
    # 提取所有图像的原始特征
    raw_features = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        fingerprint = extract_fingerprint(img)
        raw_features.append(fingerprint["raw_features"])
    
    if len(raw_features) < 2:
        print("[WARN] 训练图像不足 2 张，跳过 PCA 训练")
        return
    
    X = np.vstack(raw_features)
    
    # 标准化
    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X)
    
    # PCA
    _pca_model = PCA(n_components=n_components)
    _pca_model.fit(X_scaled)
    
    print(f"[OK] PCA 训练完成：{len(image_paths)} 张图像 -> {n_components} 维")
    print(f"     方差保留率：{_pca_model.explained_variance_ratio_.sum():.2%}")
