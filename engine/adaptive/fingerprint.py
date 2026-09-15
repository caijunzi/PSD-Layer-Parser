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
_pca_loaded = False  # 是否已尝试加载过持久化模型（避免每次调用都读盘）

# 已训练 PCA 模型的默认持久化位置（可用环境变量 ULS_PCA_MODEL 覆盖）。
# 2026-09-16：此前 _pca_model 恒为 "placeholder"、且 train_pca_from_dataset 生产零调用，
# 导致 embedding 实际是「未标准化的原始特征截断/填充」——各特征量纲差异巨大
# （如 edge_density≈0.1 与直方图计数≈1e3），余弦相似度被大尺度特征主导，CBR 检索质量受损。
# 注意：不要放在 checkpoints/ 或 models/ —— 二者被 .gitignore 忽略（用于 AI 权重），
# 放那里会导致 clone 后无模型、PCA 静默失效。本文件仅几 KB（84 维 mean/scale），随代码走。
DEFAULT_PCA_PATH = Path(__file__).resolve().parent / "fingerprint_pca.pkl"

# PCA 需要足够样本才有统计意义；低于该规模只做标准化（见 train_pca_from_dataset）
_MIN_SAMPLES_FOR_PCA = 20
_MIN_PCA_COMPONENTS = 16


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

    # 中心「主体区」统计（去掉外框 15%）——修复 2026-09-15：
    # 扫描件外框常是博物馆灰底/桌面，用它估计"背景材质"会完全失真
    # （实测 source_4000.jpg 外框 L37.3/b0.0，而画心金地是 L80/b38）。
    # 因此补充中心区统计，供材质判别优先使用；不加入 vector，避免改变 embedding 维度。
    my, mx = int(h * 0.15), int(w * 0.15)
    if (h - 2 * my) > 0 and (w - 2 * mx) > 0:
        center = lab_std[my:h - my, mx:w - mx]
    else:
        center = lab_std
    center_flat = center.reshape(-1, 3)
    c_med = np.median(center_flat, axis=0)
    center_median_L, center_median_a, center_median_b = (
        float(c_med[0]), float(c_med[1]), float(c_med[2]),
    )
    center_saturation = float(
        np.sqrt(np.mean(center[:, :, 1]) ** 2 + np.mean(center[:, :, 2]) ** 2)
    )

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
        # 中心主体区统计（材质判别优先使用；非 vector 成员，不影响 embedding）
        "center_median_LAB": (center_median_L, center_median_a, center_median_b),
        "center_saturation": center_saturation,
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


def _model_path(path: Optional[str] = None) -> Path:
    """PCA 模型路径：显式参数 > 环境变量 ULS_PCA_MODEL > 默认 checkpoints 位置。"""
    import os
    return Path(path or os.environ.get("ULS_PCA_MODEL") or DEFAULT_PCA_PATH)


def save_pca_model(path: Optional[str] = None) -> bool:
    """把当前已训练 scaler/PCA 持久化到磁盘。

    Returns:
        是否成功保存（未训练或无模型时返回 False）
    """
    global _pca_model, _scaler
    if _scaler is None or not hasattr(_scaler, "mean_"):
        return False
    try:
        import pickle
        p = _model_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            # pca 可能为 None（仅标准化的模型），加载侧已兼容
            pickle.dump({"scaler": _scaler, "pca": _pca_model,
                         "n_features": len(_scaler.mean_), "version": 1}, f)
        return True
    except Exception:
        return False


def load_pca_model(path: Optional[str] = None) -> bool:
    """从磁盘加载已训练模型（幂等；失败保持 None 不影响调用方）。

    Returns:
        是否成功加载
    """
    global _pca_model, _scaler
    p = _model_path(path)
    if not p.is_file():
        return False
    try:
        import pickle
        with p.open("rb") as f:
            obj = pickle.load(f)
        sc, pc = obj.get("scaler"), obj.get("pca")
        # scaler 必须有；pca 允许为 None（样本不足时只做标准化、不降维）
        if sc is None or not hasattr(sc, "mean_"):
            return False
        _scaler, _pca_model = sc, pc
        return True
    except Exception:
        return False


def _apply_pca(raw_vector: np.ndarray, n_components: int = 128) -> np.ndarray:
    """PCA 降维到 n_components 维（延迟导入 sklearn，避免未安装时报错）。

    2026-09-16 修复（PCA 真正生效）：
      - 首次调用会**尝试加载已训练模型**（`load_pca_model`）；加载成功则走
        「真实标准化 + PCA transform」，而不是此前的恒等占位。
      - 未训练/未安装 sklearn 时**回退**到旧的截断/填充行为（行为不变，不破坏既有单测）。
    """
    global _pca_model, _scaler, _pca_loaded

    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        # sklearn 未安装，返回截断或填充的向量
        if len(raw_vector) >= n_components:
            return raw_vector[:n_components]
        else:
            return np.pad(raw_vector, (0, n_components - len(raw_vector)), constant_values=0)

    # 首次调用：优先加载已训练模型；否则回退恒等占位（保持旧行为）
    if _pca_model is None and not _pca_loaded:
        _pca_loaded = True
        if not load_pca_model():
            _scaler = StandardScaler()
            _scaler.mean_ = np.zeros(len(raw_vector), dtype=np.float64)
            _scaler.scale_ = np.ones(len(raw_vector), dtype=np.float64)
            _pca_model = "placeholder"  # 标记已初始化（未训练）

    # 标准化（动态适配原始特征维度；维度变化理论上不会发生，保险起见重建）
    if _scaler is None or len(_scaler.mean_) != len(raw_vector):
        _scaler = StandardScaler()
        _scaler.mean_ = np.zeros(len(raw_vector), dtype=np.float64)
        _scaler.scale_ = np.ones(len(raw_vector), dtype=np.float64)
        _pca_model = "placeholder"

    scaled = (raw_vector - _scaler.mean_) / (_scaler.scale_ + 1e-8)

    # 已训练模型 → 真正 transform；占位 → 仅截断/填充
    if hasattr(_pca_model, "transform"):
        try:
            emb = _pca_model.transform(scaled.reshape(1, -1))[0]
        except Exception:
            emb = scaled
    else:
        emb = scaled

    if len(emb) >= n_components:
        return emb[:n_components]
    return np.pad(emb, (0, n_components - len(emb)), constant_values=0)


def train_pca_from_dataset(image_paths: list, n_components: int = 128,
                           save: bool = True, path: Optional[str] = None) -> bool:
    """从数据集训练 PCA 模型并（默认）持久化，供后续运行自动加载。

    2026-09-16 修复：
      - `n_components` **自适应**：PCA 的主成分数不能超过 min(样本数-1, 特征数)，
        否则 sklearn 直接报错（本项目原始特征仅 ~84 维，写死 128 必然失败）。
      - 训练后默认 `save_pca_model()` 落盘，使生产调用能真正加载使用
        （此前训练完不保存，等于没训练）。
      - 读图改用 `imread_unicode`，支持中文/带空格路径（cv2.imread 会静默返回 None）。

    Args:
        image_paths: 训练图像路径列表
        n_components: 期望降维维度（会自动收敛到合法上限）
        save: 是否持久化
        path: 持久化路径（默认 DEFAULT_PCA_PATH）

    Returns:
        是否训练成功
    """
    global _pca_model, _scaler, _pca_loaded

    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("[WARN] scikit-learn 未安装，跳过 PCA 训练")
        return False

    try:
        from engine.core.io_utils import imread_unicode
    except Exception:
        imread_unicode = None

    raw_features = []
    for img_path in image_paths:
        p = str(img_path)
        img = imread_unicode(p) if imread_unicode else cv2.imread(p)
        if img is None:
            img = cv2.imread(p)  # 兜底
        if img is None:
            print(f"[WARN] 读取失败，跳过: {p}")
            continue
        raw_features.append(extract_fingerprint(img)["raw_features"])

    if len(raw_features) < 2:
        print(f"[WARN] 训练图像不足 2 张（有效 {len(raw_features)}），跳过 PCA 训练")
        return False

    X = np.vstack(raw_features)
    n_samples, n_features = X.shape

    # 标准化始终训练——这是当前最大的实际缺陷（量纲差异让余弦相似度被大尺度特征主导）
    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X)
    _pca_loaded = True  # 已训练，避免 _apply_pca 再回退占位

    # PCA 仅在样本足够时才有统计意义：主成分数上限 min(样本数-1, 特征数)，
    # 样本太少（本项目 inputs 仅 9 张）时降维到个位数反而丢失信息 → 只标准化、不降维。
    k = int(min(n_components, n_features, max(1, n_samples - 1)))
    if n_samples >= _MIN_SAMPLES_FOR_PCA and k >= _MIN_PCA_COMPONENTS:
        _pca_model = PCA(n_components=k)
        _pca_model.fit(X_scaled)
        evr = float(_pca_model.explained_variance_ratio_.sum())
        print(f"[OK] 训练完成：{n_samples} 张图 / {n_features} 维原始特征 -> PCA {k} 维")
        print(f"     方差保留率：{evr:.2%}"
              + ("（注：原始特征维度 < 期望维度，已取上限）" if k < n_components else ""))
    else:
        _pca_model = None
        print(f"[OK] 训练完成：{n_samples} 张图 / {n_features} 维原始特征 -> 仅标准化（不降维）")
        print(f"     原因：样本数 {n_samples} < {_MIN_SAMPLES_FOR_PCA} 或可用主成分 {k} < "
              f"{_MIN_PCA_COMPONENTS}，PCA 在此规模无统计意义；"
              f"保留全部标准化特征比强行降维到 {k} 维更可靠。")

    if save:
        ok = save_pca_model(path)
        print(f"[{'OK' if ok else 'WARN'}] 模型持久化：{_model_path(path)}"
              + ("" if ok else "（保存失败，训练仅在当前进程有效）"))
    return True
