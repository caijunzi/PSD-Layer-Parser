"""
材质家族判别模块

基于指纹特征的规则判别（Stage 1）→ 后续可升级为 LightGBM（Stage 3）
5 类：金地屏风 / 宣纸水墨 / 绢本工笔 / 油画布 / 其他
"""
import numpy as np
from typing import Tuple, Dict


def classify_material_family(fingerprint: Dict) -> Tuple[str, float]:
    """
    材质家族判别（规则方式，Stage 1）
    
    Args:
        fingerprint: extract_fingerprint 返回的字典
    
    Returns:
        (family_name, confidence)
        family_name: "金地屏风" / "宣纸水墨" / "绢本工笔" / "油画布" / "其他"
        confidence: 0..1
    """
    # 边界检查：必需字段
    required = ["global_features", "texture_features", "structure_features"]
    if not all(k in fingerprint for k in required):
        return "其他", 0.1  # 降级兜底
    
    global_feats = fingerprint["global_features"]
    texture_feats = fingerprint["texture_features"]
    structure_feats = fingerprint["structure_features"]

    # 2026-09-15：四族**全部**改用中心主体区统计（此前用外框，扫描件外框多为灰底）。
    # 旧指纹无 center_* 键时回退到 bg 口径（向后兼容）。
    cL, ca, cb = global_feats.get("center_median_LAB", global_feats["bg_median_LAB"])
    csat = global_feats.get("center_saturation", global_feats["saturation"])
    edge_density = structure_feats["edge_density"]
    glcm_energy = texture_feats["glcm_energy"]

    # 规则判别（优先级从高到低）
    scores = {
        "金地屏风": _score_gold_screen(cL, ca, cb, csat, edge_density),
        "宣纸水墨": _score_ink_paper(cL, cb, csat, edge_density, glcm_energy),
        "绢本工笔": _score_silk_painting(cL, cb, csat, edge_density, glcm_energy),
        "油画布": _score_oil_canvas(cL, csat, edge_density),
        "其他": 0.3,  # 兜底分数
    }
    
    # 选择最高分
    family_name = max(scores, key=scores.get)
    confidence = scores[family_name]
    
    # 归一化置信度（确保 0..1）
    confidence = np.clip(confidence, 0.0, 1.0)
    
    return family_name, float(confidence)


def _score_gold_screen(bg_L: float, bg_a: float, bg_b: float, saturation: float, edge_density: float) -> float:
    """
    金地屏风：金黄地子 —— 高 b*（黄）+ 高亮度 + 近中性 a* + 中高饱和。

    标定（2026-09-15，source_4000.jpg 画心实测 L80 a5 b38 sat35）：
    旧规则对外框灰底（L37 b0）恒失配 → 金地被误判为油画布；改用中心区后
    以 b*（黄度）为主判据。
    """
    score = 0.0

    # 黄度（金地核心特征；标准 LAB 下金地 b* 可达 30–45）
    if bg_b > 30:
        score += 0.40
    elif bg_b > 20:
        score += 0.25
    elif bg_b > 10:
        score += 0.10

    # 亮度（金地明亮，但不若宣纸纯白）
    if bg_L >= 75:
        score += 0.30
    elif bg_L > 65:
        score += 0.15

    # a* 近中性（排除朱砂红、暖褐油画）
    if abs(bg_a) < 15:
        score += 0.15

    # 中高饱和度（金色有色度）
    if 20 < saturation < 55:
        score += 0.15

    # 装饰性边缘
    if 0.03 < edge_density < 0.10:
        score += 0.05

    return score


def _score_ink_paper(cL: float, cb: float, saturation: float,
                     edge_density: float, glcm_energy: float) -> float:
    """
    宣纸水墨：无色/淡黄纸地 + 中高亮度 + **稀疏**笔触（写意）+ 平滑墨韵。

    中心区口径重标定（2026-09-15）。与绢本工笔的区分点：水墨笔触更稀疏
    （edge_density、glcm 更低）；据此在实测样本上 ink 胜出。
    """
    score = 0.0

    # 纸地亮度（含淡墨区；中位数不必纯白）
    if cL > 82:
        score += 0.30
    elif cL > 60:
        score += 0.20

    # 无色或淡黄（排除金地 b*≈38 与浓郁设色）
    if cb < 12:
        score += 0.30
    elif cb < 20:
        score += 0.20

    # 稀疏笔触（写意）
    if edge_density < 0.10:
        score += 0.25
    elif edge_density < 0.16:
        score += 0.15

    # 平滑墨韵（纹理能量低）
    if glcm_energy < 0.22:
        score += 0.15
    elif glcm_energy < 0.28:
        score += 0.10

    return score


def _score_silk_painting(cL: float, cb: float, saturation: float,
                         edge_density: float, glcm_energy: float) -> float:
    """
    绢本工笔：米黄绢地 + 中亮度 + **细腻密集**（工笔）纹理。

    中心区口径重标定（2026-09-15）。关键区分点：绢本/织物纹理更密
    （edge_density ≥ 0.16 或 glcm ≥ 0.28），据此与写意水墨拉开差距。
    ⚠️ 无绢本真值样本，阈值为保守估计；待补样本后再校。
    """
    score = 0.0

    # 米黄绢地
    if 10 <= cb < 30:
        score += 0.20
    elif 6 <= cb < 36:
        score += 0.10

    # 中亮度
    if 65 < cL < 88:
        score += 0.20
    elif 58 < cL < 92:
        score += 0.10

    # 细腻密集（工笔/织物）——须显著高于水墨
    if edge_density >= 0.16:
        score += 0.30
    elif edge_density >= 0.12:
        score += 0.05

    if glcm_energy >= 0.28:
        score += 0.30
    elif glcm_energy >= 0.22:
        score += 0.05

    return score


def _score_oil_canvas(cL: float, saturation: float, edge_density: float) -> float:
    """
    油画布：多样亮度 + 高饱和（> 25）+ 高边缘密度（厚重笔触）。

    C3（2026-09-15）：统一到**中心区**口径（与其余三族一致）。
    ⚠️ 无油画布真值样本，阈值为保守沿用（未重标定）；待补样本后再校。
    """
    score = 0.0

    # 高饱和度（油画色彩浓郁）
    if saturation > 30:
        score += 0.4
    elif saturation > 20:
        score += 0.2

    # 高边缘密度（笔触厚重）
    if edge_density > 0.08:
        score += 0.3
    elif edge_density > 0.05:
        score += 0.15

    # 亮度多样（西式光影）
    if 30 < cL < 80:
        score += 0.2

    return score


def get_material_families() -> list:
    """
    返回支持的材质家族列表
    """
    return ["金地屏风", "宣纸水墨", "绢本工笔", "油画布", "其他"]


def explain_classification(fingerprint: Dict, family_name: str, confidence: float) -> str:
    """
    解释分类结果（调试用）
    
    Returns:
        人类可读的分类依据说明
    """
    global_feats = fingerprint["global_features"]
    texture_feats = fingerprint["texture_features"]
    structure_feats = fingerprint["structure_features"]
    
    bg_L, bg_a, bg_b = global_feats["bg_median_LAB"]
    saturation = global_feats["saturation"]
    edge_density = structure_feats["edge_density"]
    glcm_energy = texture_feats["glcm_energy"]
    
    explanation = f"分类结果：{family_name}（置信度 {confidence:.2f}）\n\n"
    explanation += f"关键特征：\n"
    explanation += f"  - 背景亮度（L*）: {bg_L:.1f}\n"
    explanation += f"  - 背景色偏（a*, b*）: ({bg_a:.1f}, {bg_b:.1f})\n"
    explanation += f"  - 饱和度: {saturation:.1f}\n"
    explanation += f"  - 边缘密度: {edge_density:.4f}\n"
    explanation += f"  - GLCM 能量: {glcm_energy:.3f}\n\n"
    
    explanation += f"判别依据：\n"
    if family_name == "金地屏风":
        explanation += f"  - 高亮度（L* > 80）✓\n" if bg_L > 80 else f"  - 高亮度（L* > 80）✗\n"
        explanation += f"  - 黄色偏移（b* > 20）✓\n" if bg_b > 20 else f"  - 黄色偏移（b* > 20）✗\n"
        explanation += f"  - 中等饱和度✓\n" if 15 < saturation < 40 else f"  - 中等饱和度✗\n"
    elif family_name == "宣纸水墨":
        explanation += f"  - 纸白（L* > 90）✓\n" if bg_L > 90 else f"  - 纸白（L* > 90）✗\n"
        explanation += f"  - 低饱和度（< 10）✓\n" if saturation < 10 else f"  - 低饱和度（< 10）✗\n"
        explanation += f"  - 稀疏笔触✓\n" if edge_density < 0.03 else f"  - 稀疏笔触✗\n"
    elif family_name == "绢本工笔":
        explanation += f"  - 中亮度（70 < L* < 85）✓\n" if 70 < bg_L < 85 else f"  - 中亮度（70 < L* < 85）✗\n"
        explanation += f"  - 中饱和度（10–25）✓\n" if 10 < saturation < 25 else f"  - 中饱和度（10–25）✗\n"
        explanation += f"  - 细腻笔触✓\n" if 0.05 < edge_density < 0.12 else f"  - 细腻笔触✗\n"
    elif family_name == "油画布":
        explanation += f"  - 高饱和度（> 25）✓\n" if saturation > 30 else f"  - 高饱和度（> 25）✗\n"
        explanation += f"  - 厚重笔触✓\n" if edge_density > 0.08 else f"  - 厚重笔触✗\n"
    else:
        explanation += f"  - 未匹配任何特定材质，归入兜底类\n"
    
    return explanation
