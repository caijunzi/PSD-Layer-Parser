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
    
    # 提取关键特征
    bg_L, bg_a, bg_b = global_feats["bg_median_LAB"]
    saturation = global_feats["saturation"]
    edge_density = structure_feats["edge_density"]
    glcm_energy = texture_feats["glcm_energy"]
    
    # 规则判别（优先级从高到低）
    scores = {
        "金地屏风": _score_gold_screen(bg_L, bg_a, bg_b, saturation, edge_density),
        "宣纸水墨": _score_ink_paper(bg_L, saturation, edge_density, glcm_energy),
        "绢本工笔": _score_silk_painting(bg_L, saturation, edge_density, glcm_energy),
        "油画布": _score_oil_canvas(bg_L, saturation, edge_density),
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
    金地屏风：高亮度（L* > 80）+ 高 b*（黄色 > 20）+ 中饱和
    """
    score = 0.0
    
    # 亮度判别（金地屏风 L* 普遍 ≥ 80，含边界 80）
    if bg_L >= 80:
        score += 0.4
    elif bg_L > 70:
        score += 0.2
    
    # 黄色偏移（b* > 20，金色特征）
    if bg_b > 20:
        score += 0.3
    elif bg_b > 10:
        score += 0.15
    
    # 饱和度（中等，15–40）
    if 15 < saturation < 40:
        score += 0.2
    
    # 边缘密度（装饰性强，中等偏高）
    if 0.02 < edge_density < 0.08:
        score += 0.1
    
    return score


def _score_ink_paper(bg_L: float, saturation: float, edge_density: float, glcm_energy: float) -> float:
    """
    宣纸水墨：高亮度（L* > 90）+ 低饱和（< 10）+ 低边缘密度
    """
    score = 0.0
    
    # 高亮度（纸白）
    if bg_L > 90:
        score += 0.4
    elif bg_L > 85:
        score += 0.2
    
    # 低饱和度（水墨无彩色）
    if saturation < 10:
        score += 0.3
    elif saturation < 15:
        score += 0.15
    
    # 低边缘密度（写意笔触稀疏）
    if edge_density < 0.03:
        score += 0.2
    elif edge_density < 0.05:
        score += 0.1
    
    # GLCM 能量低（纹理变化大）
    if glcm_energy < 0.15:
        score += 0.1
    
    return score


def _score_silk_painting(bg_L: float, saturation: float, edge_density: float, glcm_energy: float) -> float:
    """
    绢本工笔：中亮度（L* 70–85）+ 中饱和（10–25）+ 高边缘密度（细腻）
    """
    score = 0.0
    
    # 中亮度（绢本偏米黄）
    if 70 < bg_L < 85:
        score += 0.3
    elif 65 < bg_L < 90:
        score += 0.15
    
    # 中饱和度（设色工整）
    if 10 < saturation < 25:
        score += 0.3
    elif 8 < saturation < 30:
        score += 0.15
    
    # 高边缘密度（工笔细腻）
    if 0.05 < edge_density < 0.12:
        score += 0.2
    elif 0.03 < edge_density < 0.15:
        score += 0.1
    
    # GLCM 能量中等（纹理规整）
    if 0.1 < glcm_energy < 0.2:
        score += 0.1
    
    return score


def _score_oil_canvas(bg_L: float, saturation: float, edge_density: float) -> float:
    """
    油画布：多样亮度 + 高饱和（> 25）+ 高边缘密度（厚重笔触）
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
    if 30 < bg_L < 80:
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
