"""
材质家族判别模块

基于指纹特征的规则判别（Stage 1）→ 后续可升级为 LightGBM（Stage 3）
6 类：金地屏风 / 宣纸水墨 / 绢本工笔 / 油画布 / 织物壁布 / 其他

2026-09-16：新增「织物壁布」族 —— 此前无此族，`inputs/工艺壁布-*.jpeg`
（深灰/浅灰绗缝织物）被强行塞进最接近的绘画族，实测全被误判为「宣纸水墨」，
进而推荐到错误的 preset。织物（壁布/面料）的判据是**近中性**（见
`_score_fabric_wallcovering`），与四种绘画基材正交。
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

    # LBP 直方图熵：判「纹理是否**有结构**」——真实织物/绘画 ≥2，
    # 噪声与纯色 ≈0.8~1.4（局部二值模式几乎只有一个值）。仅织物族使用，不改指纹向量。
    lbp_entropy = None
    try:
        _lbp = np.asarray(texture_feats.get("lbp_hist"), dtype=np.float64)
        if _lbp.size:
            _p = _lbp / (_lbp.sum() + 1e-9)
            lbp_entropy = float(-(_p * np.log(_p + 1e-12)).sum())
    except Exception:
        lbp_entropy = None

    # 规则判别（优先级从高到低）
    scores = {
        "金地屏风": _score_gold_screen(cL, ca, cb, csat, edge_density),
        "宣纸水墨": _score_ink_paper(cL, cb, csat, edge_density, glcm_energy),
        "绢本工笔": _score_silk_painting(cL, cb, csat, edge_density, glcm_energy),
        "油画布": _score_oil_canvas(cL, csat, edge_density),
        "织物壁布": _score_fabric_wallcovering(cL, ca, cb, csat, edge_density,
                                              lbp_entropy=lbp_entropy),
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

    # 暖色、细腻且纹理明显的中心主体更符合绢本工笔；此时不能继续
    # 给宣纸水墨满分，否则真实绢本会被高 GLCM 误吸到水墨族。
    if 14 <= cb <= 24 and saturation >= 14 and glcm_energy >= 0.34 and edge_density >= 0.08:
        score -= 0.20

    return max(0.0, score)


def _score_silk_painting(cL: float, cb: float, saturation: float,
                         edge_density: float, glcm_energy: float) -> float:
    """
    绢本工笔：米黄绢地 + 中亮度 + **细腻密集**（工笔）纹理。

    中心区口径重标定（2026-09-15）。区分点：绢本/织物纹理**稠密**
    （edge ≥ 0.16，且此时 glcm ≥ 0.28 再加权），据此与写意水墨拉开差距。
    已用 `inputs/绢本工笔画-1.jpeg` 与 `inputs/绢本工笔画-2.jpeg` 真值校准：
    两者中心特征分别为 edge/glcm=`0.0915/0.4295`、`0.1234/0.3466`。
    真绢本不一定有织物样本那么高的边缘密度，因此增加“暖色绢地 + 细腻纹理”分支；
    仍保留 edge≥0.16 的密集织物分支，并用 edge 前置约束防止水墨误判。
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

    # 细腻密集（工笔/织物）——须**显著**高于水墨；阈值由实测对比定：
    #   织物 damask edge0.203/glcm0.337（稠密）vs 水墨 明代 edge0.123/glcm0.325（稀疏）
    #   → 仅以 glcm 判定会把明代水墨误判为绢本，故 **glcm 项以 edge 稠密为前提**。
    dense = edge_density >= 0.16
    if dense:
        score += 0.30
    elif edge_density >= 0.12:
        score += 0.05

    if dense and glcm_energy >= 0.28:
        score += 0.30
    elif dense and glcm_energy >= 0.22:
        score += 0.05

    # 真绢本工笔的中心主体可能是浅色、边缘中等，但纹理能量明显高于
    # 写意水墨；要求 b* / 饱和度同时达到暖绢地范围，避免仅凭 glcm
    # 把明代水墨（edge=0.123, glcm=0.325）推成绢本。
    warm_silk = 14 <= cb <= 24 and saturation >= 14 and glcm_energy >= 0.34
    if warm_silk:
        score += 0.35
    elif 14 <= cb <= 24 and saturation >= 14 and glcm_energy >= 0.30 and edge_density >= 0.08:
        score += 0.15

    return score


def _score_oil_canvas(cL: float, saturation: float, edge_density: float) -> float:
    """
    油画布：多样亮度 + 高饱和（> 25）+ 高边缘密度（厚重笔触）。

    C3（2026-09-15）：统一到**中心区**口径，并已用真值样本标定：
    `inputs/油画.jpeg` 中心 L52.2 a7.0 b27.0 sat28.1 edge0.241 glcm0.159
    → 油画布 0.70 胜出（其余族 ≤0.50），与人工判断一致。
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


def _score_fabric_wallcovering(cL: float, ca: float, cb: float,
                               saturation: float, edge_density: float,
                               lbp_entropy: float = None) -> float:
    """织物壁布（工艺壁布 / 绗缝面料 / 面料实物照）：**近中性 + 有结构纹理**。

    标定（2026-09-16，真值 `inputs/工艺壁布-1/2/3.jpeg` 中心区实测）：
      -1 L45.9 a2.0 b7.0  sat7.1  edge0.133 LBP熵2.47
      -2 L69.0 a1.0 b9.0  sat10.7 edge0.385 LBP熵2.18
      -3 L84.3 a0.0 b5.0  sat7.0  edge0.098 LBP熵2.46
    对照（同为"低彩"的绘画族，均被**合取门**挡在外）：
      绢本-1 sat15.3 b16 ／ 水墨宋代 sat14.5 b15 ／ 金地 sat34.9 b38 ／ 油画 sat28.1 b27。

    ⚠️ **两道门，缺一不可**（只靠"中性"会把噪声/纯色也判成织物，实测确如此）：
      1. **近中性合取门**：`sat<15 且 b*<15`；
      2. **结构门（LBP 熵 ≥ 2.0）**：真实织物的 LBP 分布有结构（实测 2.18~2.47），
         而纯随机噪声 1.32、灰噪声 1.36、平滑纯色 0.76 —— 这些必须排除。
    亮度**不作判据**（织物样本 L 跨 46~84）。
    """
    # 门 1：近中性合取（防止仅 sat 或仅 b* 单侧偏低而误判）
    if not (saturation < 15.0 and cb < 15.0):
        return 0.0
    # 门 2：结构门（排除噪声/纯色等无结构纹理）
    if lbp_entropy is not None and lbp_entropy < 2.0:
        return 0.0

    score = 0.0
    # 低饱和（织物多无彩/低彩）
    if saturation < 8:
        score += 0.40
    elif saturation < 12:
        score += 0.25
    else:
        score += 0.05
    # 低 b*（排除暖黄绢地/纸地与金色）
    if cb < 8:
        score += 0.35
    elif cb < 12:
        score += 0.20
    else:
        score += 0.05
    # 织物结构（拼接/绗缝线带来中高边缘密度）
    if edge_density >= 0.12:
        score += 0.20
    elif edge_density >= 0.08:
        score += 0.10
    # a* 近中性
    if abs(ca) < 6:
        score += 0.10
    return max(0.0, score)


def get_material_families() -> list:
    """
    返回支持的材质家族列表
    """
    return ["金地屏风", "宣纸水墨", "绢本工笔", "油画布", "织物壁布", "其他"]


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
    elif family_name == "织物壁布":
        explanation += f"  - 近中性合取门（饱和<15 且 b*<15）✓\n"
        explanation += f"  - 低饱和度（< 12）✓\n" if saturation < 12 else f"  - 低饱和度（< 12）✗\n"
        explanation += f"  - 低 b*（< 12）✓\n" if bg_b < 12 else f"  - 低 b*（< 12）✗\n"
        explanation += f"  - 织物结构（edge ≥ 0.08）✓\n" if edge_density >= 0.08 else f"  - 织物结构（edge ≥ 0.08）✗\n"
    else:
        explanation += f"  - 未匹配任何特定材质，归入兜底类\n"

    return explanation
