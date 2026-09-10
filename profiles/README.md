# profiles/ —— ICC 色彩配置文件

> **重要**：`psd-tools` 与 `Pillow` **均不自带** FOGRA / Japan Color 等印刷 ICC 配置文件，必须自备。
> 本目录的文件不入库（见根目录 `.gitignore`），由使用方自行获取后放入。

## ⚠️ 概念澄清：ICC 里没有 TAC（2026-09-10 修正）

本文档早期版本写"用命令确认 Pillow 能读出 **TAC 上限**（TAC 由 profile 派生，不硬编码）"，
**这是概念错误**，特此更正：

| 事实 | 说明 |
| :--- | :--- |
| **ICC profile 不含 TAC / TIL / 黑版生成字段** | ICC 规范（ISO 15076-1）未定义该字段。实测系统内 39 个 ICC 的 tag 中不存在任何 ink/tac/limit 相关项 |
| **TAC 上限是印刷工艺参数** | 由 ISO 12647-2 按纸张类别与印刷方式规定，或由印刷厂在工艺单中给定 |
| **profile 与印刷条件是「配对关系」** | 同一印刷条件对应特定 profile，而非"profile 里存着 TAC" |
| **但 profile 的转换表已按该条件优化** | profile 在构建时已按印刷条件的墨量约束与网点扩大做补偿，所以**转换结果通常自然落在该条件的 TAC 范围内** —— 这是"效果上的约束"，不是"字段里的数值" |

因此本项目的正确分工（ADR-007）：

```
ICC profile   → 色彩转换 + 黑版生成（GCR/UCR）
印刷条件       → TAC 上限 / MaxK 上限     ← 见 engine/core/ink_limiter.py
显式 TAC 压制  → 兜底保险（实测必要：ICC 转换后仍有约 0.45% 像素超出限值）
```

**实测佐证（512×256 源图片段，FOGRA39）**：

| 转换方式 | TAC 最大 | K 最大 | 说明 |
| :--- | ---: | ---: | :--- |
| 朴素 `convert('CMYK')` | 276.5% | **0.0%** | ⚠️ **完全不生成黑版**，深色区靠 CMY 三色叠印，印刷上灰平衡不稳 |
| ICC `CoatedFOGRA39` | 328.2% | **97.7%** | 有真实黑版；但超出 300% 实践限值约 0.45% 像素 |
| ICC + 显式 TAC 压制 | **300.0%** | 97.7% | 合规 |

⇒ **ICC 是必需项，不是可选项**；显式压制也必要，二者互补而非重复。

## 需要的配置文件

| 用途 | 推荐 profile | 备注 |
| :--- | :--- | :--- |
| 涂布纸默认目标 | `ISO Coated v2 (Fogra39L)` | 计划 §5.2 默认；`Coated_FOGRA39L.icc` |
| 备选涂布纸 | `Coated FOGRA39 (ISO 12647-2:2004)` | 与 Fogra39L 的 TAC/黑版曲线不同，**勿混用** |
| 日本色系 | `Japan Color 2001 Coated` | 印刷条件与 TAC 与 Fogra39L 不同 |
| 源色彩空间（默认） | `sRGB IEC61966-2.1` | Pillow 自带，无需额外文件 |
| 源色彩空间（可选） | `Adobe RGB (1998)` | 相机若用 Adobe RGB，必须在 preset 中显式声明 |

放到本目录后，在 preset 中声明：

```json
{
  "icc_path": "profiles/Coated_FOGRA39L.icc",
  "print_condition": "eci_coated_practice",
  "tac_limit_pct": null,
  "max_k_pct": null
}
```

- `print_condition` 取值见 `engine/core/ink_limiter.py` 的 `PRINT_CONDITIONS`
  （`eci_coated_practice` / `iso_coated_sheetfed` / `iso_uncoated` / `newsprint` 等）；
- `tac_limit_pct` / `max_k_pct` 显式给出时**优先级最高**（工艺参数以印刷厂给定为准）；
- 也可用命令行临时覆盖：`--icc <path>`。

## 获取与许可

- 官方渠道：ICC（color.org）、ECI（eci.org）提供的标准 profile 包。
- **落地前必须记录**：文件名、版本号、获取来源 URL、许可条款，写入 `docs/MEMORY.md` 的「许可核对」章节。
- 禁止把来源不明的 ICC 直接用于生产交付。
- ⚠️ Windows 系统自带的 `C:\Windows\System32\spool\drivers\color\*.icc`（含 FOGRA39 等 20 个 CMYK profile）
  **仅供开发验证**，其再分发与商用许可需自行核实，请勿直接用于生产交付。

## 校验

放入文件后，确认 Pillow 能加载、色彩空间正确，并查看系统推断出的印刷条件：

```bash
python -c "
from engine.core.ink_limiter import icc_summary, resolve_policy
import json
info = icc_summary('profiles/Coated_FOGRA39L.icc')
print(json.dumps(info, ensure_ascii=False, indent=2))
pol = resolve_policy(icc_path='profiles/Coated_FOGRA39L.icc')
print('TAC 上限:', pol.limit_pct, '%  MaxK:', pol.max_k_pct, '%')
print('来源    :', pol.source)
"
```
