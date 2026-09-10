# profiles/ —— ICC 色彩配置文件

> **重要**：`psd-tools` 与 `Pillow` **均不自带** FOGRA / Japan Color 等印刷 ICC 配置文件，必须自备。
> 本目录的文件不入库（见根目录 `.gitignore`），由使用方自行获取后放入。

## 需要的配置文件

| 用途 | 推荐 profile | 备注 |
| :--- | :--- | :--- |
| 涂布纸默认目标 | `ISO Coated v2 (Fogra39L)` | 计划 §5.2 默认；`Coated_FOGRA39L.icc` |
| 备选涂布纸 | `Coated FOGRA39 (ISO 12647-2:2004)` | 与 Fogra39L 的 TAC/黑版曲线不同，**勿混用** |
| 日本色系 | `Japan Color 2001 Coated` | TAC 与 Fogra39L 不同，须从 profile 派生而非硬编码（ADR-007） |
| 源色彩空间（默认） | `sRGB IEC61966-2.1` | Pillow 自带，无需额外文件 |
| 源色彩空间（可选） | `Adobe RGB (1998)` | 相机若用 Adobe RGB，必须在 preset 中显式声明 |

## 获取与许可

- 官方渠道：ICC（color.org）、ECI（eci.org）提供的标准 profile 包。
- **落地前必须记录**：文件名、版本号、获取来源 URL、许可条款，写入 `docs/MEMORY.md` 的「许可核对」章节。
- 禁止把来源不明的 ICC 直接用于生产交付。

## 校验

放入文件后，用下面的命令确认 Pillow 能正确加载并读出 TAC 上限（TAC 由 profile 派生，不硬编码）：

```bash
python -c "
from PIL import ImageCms
p = ImageCms.getOpenProfile('profiles/Coated_FOGRA39L.icc')
print('description:', p.profile.description)
print('rendering intents:', ImageCms.get_intent_supported(p))
"
```
