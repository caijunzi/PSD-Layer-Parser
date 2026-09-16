# -*- coding: utf-8 -*-
"""引擎与交付审计共享的校准常量（单一真相源，防配置漂移）。

INK_GRAY_DELTA
    墨迹灰度判定差：`gray < median(gray) - INK_GRAY_DELTA` 视为墨迹。
    使用方：
      - run_universal_engine（Step2 去墨掩模 / 未分类残层的墨迹判定）
      - tools/audit_psb（④ 内容承载的墨迹口径）
    两处**必须**同值：引擎据此决定"哪些像素要去墨并由残层承载"，
    审计据此判定"哪些墨迹被抹除却无层承载"。若引擎阈值更严（更大），
    审计口径的中暗像素会成为结构性漏检（实测 textile_damask@scale4：
    -20 → ④ 4.06% 不达标；-12 → 0.00%）。取值 12 为审计侧历史校准值。
"""
INK_GRAY_DELTA = 12.0
