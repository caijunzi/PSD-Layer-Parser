"""结果下载接口。"""
import io
import zipfile
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

# 注意：必须以模块属性方式访问（file_handler.OUTPUTS_DIR），
# 保证测试 patch 数据目录时此处可见（import 时绑定会绕过 patch）。
from core import file_handler

router = APIRouter()

# 产物文件候选（按存在性解析，兼容引擎多种命名）：
# - both 模式：result.plate.psb / result.design.psb（加后缀）；
#   manifest 每线一份 result.{plate,design}.manifest.json（plate 版=印前审计凭据优先）
# - 单模式：只有 result.psb（不加后缀！），design/plate 都回退到它
CANDIDATES = {
    "design": ["result.design.psb", "result.psb"],
    "plate": ["result.plate.psb", "result.psb"],
    "manifest": ["result.plate.manifest.json", "result.design.manifest.json",
                 "result.manifest.json"],
    # 交付前 8 维审计报告（tools/audit_psb.py 产出，任务完成后异步生成）
    "audit": ["result.audit.json"],
}


@router.get("/download/{task_id}/{file_type}")
async def download(task_id: str, file_type: str):
    """下载处理产物。

    file_type: design | plate | manifest | masks
    """
    out_dir = file_handler.OUTPUTS_DIR / task_id
    if not out_dir.exists():
        raise HTTPException(status_code=404, detail="任务产物目录不存在")

    if file_type == "masks":
        # both 模式每线一份 result.{plate,design}.masks/；单模式 result.masks/
        masks_dir = None
        for d in ("result.plate.masks", "result.design.masks", "result.masks"):
            cand = out_dir / d
            if cand.exists() and any(cand.iterdir()):
                masks_dir = cand
                break
        if masks_dir is None:
            raise HTTPException(status_code=404, detail="掩码目录不存在或为空")
        # 实时打包为 zip
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in masks_dir.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(masks_dir))
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=masks.zip"},
        )

    names = CANDIDATES.get(file_type)
    if not names:
        raise HTTPException(status_code=400, detail="未知文件类型")
    for fname in names:
        p = out_dir / fname
        if p.exists():
            return FileResponse(
                str(p),
                filename=fname,
                media_type="application/octet-stream",
            )
    raise HTTPException(status_code=404, detail=f"{file_type} 产物不存在")
