"""图片上传接口。

接收 multipart 文件 → 落临时文件 → 交给 file_handler 保存。
"""
import os
import tempfile

from fastapi import APIRouter, UploadFile, File, HTTPException

from core.file_handler import save_upload, is_allowed_file

router = APIRouter()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传单张图片，返回 file_id + 缩略图 + 推荐 preset。"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件")

    data = await file.read()
    if not is_allowed_file(file.filename, len(data)):
        raise HTTPException(
            status_code=400,
            detail="不支持的格式或超大（仅 JPG/PNG/PSD/PSB，≤100MB）",
        )

    # 先落临时文件再交给 save_upload（move 避免重复拷贝）
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix="uls_upload_", suffix=os.path.splitext(file.filename)[1]
    )
    os.close(tmp_fd)
    with open(tmp_path, "wb") as f:
        f.write(data)

    try:
        info = save_upload(tmp_path, file.filename)
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=f"保存失败: {e}")

    if not info.get("dimensions"):
        # PSD/PSB 读不到尺寸，给个兜底（不阻断流程）
        info["dimensions"] = None

    return {
        "success": True,
        "data": info,
    }
