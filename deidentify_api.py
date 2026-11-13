"""
去識別化 API 服務
======================================
串接 identify1.py 的去識別化邏輯，讓網頁前端可以透過 HTTP 上傳
結構化（CSV / JSON）或非結構化（TXT，其餘檔案類型記錄檔案資訊）資料，
並取得去識別化後的結果。

安裝需求：
    pip install fastapi uvicorn python-multipart

啟動方式：
    uvicorn deidentify_api:app --reload --port 8000

啟動後 API 文件可在 http://127.0.0.1:8000/docs 查看
"""

import csv
import hashlib
import io
import json
import random
import re
import string
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ------------------------------------------------------------------
# 去識別化核心邏輯（改自 identify1.py，並強化假名化的唯一性）
# ------------------------------------------------------------------


class DeIdentifier:
    def __init__(self, salt: str = "secret_salt_123"):
        # salt 用於增加雜湊的安全性，防止彩虹表攻擊
        self.salt = salt
        self.mapping_table: dict[str, str] = {}  # 記錄假名化對照表

    def mask_text(self, text, visible_start=0, visible_end=0, mask_char="*"):
        """遮罩處理：保留前後幾碼，中間遮掉。例："0912345678" -> "09******78" """
        if not text:
            return ""
        s = str(text)
        if visible_start + visible_end >= len(s):
            return s  # 長度太短，不遮罩
        masked_part = mask_char * (len(s) - visible_start - visible_end)
        return s[:visible_start] + masked_part + s[len(s) - visible_end:]

    def hash_text(self, text):
        """雜湊處理：轉為不可逆的 SHA-256 字串，適合需要保持唯一性的 ID / Email"""
        if not text:
            return ""
        salted_text = self.salt + str(text)
        return hashlib.sha256(salted_text.encode()).hexdigest()

    def pseudonymize(self, text, prefix="User"):
        """假名化：將姓名等資訊替換為隨機 ID（例如 User_aB3dK9），同一輸入永遠對應同一組 ID"""
        if not text:
            return ""
        if text in self.mapping_table:
            return self.mapping_table[text]
        random_id = "".join(random.choices(string.ascii_letters + string.digits, k=6))
        pseudo_id = f"{prefix}_{random_id}"
        self.mapping_table[text] = pseudo_id
        return pseudo_id


deid = DeIdentifier()

# ------------------------------------------------------------------
# 欄位類型自動判斷規則（依欄位名稱關鍵字）
# ------------------------------------------------------------------

FIELD_RULES = [
    (re.compile(r"(name|姓名|名字)", re.I), "mask_phone"),
    (re.compile(r"(phone|mobile|電話|手機)", re.I), "mask_phone"),
    (re.compile(r"(email|mail|信箱|郵件)", re.I), "hash"),
    (re.compile(r"(address|addr|地址|住址)", re.I), "mask_generic"),
    (re.compile(r"(id|身分證|證號)", re.I), "hash"),
]

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_PATTERN = re.compile(r"(\+?886[-\s]?)?0?9\d{2}[-\s]?\d{3}[-\s]?\d{3}")

STRUCTURED_EXT = {"csv", "xlsx", "xls", "json", "xml", "tsv"}
UNSTRUCTURED_TEXT_EXT = {"txt"}


def classify_field(field_name: str) -> str:
    for pattern, action in FIELD_RULES:
        if pattern.search(field_name):
            return action
    return "mask_generic"


def apply_action(action: str, value: Any):
    if value is None:
        return value
    value = str(value)
    if action == "pseudonymize":
        return deid.pseudonymize(value)
    if action == "mask_phone":
        return deid.mask_text(value, 2, 2)
    if action == "hash":
        return deid.hash_text(value)
    # mask_generic：僅保留頭尾各 1 碼
    return deid.mask_text(value, 1, 1)


def dedentify_row(row: dict) -> dict:
    return {key: apply_action(classify_field(key), val) for key, val in row.items()}


def deidentify_text(text: str) -> str:
    """對純文字內容中出現的 Email / 電話號碼做遮罩，其餘文字保留"""

    def _mask_email(m: re.Match) -> str:
        return deid.hash_text(m.group(0))[:12] + "…"

    def _mask_phone(m: re.Match) -> str:
        return deid.mask_text(m.group(0), 2, 2)

    text = EMAIL_PATTERN.sub(_mask_email, text)
    text = PHONE_PATTERN.sub(_mask_phone, text)
    return text


# ------------------------------------------------------------------
# FastAPI 應用程式
# ------------------------------------------------------------------

app = FastAPI(title="去識別化 API", version="1.0.0")

# 開發階段先全開，正式上線請改成實際網域白名單
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


async def process_single_file(upload: UploadFile) -> dict:
    ext = get_extension(upload.filename)
    raw = await upload.read()

    try:
        if ext == "csv":
            text = raw.decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(text))
            rows = [deidentify_row(row) for row in reader]
            return {
                "filename": upload.filename,
                "category": "structured",
                "format": "csv",
                "row_count": len(rows),
                "data": rows,
            }

        if ext == "json":
            text = raw.decode("utf-8")
            parsed = json.loads(text)
            if isinstance(parsed, list):
                rows = [deidentify_row(r) if isinstance(r, dict) else r for r in parsed]
            elif isinstance(parsed, dict):
                rows = deidentify_row(parsed)
            else:
                rows = parsed
            return {
                "filename": upload.filename,
                "category": "structured",
                "format": "json",
                "data": rows,
            }

        if ext in UNSTRUCTURED_TEXT_EXT:
            text = raw.decode("utf-8", errors="ignore")
            return {
                "filename": upload.filename,
                "category": "unstructured",
                "format": "txt",
                "data": deidentify_text(text),
            }

        # 其他非結構化檔案（PDF / 圖片 / Word 等）：先不解析內容，
        # 僅回報檔案資訊，避免誤處理二進位內容
        return {
            "filename": upload.filename,
            "category": "unstructured",
            "format": ext or "unknown",
            "data": None,
            "note": "此檔案類型未解析內容，僅記錄檔案資訊。如需解析 PDF / 圖片文字，"
            "可另外加裝對應套件（如 pypdf、pytesseract）。",
            "size_bytes": len(raw),
        }

    except Exception as exc:  # noqa: BLE001
        return {
            "filename": upload.filename,
            "category": "error",
            "error": f"處理失敗：{exc}",
        }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/deidentify/batch")
async def deidentify_batch(files: list[UploadFile] = File(...)):
    """
    對應網頁的「送出資料」按鈕：一次接收多個檔案（結構化 + 非結構化混合），
    自動判斷類型並個別去識別化，回傳每個檔案的處理結果。
    """
    results = [await process_single_file(f) for f in files]
    return JSONResponse({"count": len(results), "results": results})


@app.post("/api/deidentify/structured")
async def deidentify_structured(file: UploadFile = File(...)):
    """單一結構化檔案（CSV / JSON）去識別化"""
    result = await process_single_file(file)
    return JSONResponse(result)


@app.post("/api/deidentify/unstructured")
async def deidentify_unstructured(file: UploadFile = File(...)):
    """單一非結構化檔案去識別化（目前完整支援 .txt，其餘類型回傳檔案資訊）"""
    result = await process_single_file(file)
    return JSONResponse(result)
