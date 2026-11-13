# -*- coding: utf-8 -*-
"""
app.py
FastAPI 後端：
  POST /api/deidentify
    - 接收前端上傳的檔案 (multipart/form-data, field name = "file")
    - 可選參數 strategy_overrides（JSON 字串，例如 {"EMAIL":"redact"}）
    - 自動判斷副檔名決定走「結構化」或「非結構化」流程
    - 回傳：處理後檔案（base64）+ 偵測報告 (JSON)

啟動方式：
    uvicorn app:app --reload --port 8000

需要安裝的套件：
    pip install fastapi "uvicorn[standard]" python-multipart pandas openpyxl python-docx pypdf pdfplumber reportlab
"""

import base64
import hashlib
import hmac
import io
import json
import os
import threading
import time
import uuid
from typing import Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from detection import match_column_by_name, guess_column_by_sample, detect_text_pii, detect_names_ner, PATTERNS, VALIDATION_PATTERNS
from deidentify_api import DeIdentifier


app = FastAPI(title="資料去識別化服務")

# 開發階段先開放全部來源，正式環境請限制為前端網域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STRUCTURED_EXTS = {"csv", "xlsx", "xls", "json"}
PLAIN_TEXT_EXTS = {"txt", "md"}
DOCX_EXTS = {"docx"}
PDF_EXTS = {"pdf"}

# ---------------------------------------------------------------------------
# 去識別化引擎（沿用 deidentify_api.py 的遮罩 / 雜湊 / 假名化邏輯）
# ---------------------------------------------------------------------------
deid = DeIdentifier()

# 每種偵測類型預設要套用的策略：mask（遮罩）/ hash（雜湊）/ pseudonymize（假名化）/ redact（整個抹除）
DEFAULT_STRATEGY = {
    "NAME": "mask",
    "TW_ID": "hash",
    "TW_ARC": "hash",
    "MOBILE": "mask",
    "PHONE": "mask",
    "EMAIL": "mask",
    "ADDRESS": "mask",
    "BIRTHDAY": "mask",
    "CREDIT_CARD": "hash",
    "BANK_ACCOUNT": "hash",
    "OCCUPATION": "redact",
    "PASSPORT": "mask",
    "NHI_CARD": "mask",
    "IP": "hash",   # IPv4：雜湊後取 12 碼（HASH_TRUNCATE_LENGTH）
    "MAC": "hash",  # MAC 位址：雜湊後取 12 碼
    # 本機 Ollama 輔助偵測（llm_detector.py）可能回傳的類型
    "TITLE_NAME": "pseudonymize",
    "ORG": "mask",
    "LLM_OTHER": "mask",
}


# 雜湊結果截斷長度：SHA-256 完整輸出是 64 碼，對身分證號這類欄位來說太長、不好閱讀。
# 只取前 12 碼（48 bits）仍然是不可逆的單向雜湊，碰撞機率極低
# （例如 10 萬筆資料裡任兩筆撞在一起的機率約 0.002%），對一般規模的資料集來說很安全。
# 如果之後資料量成長到百萬筆以上、想要更保守，可以把這個數字調大（例如 16）。
HASH_TRUNCATE_LENGTH = 12


import re

# 複姓（兩個字的姓），其餘一律當單姓處理
COMPOUND_SURNAMES = (
    "歐陽", "司馬", "上官", "諸葛", "夏侯", "皇甫", "司徒", "東方", "南宮", "令狐",
    "慕容", "宇文", "長孫", "公孫", "軒轅", "淳于", "尉遲", "端木", "万俟", "澹台",
)

# 台灣區碼（含開頭 0）。四碼：0826 烏坵、0836 馬祖；三碼：037/049/082/089；其餘兩碼
_AREA_4 = ("0826", "0836")
_AREA_3 = ("037", "049", "082", "089")


def mask_keep_prefix(text: str, keep: int = 4) -> str:
    """保留前 keep 個英數字元，其餘英數字換成 *，分隔符號（空白、-）原樣保留。
    例：312345678 → 3123*****、1234-5678-9012 → 1234-****-****"""
    out, seen = [], 0
    for ch in text.strip():
        if ch.isalnum():
            out.append(ch if seen < keep else "*")
            seen += 1
        else:
            out.append(ch)
    return "".join(out)


def mask_name(text: str) -> str:
    """姓名遮罩：保留姓氏，其餘每個字換成 *。
    中文名：單姓保留 1 字、複姓保留 2 字（例：王小明 → 王**、歐陽娜娜 → 歐陽**）。
    非中文名（英文等）只保留第一個字元。"""
    s = text.strip()
    if not s:
        return text
    if re.search(r"[\u4e00-\u9fff]", s):
        # 只處理開頭連續的中文字，後面如果接了括號英文名等，一併遮掉
        surname_len = 2 if s[:2] in COMPOUND_SURNAMES and len(s) > 2 else 1
        return s[:surname_len] + "*" * (len(s) - surname_len)
    return s[:1] + "*" * (len(s) - 1)


def mask_phone(text: str) -> str:
    """電話遮罩（連同分隔符號原樣保留，只把數字換成 *）：
    - 手機：保留前四碼（0912-345-678 → 0912-***-***）
    - 市話：保留區號 + 本地號碼前四碼（02-1234-5678 → 02-1234-****）
    - 國際格式 +886 開頭的照同樣規則，+886 本身保留
    - 分機（轉/分機/ext/#）的數字全部遮掉"""
    s = text.strip()
    m = re.match(r"^(.*?)(\s*(?:轉|分機|[Ee][Xx][Tt]\.?|#)\s*\d+)?$", s)
    main, ext = m.group(1), m.group(2) or ""

    prefix = ""
    cc = re.match(r"^[(（]?\+886[)）]?[-\s]?", main)
    if cc:
        prefix = cc.group(0)
        main = main[cc.end():]
        digits = "0" + re.sub(r"\D", "", main)  # 國際格式省略了開頭的 0，補回來判斷類型
        added_zero = True
    else:
        digits = re.sub(r"\D", "", main)
        added_zero = False

    if digits.startswith("09"):
        keep = 4  # 手機：前四碼
    else:
        area = 4 if digits.startswith(_AREA_4) else 3 if digits.startswith(_AREA_3) else 2
        keep = area + 4  # 市話：區號 + 本地號碼前四碼

    if added_zero:
        keep -= 1  # 補上去的 0 不在原字串裡

    out, seen = [], 0
    for ch in main:
        if ch.isdigit():
            out.append(ch if seen < keep else "*")
            seen += 1
        else:
            out.append(ch)
    ext_masked = re.sub(r"\d", "*", ext)
    return prefix + "".join(out) + ext_masked


_ADDR_HEAD = re.compile(r"^[\u4e00-\u9fff]{2}[縣市](?:[\u4e00-\u9fff]{1,3}?[區鄉鎮市])?")


def mask_address(text: str) -> str:
    """地址遮罩：保留到「區（鄉鎮市）」為止，後面全部換成 *
    （台北市松山區八德路三段 → 台北市松山區*******）。
    認不出縣市＋區的格式時，退回保留頭尾各 1 字。"""
    s = text.strip()
    m = _ADDR_HEAD.match(s)
    if not m:
        return deid.mask_text(s, 1, 1)
    head = m.group(0)
    return head + "*" * (len(s) - len(head))


def mask_email(text: str) -> str:
    """Email 遮罩：@ 前面的帳號每個字元換成 *，保留 @ 與後面的網域。
    shuhong.lin.tw@gmail.com → ***************@gmail.com
    沒有 @ 的內容（不是正常 Email）退回通用規則，保留頭尾各 1 碼。"""
    s = text.strip()
    if "@" not in s:
        return deid.mask_text(s, 1, 1)
    local, _, domain = s.rpartition("@")
    return "*" * len(local) + "@" + domain


def mask_preview(original: str, pii_type: str, strategy: str) -> str:
    """報告「原文片段」欄的預覽：策略是 mask 時，套用跟實際輸出一樣的類型遮罩規則
    （姓名留姓氏、電話留前四碼／區號、地址留到區），其他策略維持頭尾各留 1 碼，
    避免預覽比實際輸出洩漏更多原文。"""
    if strategy == "mask":
        return apply_strategy(original, "mask", pii_type)
    return deid.mask_text(original, 1, 1)


def apply_strategy(value, strategy: str, pii_type: Optional[str] = None) -> str:
    """依指定策略處理單一值"""
    if value is None:
        return value
    text = str(value)
    if text == "" or text.lower() == "nan":
        return text

    if strategy == "hash":
        return deid.hash_text(text)[:HASH_TRUNCATE_LENGTH]
    if strategy == "hash_full":
        # 保留完整 64 碼雜湊值（碰撞機率趨近於零，但很長），需要更高安全等級時可用
        return deid.hash_text(text)
    if strategy == "pseudonymize":
        return deid.pseudonymize(text)
    if strategy == "redact":
        return "[REDACTED]"
    # 預設：mask。姓名／電話／地址各有自己的遮罩規則，其他類型保留頭尾各 1 碼
    if pii_type in ("NAME", "TITLE_NAME"):
        return mask_name(text)
    if pii_type in ("MOBILE", "PHONE"):
        return mask_phone(text)
    if pii_type == "ADDRESS":
        return mask_address(text)
    if pii_type == "EMAIL":
        return mask_email(text)
    if pii_type in ("PASSPORT", "NHI_CARD"):
        return mask_keep_prefix(text, 4)
    if is_suggested_type(pii_type):  # 使用者勾選的建議項目：短詞（如「肺癌」）保留頭尾會等於沒遮，所以整個遮掉
        return text[0] + "*" * (len(text) - 1) if len(text) >= 4 else "*" * len(text)
    return deid.mask_text(text, 1, 1)


# ---------------------------------------------------------------------------
# 使用者自選的欄位處理方式（確認畫面上每個欄位各自挑一種）：
#   keep 不處理｜mask 遮罩（可設保留前 n、後 m 字）｜hash 雜湊｜redact 刪除內容
#   bucket 數字分組（例如年齡 5 歲一組 → 25–29）｜date 日期只留年或年月｜drop 刪除整欄
# 這些方法不管欄位是什麼類型都能用，所以系統認不得的欄位（年齡、郵遞區號、診斷…）也能處理。
# ---------------------------------------------------------------------------
CUSTOM_METHOD_LABELS = {
    "keep": "不處理", "mask": "遮罩", "hash": "雜湊", "redact": "刪除內容",
    "bucket": "數字分組", "date": "日期泛化", "drop": "刪除整欄",
}


def _custom_label(spec: dict) -> str:
    m = spec.get("method", "keep")
    extra = ""
    if m == "mask":
        extra = f"（留前{int(spec.get('head', 1))}後{int(spec.get('tail', 0))}字）"
    elif m == "bucket":
        off = spec.get("offset", 0) or 0
        extra = f"（每 {spec.get('width', 5)} 一組" + (f"，從 {off} 起" if off else "") + "）"
    elif m == "date":
        extra = "（只留年）" if spec.get("level") == "year" else "（留年月）"
    return "自訂：" + CUSTOM_METHOD_LABELS.get(m, m) + extra


def apply_custom_method(value, spec: dict):
    """依使用者設定處理單一儲存格；回傳 (新值, 是否無法處理)。"""
    text = "" if value is None else str(value)
    if text.strip() == "" or text.strip().lower() == "nan":
        return value, False
    m = spec.get("method", "keep")
    if m == "keep":
        return value, False
    if m == "hash":
        return deid.hash_text(text)[:HASH_TRUNCATE_LENGTH], False
    if m in ("redact", "drop"):
        return "[已刪除]", False
    if m == "mask":
        head = max(0, int(spec.get("head", 1)))
        tail = max(0, int(spec.get("tail", 0)))
        if len(text) <= head + tail:
            return "*" * len(text), False
        return text[:head] + "*" * (len(text) - head - tail) + (text[len(text) - tail:] if tail else ""), False
    if m == "bucket":
        try:
            num = float(text.replace(",", ""))
            width = float(spec.get("width", 5))
            if width <= 0:
                return "無效資料", True
        except ValueError:
            return "無效資料", True
        off = float(spec.get("offset", 0) or 0)  # 分組起點：例如年齡從 1 開始 → 26–30、31–35
        base = ((num - off) // width) * width + off
        lo = int(base) if base == int(base) else base
        hi = lo + int(width) - 1 if width == int(width) else lo + width
        return f"{lo}–{hi}", False
    if m == "date":
        mt = re.match(r"\s*(\d{4})(?:[-/.年]\s*(\d{1,2}))?", text)
        if not mt:
            return "無效資料", True
        if spec.get("level") == "year" or not mt.group(2):
            return mt.group(1), False
        return f"{mt.group(1)}-{int(mt.group(2)):02d}", False
    return value, False


# 無對應正則規則可驗證格式的類型（判斷完全依賴欄位名稱，任何內容都算合法）
NO_FORMAT_VALIDATION_TYPES = {"NAME", "BIRTHDAY"}


def is_valid_format(value, detected_type: str) -> bool:
    """
    檢查儲存格內容是否真的符合該偵測類型的正確格式。
    例如欄位被判斷成「身分證」，但內容是 A345608970（第二碼不是 1 或 2），
    這種格式不合法的資料要防呆攔下來，而不是照樣硬套遮罩策略。
    """
    if detected_type in NO_FORMAT_VALIDATION_TYPES:
        return True

    if detected_type == "IP":
        # IP 欄位也可能是 IPv6（2001:db8::1），沒有對應的偵測規則，但不能因此判成無效資料而把資料丟掉
        ip_text = str(value).strip()
        return bool(PATTERNS["IP"].fullmatch(ip_text)) or (
            bool(re.fullmatch(r"[0-9A-Fa-f:.]{2,45}", ip_text)) and ip_text.count(":") >= 2
        )

    if detected_type == "ADDRESS":
        # 欄位名稱已經是「地址」，儲存格寫法千變萬化（只寫縣市、沒寫縣市、含樓層備註…），
        # 用完整比對太容易把正常地址誤判成無效資料。只要含有任一地址用字就算合法，
        # 純英數亂碼這類明顯不是地址的內容才攔下來。
        return bool(re.search(r"[縣市區鄉鎮路街巷弄號樓段]", str(value)))

    pattern = PATTERNS.get(detected_type) or VALIDATION_PATTERNS.get(detected_type)
    if pattern is None:
        # 這個類型沒有對應的正則規則（例如 ADDRESS 是結構化比對不到的類型），一律視為合法
        return True

    text = str(value).strip()
    return bool(pattern.fullmatch(text))


def get_ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


# CSV／純文字常見編碼，依序嘗試。很多從 Excel（Windows 繁體中文版）或舊系統匯出的
# CSV 其實是 Big5（或相容的 CP950）編碼，不是 UTF-8，直接用 utf-8 讀會整個失敗，
# 報錯「'utf-8' codec can't decode byte 0xbb in position 0: invalid start byte」
# ——0xbb 這種位元組正是典型的 Big5 字元開頭，UTF-8 解碼器完全看不懂。
TEXT_ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "big5", "cp950", "gb18030")


def _decode_bytes_best_effort(raw: bytes) -> str:
    """依序嘗試幾種常見編碼解碼成文字；全部失敗的話，最後用 utf-8 並把看不懂的
    位元組替換掉，至少讓檔案能處理，而不是整個拋錯中斷。"""
    for enc in TEXT_ENCODING_CANDIDATES:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _read_csv_rows_best_effort(raw: bytes, **kwargs) -> list:
    """依序嘗試幾種常見編碼讀取 CSV，回傳「原始列資料」（不解讀表頭，每一列都是字串
    list），讀到第一個不會噴 UnicodeDecodeError 的編碼就用。"""
    for enc in TEXT_ENCODING_CANDIDATES:
        try:
            raw_df = pd.read_csv(io.BytesIO(raw), dtype=str, header=None, encoding=enc, **kwargs)
            break
        except UnicodeDecodeError:
            raw_df = None
            continue
    else:
        # 全部編碼都失敗：退而求其次，用「替換看不懂的位元組」的方式硬是讀進來，
        # 至少不要讓整個上傳直接處理失敗。
        text = _decode_bytes_best_effort(raw)
        raw_df = pd.read_csv(io.StringIO(text), dtype=str, header=None, **kwargs)

    # 先 fillna("") 再轉字串：不能依賴 .astype(str) 自己處理缺值——pandas 3.x
    # 如果欄位本身已經是新版原生 str dtype，.astype(str) 可能被當成「已經是字串了」
    # 直接跳過轉換，缺值儲存格會維持 float('nan') 而不是字串，之後再誤用成字串
    # 處理（例如 .strip()）就會整個炸掉。
    return raw_df.fillna("").astype(str).values.tolist()


def load_dataframe(raw: bytes, ext: str) -> pd.DataFrame:
    # dtype=str：避免 pandas 自動把數字型欄位（手機、帳號、身分證號等）轉成數值，
    # 導致開頭的 0 不見、或超長數字被轉成科學記號，進而被防呆機制誤判成「無效資料」。
    #
    # CSV／XLSX 都先用 header=None 整張表原封不動讀進來（不讓 pandas 自己猜表頭），
    # 再交給跟 docx 共用的「多列表頭偵測」邏輯處理。這是因為不少官方表單／Excel
    # 範本的表頭其實橫跨兩列（合併儲存格：第一列是大分類、第二列才是細項欄名），
    # 如果直接讓 pandas 猜表頭，第二列會被誤判成一筆資料把整張表弄亂；沒有文字的
    # 合併儲存格欄位，pandas 也會自動編號成「Unnamed: 4」這種欄名，而英文關鍵字
    # "name" 剛好是 "Unnamed" 的子字串，會被誤判成姓名欄而整欄被假名化覆蓋掉。
    if ext == "csv":
        rows = _read_csv_rows_best_effort(raw)
        return _rows_to_dataframe_with_header_detection(rows)
    if ext in ("xlsx", "xls"):
        raw_df = pd.read_excel(io.BytesIO(raw), dtype=str, header=None)
        rows = raw_df.fillna("").astype(str).values.tolist()
        return _rows_to_dataframe_with_header_detection(rows)
    if ext == "json":
        return pd.read_json(io.BytesIO(raw), dtype=str)
    raise ValueError(f"不支援的結構化格式: {ext}")


def dataframe_to_bytes(df: pd.DataFrame, ext: str) -> bytes:
    buf = io.BytesIO()
    pre = df.attrs.get("preamble") or []
    if ext == "csv":
        if pre:
            import csv as _csv

            sio = io.StringIO()
            _csv.writer(sio).writerows(pre)
            sio.write(df.to_csv(index=False))
            buf.write(sio.getvalue().encode("utf-8-sig"))
        else:
            df.to_csv(buf, index=False, encoding="utf-8-sig")
    elif ext in ("xlsx", "xls"):
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, startrow=len(pre))
            ws = writer.sheets["Sheet1"]
            for r, row in enumerate(pre, start=1):
                for c, v in enumerate(row, start=1):
                    if v != "":
                        ws.cell(row=r, column=c, value=v)
    elif ext == "json":
        buf.write(df.to_json(orient="records", force_ascii=False).encode("utf-8"))
    return buf.getvalue()


def extract_text_from_docx(raw: bytes) -> str:
    import docx  # python-docx

    doc = docx.Document(io.BytesIO(raw))
    # 段落之間的換行是真的段落分隔，不能接回去；只整理空白，讓 Word 與 PDF 的偵測結果一致
    return normalize_text_spacing("\n".join(p.text for p in doc.paragraphs))


def _dedupe_columns(names):
    """
    幫重複或空白的欄位名稱編號，避免 pandas 把同名欄位當成同一組（那樣 df[col] 會回傳
    DataFrame 而不是 Series，後續 .tolist()、.apply() 等操作就會整個炸掉）。
    例如簽到表這種表格常有合併儲存格造成的重複/空白標題（如兩欄都叫「簽名」）。
    """
    seen = {}
    result = []
    for raw_name in names:
        name = (raw_name or "").strip() or "欄位"
        if name in seen:
            seen[name] += 1
            result.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            result.append(name)
    return result


def _looks_like_pii_value(text: str) -> bool:
    """這段文字有沒有任何部分長得像真正的個資內容（而不是表頭標籤文字）。"""
    return any(p.search(text) for p in PATTERNS.values())


def _guess_name_column_by_sample(sample_values, sample_size: int = 20) -> bool:
    """
    姓名沒有固定格式，沒辦法像電話、身分證那樣用正則判斷內容，平常都是靠欄位名稱
    （欄名裡有「姓名」兩個字）才抓得到。但如果資料根本沒有表頭（直接條列個人資料，
    沒有任何欄位名稱可以比對），姓名欄就完全沒有線索可以判斷了。
    這裡用 CKIP NER 當最後一道手段：抽樣這一欄的內容，如果幾乎每一格「整格」都被
    辨識成人名（而不是一句話裡剛好帶到一個名字），就判斷這欄是姓名欄。
    CKIP 套件沒裝好的話 detect_names_ner 會直接回傳空結果，這個判斷就會很自然地
    判定為「不是姓名欄」，不會影響其他偵測邏輯或讓程式出錯。
    """
    values = []
    for v in sample_values[:sample_size]:
        sv = str(v).strip() if v is not None else ""
        if sv and sv.lower() != "nan":
            values.append(sv)
    if not values:
        return False

    hits = 0
    for v in values:
        entities = detect_names_ner(v)
        covered = sum(max(0, e[2] - e[1]) for e in entities)
        if covered >= len(v) * 0.6:
            hits += 1
    return (hits / len(values)) >= 0.6


def _detect_header_row_count(rows, max_header_rows: int = 4) -> int:
    """
    判斷表格最前面有幾列其實是「合併儲存格造成的多列表頭」，而不是真正的資料列。

    許多公家機關的表單（例如簽到表、檢查表）習慣把表頭拆成兩列：第一列是大分類
    （用合併儲存格橫跨好幾欄，例如「服裝儀容檢查」合併跨 7 欄），第二列才是真正
    的細項欄名（例如「頭髮」「鬍子」「制服」）。如果只把第一列當表頭，第二列會
    被誤判成一筆資料，汙染掉整張表（姓名欄會多出一筆假資料「姓 名」、地址欄會
    多出一筆假資料「服勤處所」...），後面所有真正的資料列也會因此完全沒被正確
    偵測到。

    判斷邏輯：從第 2 列開始，只要這一列同時符合以下兩個條件，就當作表頭的延伸：
      1. 這一列沒有任何儲存格內容長得像真正的個資資料（代表它不是資料列）
      2. 這一列大部分儲存格內容，跟目前已知的表頭文字重複（合併儲存格造成的延伸列，
         同一欄位在兩列會顯示一樣的文字），或是內容本身很短、像標籤而不是資料值
    一旦某一列看起來像真正的資料（出現個資格式，或內容跟表頭完全對不起來），
    就停止往下併入表頭。最多檢查 max_header_rows 列，避免判斷失誤時整張表都被當表頭。

    還有一種情況是「整份表格根本沒有表頭」——有些資料是直接條列個人資料，
    沒有「姓名／身分證／電話」這種標題列，第一列本身就已經是一筆真正的資料
    （例如「林大文,0935987423,A123456789,台北市...」）。這種情況如果照舊把
    第一列當表頭，第一列的資料就會被誤認成欄位名稱、完全不會被去識別化處理
    （因為表頭本身不會被當成資料來遮罩），只有後面幾列才會被處理到。
    所以這裡會先檢查第一列本身「像不像資料」：如果第一列裡有好幾格內容本身
    就符合個資格式（電話、身分證字號、地址...），就直接判斷這份表格沒有表頭，
    回傳 0，所有列（包含第一列）都當成資料處理。
    """
    if not rows:
        return 0

    first_row_non_empty = [c.strip() for c in rows[0] if c.strip()]
    if first_row_non_empty:
        pii_hits = sum(1 for c in first_row_non_empty if _looks_like_pii_value(c))
        # 門檻：格數很少的表格（例如只有 2~3 欄）只要中一格就當作沒有表頭；
        # 格數較多的表格則要求至少中兩格，避免只是剛好有一欄格式像而誤判。
        hit_threshold = 1 if len(first_row_non_empty) <= 3 else 2
        if pii_hits >= hit_threshold:
            return 0

    if len(rows) < 2:
        return 1

    ncols = len(rows[0])
    header_rows = [rows[0]]

    for row in rows[1 : max_header_rows]:
        if len(row) != ncols:
            break

        non_empty = [c.strip() for c in row if c.strip()]
        if not non_empty:
            break

        if any(_looks_like_pii_value(c) for c in non_empty):
            break  # 出現看起來像真正個資的內容，這一列是資料列，表頭到此結束

        if re.fullmatch(r"\d{1,3}", non_empty[0]):
            # 第一個有字的儲存格是純序號（1、2、3…）→ 這是資料列的「編號」欄，不是表頭。
            # 否則「1 | 陳湧和 | 臺北市交通管制工程處」這種內容很短、很多空格的資料列，
            # 會被下面「內容很短、像標籤」的規則誤當成表頭，導致整列（包含姓名）完全沒被遮罩。
            break

        seen_header_texts = {c.strip() for hr in header_rows for c in hr if c.strip()}
        repeat_hits = sum(1 for c in non_empty if c in seen_header_texts)
        avg_len = sum(len(c) for c in non_empty) / len(non_empty)

        if repeat_hits >= 1 or avg_len <= 6:
            header_rows.append(row)
        else:
            break

    return len(header_rows)


def _merge_header_rows(header_rows):
    """
    把偵測到的多列表頭合併成一列欄位名稱。同一欄位如果兩列文字相同（合併儲存格的延伸），
    只取一次；如果不同（第一列是分類、第二列是細項，例如「服裝儀容檢查」+「頭髮」），
    就把兩者串起來，變成「服裝儀容檢查 頭髮」這種更精確、彼此不重複的欄名。
    """
    ncols = len(header_rows[0])
    merged = []
    for c in range(ncols):
        parts = []
        prev = None
        for row in header_rows:
            text = row[c].strip() if c < len(row) else ""
            if text and text != prev:
                parts.append(text)
            if text:
                prev = text
        merged.append(" ".join(parts))
    return merged


def _rows_to_dataframe_with_header_detection(rows) -> pd.DataFrame:
    """
    共用邏輯：給一份「原始列資料」（每一列都是字串 list，可能包含橫跨多列的合併表頭），
    自動判斷表頭橫跨幾列、合併表頭文字，組出最終的 DataFrame。
    DOCX／CSV／XLSX 的結構化資料都共用這份判斷邏輯，統一處理合併儲存格造成的
    多列表頭問題，避免表頭被誤判成資料、或欄位沒有文字時被自動編號成容易跟關鍵字
    誤判撞在一起的欄名（例如 pandas 原本會自動編號成「Unnamed: 4」，而英文關鍵字
    "name" 剛好是 "Unnamed" 的子字串，會害整欄被誤判成姓名欄）。
    """
    rows = [list(r) for r in rows]
    if not rows:
        return pd.DataFrame()

    # 表格上方常有「標題」「說明」這類只佔一格的文字列（或空白列），它們不是表頭也不是資料。
    # 先把它們拿掉（留著原樣，輸出時再放回原位），否則標題會被當成表頭、真正的欄名被當成資料，
    # 整張表的欄位都認不出來。條件：整張表至少 3 欄，且該列只有 1 格有內容（或整列空白）。
    preamble = []
    if max(len(r) for r in rows) >= 3:
        while len(rows) > 2 and sum(1 for c in rows[0] if str(c).strip()) <= 1:
            preamble.append(rows.pop(0))
    df = _rows_to_dataframe_core(rows)
    if preamble:
        df.attrs["preamble"] = [[str(c) for c in r] for r in preamble]
    return df


def _rows_to_dataframe_core(rows) -> pd.DataFrame:
    header_row_count = _detect_header_row_count(rows)

    if header_row_count == 0:
        # 整份表格沒有表頭，是直接條列的個人資料——用通用的「欄位1／欄位2...」
        # 當欄名，所有列（包含第一列）都當成真正的資料來處理。沒有欄位名稱可以
        # 比對關鍵字，後續會完全靠「逐欄抽樣內容格式」(guess_column_by_sample)
        # 來判斷每一欄是什麼類型，不會因為少了表頭就整欄漏判。
        ncols = max(len(r) for r in rows)
        header = [f"欄位{i + 1}" for i in range(ncols)]
        data_rows = rows
    elif len(rows) < 2:
        header = _dedupe_columns(rows[0])
        return pd.DataFrame(columns=header)
    else:
        header_rows = rows[:header_row_count]
        data_rows = rows[header_row_count:]
        header = _merge_header_rows(header_rows) if len(header_rows) > 1 else header_rows[0]
        header = _dedupe_columns(header)

    if not data_rows:
        return pd.DataFrame(columns=header)

    # 把每一列補齊/截斷成跟表頭一樣的欄位數，避免欄位數對不齊造成 DataFrame 出錯
    ncols = len(header)
    normalized_rows = []
    for r in data_rows:
        r = list(r[:ncols]) + [""] * max(0, ncols - len(r))
        normalized_rows.append(r)

    return pd.DataFrame(normalized_rows, columns=header)


def extract_tables_from_docx(raw: bytes):
    """
    抓出 .docx 裡真正的「表格物件」（跟一般段落文字是分開存的），
    每個表格轉成一個 DataFrame。會自動偵測表頭是否橫跨多列（常見於合併儲存格的
    官方表單），並把多列表頭合併成單一列正確的欄位名稱，避免表頭被誤判成資料。
    少於 2 列（沒有標題列+至少1筆資料）的表格會被忽略，不當成表格處理。
    """
    import docx  # python-docx

    doc = docx.Document(io.BytesIO(raw))
    tables = []
    for table in doc.tables:
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        if len(rows) < 2:
            continue

        df = _rows_to_dataframe_with_header_detection(rows)
        if df.empty:
            continue
        tables.append(df)
    return tables


def sniff_table_delimiter(text: str, min_rows: int = 2, min_cols: int = 2) -> Optional[str]:
    """
    判斷一段純文字內容是否其實是「貼上來的表格」（例如從 Excel 複製貼上存成 .txt）。
    做法：試幾種常見分隔符號，如果每一行切出來的欄位數量都一致、且至少有 min_cols 欄，
    就視為表格，回傳偵測到的分隔符號；否則回傳 None（當成一般自由文字處理）。
    """
    lines = [ln for ln in text.strip().splitlines() if ln.strip() != ""]
    if len(lines) < min_rows:
        return None

    for delim in (",", "\t", ";", "|"):
        counts = [len(line.split(delim)) for line in lines]
        if counts[0] >= min_cols and len(set(counts)) == 1:
            return delim
    return None


_CJK = "一-鿿　-〿＀-￯"  # 中日韓文字 + 全形標點


def normalize_text_spacing(text: str) -> str:
    """
    整理文字裡的多餘空白，讓 Word / PDF / TXT 的偵測結果一致（不動換行）。
    常見問題：數字與連字號間被塞空白（0987 -654-321）、中文與數字間有空白
    （1991 年 10 月 24 日、二段 102 號），正則比對會被這些空白擋住。
    """
    import re

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t ]+\n", "\n", text)

    # 數字與連字號之間的多餘空白：4312 -8901 -2345 / 04521 -000 / 0987 - 654
    text = re.sub(r"(?<=\d)[ \t]+(?=[-–—－])", "", text)
    text = re.sub(r"(?<=[-–—－])[ \t]+(?=\d)", "", text)

    # 英數編號的連字號前後被塞空白（PDF 擷取常見，例如「MRN -4567890」「NCKU -B-103099」「EA7583CD -A667」），
    # 不處理的話編號會被空白切成兩段，偵測只抓到前半（MRN）。只處理「連字號緊貼英數字」那一側有空白的情況。
    text = re.sub(r"(?<=[A-Za-z0-9])[ \t]+(?=-[A-Za-z0-9])", "", text)
    text = re.sub(r"(?<=[A-Za-z0-9]-)[ \t]+(?=[A-Za-z0-9])", "", text)

    # 中文字（含全形標點）旁邊的空白一律移除：「號5  樓」→「號5樓」
    text = re.sub(rf"[ \t]+(?=[{_CJK}])", "", text)
    text = re.sub(rf"(?<=[{_CJK}])[ \t]+", "", text)
    return text


def normalize_pdf_text(text: str) -> str:
    """
    把 PDF 抽出來的文字整理成跟 Word 段落一樣的「連續文字」。

    PDF 沒有「段落」的概念，抽文字時依頁面寬度硬換行，換行可能切在號碼或地址中間
    （0912-\\n345-678、新生南路\\n三段），導致正則比對失敗。
    處理順序：先整理空白，再把「句子中間的硬換行」接回去；
    空行（連續兩個換行）視為真正的段落分隔，保留不動。
    """
    import re

    text = normalize_text_spacing(text)

    # 接回被硬換行切開的內容（單一換行；連續空行保留為段落分隔）
    #    - 兩側任一邊是中文：直接接起來
    #    - 左邊以連字號結尾、或右邊以連字號開頭、或兩側都是數字：直接接起來（號碼被切開）
    #    - 兩側都是英文單字：補一個空白（避免 word\nword 黏在一起）
    def _join(m):
        left, right = m.group(1), m.group(2)
        if re.match(rf"[{_CJK}]", left) or re.match(rf"[{_CJK}]", right):
            return left + right
        if left in "-–—－" or right in "-–—－" or (left.isdigit() and right.isdigit()):
            return left + right
        return left + " " + right

    text = re.sub(r"(\S)[ \t]*\n(?!\n)[ \t]*(\S)", _join, text)

    # 4. 連續 3 個以上換行壓成 2 個，避免大量空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pdf_reader_class():
    """優先用 pypdf（PyPDF2 的後繼版本，文字擷取的空白與換行判斷比舊版 PyPDF2 準確），沒有才退回 PyPDF2。"""
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader
    return PdfReader


def extract_text_from_pdf(raw: bytes) -> str:
    PdfReader = _pdf_reader_class()

    reader = PdfReader(io.BytesIO(raw))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return normalize_pdf_text(text)


# ---------------------------------------------------------------------------
# key-value 樣式表格偵測：有些「個人資料表」不是一欄一種資料的橫向表格，
# 而是直式的「每一列代表一個欄位」，例如：
#   欄位名稱 (Field)        | 資料內容 (Value)              | 備註 (Notes)
#   身分證字號/護照號碼 (ID) | A123456789                   | 機密資訊
#   聯絡電話 (Phone Number) | +886-912-345-678             | 行動電話
# 這種表格「資料內容」那一整欄混了姓名、生日、身分證、電話、地址…各種完全不同格式的值，
# 用「整欄判斷同一種類型」的邏輯會完全偵測不到（欄位名稱對不上關鍵字，逐列格式又兜不出
# 統一的正則命中率），因此需要另外偵測「這其實是一欄欄位名稱 + 一欄對應的值」的版型，
# 改成逐列用該列自己的欄位名稱來判斷這一列的值要套用什麼類型。
# ---------------------------------------------------------------------------
LABEL_COLUMN_HINTS = ["欄位名稱", "欄位", "field", "項目", "attribute", "key"]
VALUE_COLUMN_HINTS = ["資料內容", "內容", "值", "value", "data", "資料"]


def _find_label_value_columns(df: pd.DataFrame):
    """
    嘗試偵測「每一列代表一筆欄位資料」的 key-value 樣式表格。
    回傳 (label_col_idx, value_col_idx)；找不到則回傳 None。
    """
    cols = [str(c) for c in df.columns]

    # 1. 優先用欄位標題本身的關鍵字猜（最常見、最準確，例如「欄位名稱」「資料內容」）
    label_idx, value_idx = None, None
    for i, name in enumerate(cols):
        lname = name.strip().lower()
        if label_idx is None and any(h in lname for h in LABEL_COLUMN_HINTS):
            label_idx = i
        if value_idx is None and any(h in lname for h in VALUE_COLUMN_HINTS):
            value_idx = i
    if label_idx is not None and value_idx is not None and label_idx != value_idx:
        return label_idx, value_idx

    # 2. 退而求其次：統計哪一欄的「內容」本身大多能比對出好幾種不同的個資關鍵字
    #    （代表這欄放的其實是欄位名稱，不是欄位值本身）
    best_idx, best_ratio = None, 0.0
    for i in range(len(cols)):
        # 注意：不能靠 pandas 的 .astype(str) 保證拿到的一定是字串——pandas 3.x
        # 如果欄位本身的 dtype 已經是新版原生 str 型別，.astype(str) 會被當成
        # 「已經是字串了」直接跳過轉換，缺值儲存格仍然會是 Python 的 float('nan')，
        # 不是字串 "nan"，這時候再呼叫 .strip() 就會炸出
        # 'float' object has no attribute 'strip'。所以這裡改成逐一手動用 str() 轉換，
        # 不依賴 .astype(str) 幫忙處理缺值。
        non_empty = []
        for v in df.iloc[:, i]:
            sv = str(v).strip() if v is not None else ""
            if sv and sv.lower() != "nan":
                non_empty.append(sv)
        if len(non_empty) < 3:
            continue
        hit_types = [match_column_by_name(v) for v in non_empty]
        hits = [t for t in hit_types if t]
        ratio = len(hits) / len(non_empty)
        distinct = len(set(hits))
        if ratio >= 0.4 and distinct >= 3 and ratio > best_ratio:
            best_idx, best_ratio = i, ratio

    if best_idx is None:
        return None

    candidates = [i for i in range(len(cols)) if i != best_idx]
    if not candidates:
        return None
    value_idx = best_idx + 1 if (best_idx + 1) in candidates else candidates[0]
    return best_idx, value_idx


# ---------------------------------------------------------------------------
# 結構化資料去識別化：逐欄位偵測類型 -> 套用策略
# ---------------------------------------------------------------------------
def restore_phone_leading_zero(value, detected_type: str):
    """
    補回被 Excel 吃掉的電話開頭 0。

    Excel 把電話當成「數字」儲存時，開頭的 0 會消失：
      手機 0912345678 → 912345678（9 碼、9 開頭）
      台北市話 02-23456789 → 223456789（9 碼、2~8 開頭）
      苗栗市話 037-123456 → 37123456（8 碼、2~8 開頭）
    台灣的區碼不會是 9 開頭，所以：
      - 9 碼且 9 開頭 → 手機，補 0
      - 8～9 碼且 2~8 開頭 → 市話，補 0（補完 9～10 碼，符合市話長度）
    只處理「純數字（可含破折號、空白、括號、Excel 小數殘留 .0）」且「沒有 0 或 + 開頭」的值，
    其他格式一律原樣回傳，不亂猜。只在欄位／標籤已經被判斷為電話時才呼叫。
    回傳補完 0 的字串；沒有需要補就回傳原值。
    """
    if detected_type not in ("MOBILE", "PHONE") or value is None:
        return value
    text = str(value).strip()
    text = re.sub(r"^(\d+)\.0+$", r"\1", text)  # Excel／pandas 的浮點數殘留：912345678.0
    if not text or text[0] in "0+" or not re.fullmatch(r"[\d\s\-()]+", text):
        return value if text == str(value).strip() else text
    digits = re.sub(r"[\s\-()]", "", text)
    if not digits or digits[0] == "0":
        return text
    is_mobile = len(digits) == 9 and digits[0] == "9"
    is_landline = len(digits) in (8, 9) and digits[0] in "2345678"
    if not (is_mobile or is_landline):
        return text
    return "(0" + text[1:] if text.startswith("(") else "0" + text


# ---------------------------------------------------------------------------
# 沒被辨識出類型的欄位：逐格掃描 + 疑似識別欄位提示
#
# 欄位名稱、欄位內容抽樣都判斷不出類型的欄位（例如「備註」「聯絡方式」），內容可能其實夾帶
# 電話、Email、地址等個資。所以改成每一格都用文字規則掃一次，有發現才遮罩。
# 掃描時「刻意不含」銀行帳號、信用卡、生日：沒有欄位名稱當線索時，10~16 碼的純數字很可能只是
# 訂單編號、流水號，硬遮罩會把正常資料弄壞。也不跑 CKIP 姓名辨識（逐格呼叫模型太慢）。
# ---------------------------------------------------------------------------
SCAN_TYPES = {"EMAIL", "MOBILE", "PHONE", "TW_ID", "TW_ARC", "ADDRESS", "IP", "MAC", "PASSPORT", "NHI_CARD"}

# 疑似識別欄位：幾乎每格內容都不一樣、而且內容很短（編號、學號、帳號、姓名…），這種欄位通常
# 能直接指認出一個人，但沒有欄名或格式可以判斷，只提示、不自動遮罩，由使用者確認。
SUSPECT_MIN_ROWS = 5         # 至少要有這麼多筆非空白資料才判斷（太少筆沒意義）
SUSPECT_UNIQUE_RATIO = 0.9   # 不重複比例門檻
SUSPECT_MAX_AVG_LEN = 24     # 平均長度上限：太長的通常是自由文字（備註句子），不是識別碼


def _scan_unrecognized_column(values, overrides: dict):
    """
    逐格用文字規則掃描。回傳 (處理後的值 list, 有被處理的儲存格數, 發現的類型 set, 用到的策略 set)。
    沒有任何發現的儲存格原封不動（連空白都不改），有發現的才會經過空白整理後取代。
    """
    out, hit_cells, types_found, strategies = [], 0, set(), set()
    for v in values:
        text = "" if v is None else str(v)
        if not text.strip() or text.strip().lower() == "nan":
            out.append(v)
            continue
        norm = normalize_text_spacing(text)
        matches = detect_text_pii(norm, use_llm=False, use_ner=False, only_types=SCAN_TYPES)
        # 只算「真的有字被改掉」的比對結果：例如「臺北市松山區公所」的地址規則只抓到「臺北市松山區」，
        # 而地址遮罩本來就保留到「區」，結果內容完全沒變，不能算成「已遮罩」誤導使用者。
        changed = []
        for m in matches:
            strat = overrides.get(m[0], DEFAULT_STRATEGY.get(m[0], "mask"))
            if apply_strategy(m[3], strat, m[0]) != m[3]:
                changed.append(m)
                types_found.add(m[0])
                strategies.add(strat)
        if not changed:
            out.append(v)
            continue
        new_text, _ = _replace_matches(norm, changed, overrides)
        out.append(new_text)
        hit_cells += 1
    return out, hit_cells, types_found, strategies


def _suspect_identifier_note(values):
    """判斷這欄像不像識別碼，像的話回傳提示文字，不像回傳 None。"""
    texts = [str(v).strip() for v in values if v is not None and str(v).strip() and str(v).strip().lower() != "nan"]
    n = len(texts)
    if n < SUSPECT_MIN_ROWS:
        return None
    if all(re.fullmatch(r"\d{1,3}", t) for t in texts):
        return None  # 1、2、3… 這種流水序號不是識別碼
    distinct = len(set(texts))
    avg_len = sum(len(t) for t in texts) / n
    if distinct / n < SUSPECT_UNIQUE_RATIO or avg_len > SUSPECT_MAX_AVG_LEN:
        return None
    # 光是「每格都不一樣」會誤抓很多一般欄位（例如小表格裡每列剛好都不同的機關名稱），
    # 所以再要求內容長得像識別碼：大多含數字（編號、學號、帳號、員工證號…），
    # 或大多是 2~4 個中文字（像人名）。
    has_digit = sum(1 for t in texts if re.search(r"\d", t)) / n
    name_like = sum(1 for t in texts if re.fullmatch(r"[\u4e00-\u9fff]{2,4}", t)) / n
    if has_digit >= 0.8 or name_like >= 0.8:
        return f"疑似識別欄位：{n} 筆中有 {distinct} 筆互不相同，建議確認是否需要遮罩"
    return None


_KADAPT_ERROR = ""
try:  # k 匿名：演算法用現成的 ANJANA 套件，kanon_adapter.py 只是轉接層；沒裝時其他功能照常
    import kanon_adapter as kadapt
except Exception as _e:  # noqa: BLE001
    kadapt = None
    _KADAPT_ERROR = f"{type(_e).__name__}: {_e}"


def _kanon_unavailable_reason() -> str:
    """說明為什麼 k 匿名不能用（含實際的錯誤訊息和目前跑後端的 Python 路徑，方便判斷是不是裝在別的 Python）。"""
    import sys
    detail = _KADAPT_ERROR if kadapt is None else getattr(kadapt, "IMPORT_ERROR", "")
    return f"無法載入 k 匿名套件（ANJANA）。實際錯誤：{detail or '未知'}；目前後端使用的 Python：{sys.executable}"


def _apply_k_anonymity(df, result_df, report, sel, source: str, k_results: list, allow_delete: bool = True):
    """使用者在確認畫面勾「k 匿名」並指定準識別碼欄位時：對那些欄位用原始值交給 ANJANA 泛化到達標，
    取代已經處理過的結果；若需刪除少數列，結果表也同步刪列。回傳 (結果表, 報告)。"""
    sel = sel or {}
    if not sel.get("k_on") or sel.get("analyze"):
        return result_df, report
    prefix = f"{source}::"
    methods = sel.get("column_methods") or {}
    qcols = []
    for key in sel.get("k_quasi") or []:
        if not key.startswith(prefix):
            continue
        c = key.split("::", 1)[1]
        if c in df.columns and c in result_df.columns and (methods.get(key) or {}).get("method") not in ("drop", "hash", "redact"):
            qcols.append(c)
    if not qcols:
        return result_df, report
    k = max(2, int(sel.get("k_threshold") or 5))
    info_base = {"source": source, "threshold": k, "quasi_columns": qcols}
    if kadapt is None or not kadapt.AVAILABLE:
        k_results.append({**info_base, "error": _kanon_unavailable_reason()})
        return result_df, report
    r = kadapt.run_k_anonymity(
        df.reset_index(drop=True), qcols, k=k,
        suppress_pct=float(sel.get("k_suppress_pct", 10)) if allow_delete else 0.0,
        force_top=[key.split("::", 1)[1] for key in (sel.get("k_top") or []) if key.startswith(prefix)],
    )
    if not r.get("ok"):
        k_results.append({**info_base, "k_before": r.get("k_before"), "error": r.get("error") or "未能達到目標 k"})
        return result_df, report
    out = result_df.reset_index(drop=True).iloc[r["kept_index"]].reset_index(drop=True)
    for c in qcols:
        out[c] = r["df"][c].values
    out.attrs = dict(result_df.attrs)
    for rep in report:
        if rep["column"] in r["levels"]:
            rep["strategy"] = f"k匿名泛化（{r['levels'][rep['column']]}）"
            rep["quasi"] = True
    k_results.append({**info_base, "k_before": r["k_before"], "k": r["k_after"], "passed": True,
                      "levels": r["levels"], "suppressed_rows": r["suppressed_rows"], "original_rows": r["original_rows"]})
    return out, report


def deidentify_dataframe(df: pd.DataFrame, overrides: Optional[dict] = None, sel: Optional[dict] = None, source: str = ""):
    overrides = overrides or {}
    col_specs = _col_specs(sel, source)
    skip_cols = {k for k, v in col_specs.items() if v.get("method") == "keep"}
    analyze = bool((sel or {}).get("analyze"))
    drop_idx = []
    result_df = df.copy()
    report = []

    # 先處理 key-value 樣式的欄位／值配對（若有偵測到的話），逐列用該列自己的
    # 欄位名稱判斷類型；處理過的兩欄之後就跳過，不再用整欄邏輯重複處理一次。
    handled_cols = set()
    kv = _find_label_value_columns(df)
    if kv is not None:
        label_idx, value_idx = kv
        handled_cols = {label_idx, value_idx}
        value_col_name = str(df.columns[value_idx])

        for row_i in range(len(df)):
            label_text = str(df.iloc[row_i, label_idx] or "").strip()
            raw_value = df.iloc[row_i, value_idx]
            value_text = "" if raw_value is None else str(raw_value)

            if not label_text or value_text == "" or value_text.strip().lower() == "nan":
                continue

            detected_type = match_column_by_name(label_text)
            if not detected_type:
                continue  # 這個欄位名稱不是已知的個資類型，原樣保留
            if f"{value_col_name}［{label_text}］" in skip_cols:
                report.append(
                    {
                        "column": f"{value_col_name}［{label_text}］",
                        "detected_type": detected_type,
                        "strategy": "原樣保留",
                        "affected_rows": 0,
                        "invalid_rows": 0,
                        "processed": False,
                        "note": "依使用者選擇不遮罩",
                    }
                )
                continue

            strategy = (
                overrides.get(label_text)
                or overrides.get(detected_type)
                or DEFAULT_STRATEGY.get(detected_type, "mask")
            )

            raw_value = restore_phone_leading_zero(raw_value, detected_type)  # 補回被 Excel 吃掉的 0
            if is_valid_format(raw_value, detected_type):
                new_value = apply_strategy(raw_value, strategy, detected_type)
                invalid = 0
            else:
                new_value = "無效資料"
                invalid = 1

            result_df.iat[row_i, value_idx] = new_value
            report.append(
                {
                    "column": f"{value_col_name}［{label_text}］",
                    "detected_type": detected_type,
                    "strategy": strategy,
                    "affected_rows": 1 - invalid,
                    "invalid_rows": invalid,
                }
            )

    # 一般「整欄都是同一種類型」的欄位，照舊邏輯逐欄處理（排除掉上面已經用
    # key-value 模式處理過的欄位，避免同一欄被處理兩次）。
    # 用「位置」而不是「欄名」逐欄處理：就算欄名重複（例如簽到表合併儲存格造成
    # 兩欄都叫「簽名」），df.iloc[:, i] 一定能正確抓到單一欄，不會抓成一整個子表格。
    for i, col in enumerate(df.columns):
        if i in handled_cols:
            continue
        series = df.iloc[:, i]

        spec = col_specs.get(str(col))
        if spec and (spec.get("method") not in ("keep", None)):
            ctype = match_column_by_name(str(col)) or "—"
            bad = 0
            outs = []
            for v in series.tolist():
                nv, b = apply_custom_method(v, spec)
                bad += int(b)
                outs.append(nv)
            result_df.iloc[:, i] = outs
            if spec["method"] == "drop":
                drop_idx.append(i)
            nonblank = sum(1 for v in series.tolist() if str(v).strip() not in ("", "nan"))
            report.append(
                {
                    "column": str(col),
                    "detected_type": ctype,
                    "strategy": _custom_label(spec),
                    "affected_rows": nonblank - bad,
                    "invalid_rows": bad,
                    "custom": True,
                }
            )
            continue

        if str(col) in skip_cols:
            report.append(
                {
                    "column": str(col),
                    "detected_type": "—",
                    "strategy": "原樣保留",
                    "affected_rows": 0,
                    "invalid_rows": 0,
                    "processed": False,
                    "note": "依使用者選擇不遮罩",
                }
            )
            continue

        # 1. 先用欄位名稱判斷類型，判斷不出來再抽樣欄位內容
        detected_type = match_column_by_name(str(col))
        if not detected_type:
            # 直接傳原始值給 guess_column_by_sample，讓它自己用 str() 處理每一個值
            # （不要先 .astype(str)：pandas 3.x 對已經是 str dtype 的欄位，.astype(str)
            # 可能直接跳過不轉換，缺值儲存格會維持 float('nan') 而不是字串）
            detected_type = guess_column_by_sample(series.tolist())
        if not detected_type and _guess_name_column_by_sample(series.tolist()):
            # 姓名沒有固定格式，PATTERNS 裡沒有對應的正則規則可以用內容抽樣判斷，
            # 平常都是靠欄位名稱（例如欄名叫「姓名」）才抓得到。但沒有表頭、直接
            # 條列個資的資料完全沒有欄位名稱可以比對，這時候最後再用 CKIP NER
            # 輔助判斷：如果這一欄抽樣起來幾乎整格都被辨識成人名，就當作姓名欄。
            detected_type = "NAME"

        if not detected_type:
            # 判斷不出類型的欄位：先逐格掃描有沒有夾帶個資（例如備註裡寫了電話）
            scanned, hit_cells, found_types, found_strategies = _scan_unrecognized_column(series.tolist(), overrides)
            if hit_cells:
                result_df.iloc[:, i] = scanned
                report.append(
                    {
                        "column": str(col),
                        "detected_type": "文字掃描：" + "、".join(sorted(found_types)),
                        "strategy": "、".join(sorted(found_strategies)),
                        "affected_rows": hit_cells,
                        "invalid_rows": 0,
                        "scanned": True,
                    }
                )
                continue

            # 掃描後也沒有發現：原樣保留，但要在報告裡列出來，讓使用者知道有哪些欄位沒被處理
            # （processed=False 讓網頁用不同樣式顯示）
            all_empty = all(str(v).strip() in ("", "nan") for v in series.tolist())
            entry = {
                "column": str(col),
                "detected_type": "—",
                "strategy": "原樣保留",
                "affected_rows": 0,
                "invalid_rows": 0,
                "processed": False,
                "note": "整欄空白" if all_empty else "已逐格掃描，未發現個資",
            }
            suspect = None if all_empty else _suspect_identifier_note(series.tolist())
            if suspect:
                entry["suspect"] = True
                entry["note"] = suspect
            report.append(entry)
            continue

        # 2. 決定策略：優先用欄位名稱的覆寫，其次用類型的覆寫，最後用預設值
        strategy = overrides.get(col) or overrides.get(detected_type) or DEFAULT_STRATEGY.get(detected_type, "mask")

        invalid_count = 0
        zero_restored = 0

        def process_cell(v):
            nonlocal invalid_count, zero_restored
            text = str(v) if v is not None else ""
            if text == "" or text.lower() == "nan":
                return v  # 空值不處理、不算無效
            fixed = restore_phone_leading_zero(v, detected_type)  # 補回被 Excel 吃掉的 0
            if str(fixed) != text:
                zero_restored += 1
                v = fixed
            if not is_valid_format(v, detected_type):
                invalid_count += 1
                return "無效資料"
            return apply_strategy(v, strategy, detected_type)

        result_df.iloc[:, i] = series.apply(process_cell)

        entry = {
            "column": str(col),
            "detected_type": detected_type,
            "strategy": strategy,
            "affected_rows": int(df[col].notna().sum()) - invalid_count,
            "invalid_rows": invalid_count,
        }
        if zero_restored:
            entry["zero_restored_rows"] = zero_restored  # 有幾列的開頭 0 是被補回來的
        report.append(entry)

    if drop_idx:
        attrs = dict(result_df.attrs)
        result_df = result_df.iloc[:, [j for j in range(result_df.shape[1]) if j not in drop_idx]]
        result_df.attrs.update(attrs)

    if analyze:
        # 確認畫面用：每欄附幾個範例值（每格最多 20 字），讓使用者看得出欄位是什麼
        by_name = {}
        for j, c in enumerate(df.columns):
            by_name.setdefault(str(c), j)
        for r in report:
            j = by_name.get(r["column"])
            if j is None:
                continue
            seen = []
            for v in df.iloc[:, j].tolist():
                t = str(v).strip()
                if t and t.lower() != "nan" and t not in seen:
                    seen.append(t)
                if len(seen) >= 3:
                    break
            r["samples"] = [t[:20] for t in seen]

    return result_df, report


# ---------------------------------------------------------------------------
# 非結構化文字去識別化：正則偵測 PII -> 套用策略並回填原文
# ---------------------------------------------------------------------------
def deidentify_text(text: str, overrides: Optional[dict] = None, use_llm: bool = False):
    overrides = overrides or {}
    matches = detect_text_pii(text, use_llm=use_llm)  # [(type, start, end, original), ...]

    result_text = text
    # 從後往前取代，避免前面的取代改變後面字元的位置
    for pii_type, start, end, original in sorted(matches, key=lambda m: m[1], reverse=True):
        strategy = overrides.get(pii_type, DEFAULT_STRATEGY.get(pii_type, "mask"))
        replacement = apply_strategy(original, strategy, pii_type)
        result_text = result_text[:start] + replacement + result_text[end:]

    report = []
    for pii_type, start, end, original in matches:
        strategy = overrides.get(pii_type, DEFAULT_STRATEGY.get(pii_type, "mask"))
        report.append(
            {
                "type": pii_type,
                "original_preview": mask_preview(original, pii_type, strategy),  # 報告裡也不顯示完整原文
                "strategy": strategy,
            }
        )

    return result_text, report


# ---------------------------------------------------------------------------
# 處理進度追蹤：讓前端可以用「先送出、再輪詢進度」的方式顯示真實的後端處理進度，
# 而不是單一個請求整個處理完才一次回傳（那樣中途完全沒辦法回報進度）。
#
# JOBS 是一份存在記憶體裡的工作狀態表（job_id -> 狀態），用一個背景執行緒實際跑
# 處理邏輯，處理過程中會不斷更新這份狀態，前端再用 job_id 定期來問目前進度。
#
# 注意：這是單一程序的記憶體內儲存，如果之後把 uvicorn 開成多個 worker 程序，
# 每個 worker 會有自己獨立的一份 JOBS，跨 worker 查不到彼此的工作——這個開發用的
# 單一伺服器沒問題，但要正式多工部署的話，進度狀態需要改放到 Redis 之類的共用儲存。
# ---------------------------------------------------------------------------
JOBS_LOCK = threading.Lock()
JOBS: dict = {}


def _set_job(job_id: str, **kwargs):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)


def _cleanup_job_later(job_id: str, delay_seconds: int = 600):
    """工作完成（成功或失敗）後，過一段時間自動把紀錄清掉，避免記憶體一直累積。"""
    timer = threading.Timer(delay_seconds, lambda: JOBS.pop(job_id, None))
    timer.daemon = True
    timer.start()


# ---------------------------------------------------------------------------
# 「系統建議」項目：不在預設遮罩範圍內，但可能是敏感資料（病歷號、病症、用藥、收入…）。
# 只在「先分析再勾選」的流程中列出來給使用者參考，預設不遮罩；使用者勾選了才會遮罩。
# 沒走確認流程（直接輸出）時完全不會碰這些內容。
# 想增加項目：在 SUGGEST_LABELS 加一行「類型名稱: 欄位標籤」，或在 SUGGEST_KEYWORDS 加關鍵字。
# ---------------------------------------------------------------------------
_SUGGEST_VALUE_ID = r"[A-Za-z0-9][A-Za-z0-9\-_/]{2,24}"
_SUGGEST_VALUE_TEXT = r"[^\n；;。]{1,40}"
SUGGEST_LABELS = {
    "病歷號": (r"病歷(?:號碼|編號|號)|病例號|病患編號|病人編號|就診序號|門診號|住院號", _SUGGEST_VALUE_ID),
    "診斷／病症": (r"診斷|病名|病症|症狀|疾病|主訴|病史|過敏史|過敏|病況", _SUGGEST_VALUE_TEXT),
    "用藥": (r"用藥|藥物|處方|藥名", _SUGGEST_VALUE_TEXT),
    "手術／治療": (r"手術|治療|療程", _SUGGEST_VALUE_TEXT),
    "收入／薪資": (r"年收入|月收入|收入|月薪|年薪|薪資|薪水", r"[^\n，,；;。]{1,20}"),
    "性別": (r"性別", r"[男女]"),
    "學號／員編": (r"學號|員工編號|員編|工號", _SUGGEST_VALUE_ID),
    "車牌": (r"車牌(?:號碼)?", r"[A-Za-z0-9]{2,4}[-－]?[A-Za-z0-9]{2,4}"),
}
SUGGEST_KEYWORDS = {
    "診斷／病症": [
        "高血壓", "糖尿病", "心臟病", "心肌梗塞", "氣喘", "憂鬱症", "躁鬱症", "思覺失調", "焦慮症", "失智症",
        "癲癇", "中風", "腎衰竭", "洗腎", "愛滋", "HIV", "B型肝炎", "C型肝炎", "肝硬化", "痛風", "癌症",
        "肺癌", "肝癌", "胃癌", "乳癌", "大腸癌", "直腸癌", "子宮頸癌", "卵巢癌", "攝護腺癌", "甲狀腺癌",
        "胰臟癌", "食道癌", "鼻咽癌", "口腔癌", "膀胱癌", "腎臟癌", "皮膚癌", "血癌", "腦瘤", "慢性胃潰瘍",
        "支氣管炎", "肺炎", "氣胸", "骨折",
    ],
}
SUGGEST_TYPES = set(SUGGEST_LABELS) | set(SUGGEST_KEYWORDS)
_NARRATIVE_GENDER_RE = re.compile(r"[\u4e00-\u9fff]{2,4}\s*[，,（(]\s*(?P<g>男|女)\s*[，,）);；。]")

# 通用「欄位標籤：值」自動探索：不管標籤是什麼（血型、病歷號、宗教、案號、保單號碼…），
# 文件裡只要長得像「標籤：值」就列出來，依標籤分組讓使用者自己決定要不要遮、怎麼遮。
# 上面 SUGGEST_LABELS 只是用來把「比較可能敏感」的標籤排在前面，不是篩選條件。
_GENERIC_LABEL = r"[一-鿿A-Za-z][一-鿿A-Za-z0-9／/（）()_]{0,11}"
_GENERIC_RE = re.compile(
    rf"(?:(?<=^)|(?<=[\n，,；;。、\s]))(?P<label>{_GENERIC_LABEL})\s*[:：]\s*"
    rf"(?P<v>(?:(?!\s+{_GENERIC_LABEL}\s*[:：])[^\n；;。]){{1,40}})",
    re.M,
)
# 敘述句裡的「編號類」內容：「病歷號碼是 MRN-4567890」「確診代碼 ICD-10-CM I49.8」「員工工號是 EMP-9527」
# 規則：某個以「號碼／代碼／編號／字號／序號／工號／學號／ID／Token…」結尾的詞，後面（可隔「是／為／寫著」）
# 緊接著一串含數字的編號，就列為建議項目，標籤詞本身就是類型名稱。
_ID_NOUN = r"(?:號碼|號|代碼|代號|編號|字號|序號|識別碼|識別ID|ID|Token|權杖|碼)"
_ID_LABEL_RE = re.compile(
    rf"(?P<label>[一-鿿A-Za-z]{{1,9}}?{_ID_NOUN}(?:[（(][A-Za-z]{{2,10}}[）)])?)\s*(?:是|為|寫著|則維持在|維持在|：|:|=)?\s*[「（(]?"
    r"(?P<v>(?=[A-Za-z0-9\-_.=]*\d)[A-Za-z0-9@][A-Za-z0-9\-_.=@]{2,40}(?: (?=[A-Za-z0-9.\-]*\d)[A-Za-z0-9][A-Za-z0-9.\-]*)?)"
)
_LABEL_FILLERS = re.compile(r"[的與及和我是為在有於把將從向給對]")
_LIKELY_LABEL_RE = re.compile("|".join(l for l, _ in SUGGEST_LABELS.values()))
_SUGGESTED_SEEN: set = set()   # 本程序偵測過的所有建議類型（含自動探索到的標籤）
_SUGGESTED_LIKELY: set = set()  # 其中「較可能敏感」的
_GENERIC_TYPES: set = set()     # 自動探索（非固定清單）的標籤類型


def is_suggested_type(t: str) -> bool:
    return t in SUGGEST_TYPES or t in _SUGGESTED_SEEN


def detect_suggested(text: str):
    """回傳 [(建議類型, 起點, 終點, 原文), ...]：標籤後面的內容（例如「病歷號：A0012345」只取後面的值）＋關鍵字。"""
    found = []
    for stype, (labels, value) in SUGGEST_LABELS.items():
        for m in re.finditer(rf"(?:{labels})\s*(?P<conn>[:：]|則維持在|維持在|是|為|寫著)\s*(?P<v>{value})", text):
            v_start, v_end = m.start("v"), m.end("v")
            if value in (_SUGGEST_VALUE_ID,):
                parts = [(v_start, text[v_start:v_end])]
            else:  # 文字型的值（例如「診斷：高血壓、糖尿病」）依頓號、逗號、斜線拆成各自一項
                parts, pos = [], v_start
                for piece in re.split(r"([、,，/])", text[v_start:v_end]):
                    if piece not in ("、", ",", "，", "/"):
                        parts.append((pos, piece))
                    pos += len(piece)
                if m.group("conn") not in (":", "："):
                    parts = parts[:1]  # 句子裡的「診斷為子宮肌瘤，處方…」只取逗號前的第一段，避免把整句抓進來
            for ps, piece in parts:
                val = piece.strip(" 　")
                if val:
                    lead = len(piece) - len(piece.lstrip(" 　"))
                    found.append((stype, ps + lead, ps + lead + len(val), val))
    for stype, words in SUGGEST_KEYWORDS.items():
        for w in sorted(words, key=len, reverse=True):
            for m in re.finditer(re.escape(w), text):
                found.append((stype, m.start(), m.end(), w))
    # 通用探索：任何「標籤：值」。固定清單已經抓到的位置不重複列。
    fixed_spans = [(m[1], m[2]) for m in found]
    for m in _GENERIC_RE.finditer(text):
        label = m.group("label").strip()
        if label.upper() in DEFAULT_STRATEGY:
            label = "欄位：" + label  # 避免跟內建類型名稱（例如 IP、MAC）撞名
        gtype = label
        v_start = m.start("v")
        pos = v_start
        for piece in re.split(r"([、,，/])", m.group("v")):
            if piece not in ("、", ",", "，", "/"):
                val = piece.strip(" 　")
                if val:
                    lead = len(piece) - len(piece.lstrip(" 　"))
                    a, b = pos + lead, pos + lead + len(val)
                    if not any(a < fb and fa < b for fa, fb in fixed_spans):
                        found.append((gtype, a, b, val))
                        _GENERIC_TYPES.add(gtype)
                        if _LIKELY_LABEL_RE.fullmatch(label) or re.fullmatch(r"[A-Za-z]*\d[A-Za-z0-9\-_/]{3,}", val):
                            _SUGGESTED_LIKELY.add(gtype)
            pos += len(piece)
    # 敘述句裡沒有「性別」標籤的寫法：「林冠廷，男，…」「黃雅惠（女，…）」（姓名後面緊接著一個男／女）
    for m in _NARRATIVE_GENDER_RE.finditer(text):
        found.append(("性別", m.start("g"), m.end("g"), m.group("g")))
    # 敘述句裡的編號類內容（病歷號碼是…、確診代碼…、員工工號是…）
    all_spans = fixed_spans + [(m[1], m[2]) for m in found]
    for m in _ID_LABEL_RE.finditer(text):
        label = _LABEL_FILLERS.split(m.group("label"))[-1]
        label = re.sub(r"^[A-Za-z]+(?=[一-鿿])", "", label)  # 去掉黏在中文前面的英文殘字（例如 Cookie 被截成 kie）
        if len(label) > 10:
            label = label[-10:]
        if len(label) < 2:
            continue
        a, b = m.start("v"), m.end("v")
        val = text[a:b].rstrip(". ")
        b = a + len(val)
        if len(val) < 3 or any(a < fb and fa < b for fa, fb in all_spans):
            continue
        if label.upper() in DEFAULT_STRATEGY:
            label = "欄位：" + label
        found.append((label, a, b, val))
        _GENERIC_TYPES.add(label)
        _SUGGESTED_LIKELY.add(label)
        all_spans.append((a, b))
    for m in found:
        _SUGGESTED_SEEN.add(m[0])
        if m[0] in SUGGEST_TYPES:
            _SUGGESTED_LIKELY.add(m[0])
    # 去重疊：先取起點最前、較長的
    found.sort(key=lambda m: (m[1], -(m[2] - m[1])))
    out, last_end = [], -1
    for m in found:
        if m[1] >= last_end:
            out.append(m)
            last_end = m[2]
    return out


def _merge_suggested(text: str, matches: list, sel: Optional[dict], llm_extra: Optional[list] = None) -> list:
    """分析模式：加入全部建議項目；輸出模式：只加入使用者勾選（include_items）的建議項目。"""
    if not sel:
        return matches
    analyze = bool(sel.get("analyze"))
    inc = set(sel.get("include_items", []))
    if not analyze and not inc:
        return matches
    extra = []
    cands = detect_suggested(text)
    for lt, a, b, w in (llm_extra or []):  # 本機 LLM 自己判斷出來的敏感內容，規則抓到的位置優先
        if any(a < cb and ca < b for _, ca, cb, _ in cands):
            continue
        cands.append((lt, a, b, w))
        _GENERIC_TYPES.add(lt)
        _SUGGESTED_SEEN.add(lt)
        _SUGGESTED_LIKELY.add(lt)
    for m in cands:
        if not analyze and _item_id(m[0], m[3]) not in inc:
            continue
        overl = [b for b in matches if m[1] < b[2] and b[1] < m[2]]
        if overl and m[0] in _GENERIC_TYPES:
            continue  # 自動探索到的項目：預設規則已經處理的位置一律以預設規則為準
        if overl and not all(m[1] <= b[1] and b[2] <= m[2] for b in overl):
            continue  # 只有部分重疊，不處理，避免取代位置錯亂
        if overl:
            # 預設規則抓到的是建議項目的一部分（例如病歷號 A0012345 被誤認成電話）：以建議項目為準。
            # 使用者沒勾選時，輸出階段不會加入建議項目，預設規則的結果就原樣保留。
            matches = [b for b in matches if b not in overl]
        extra.append(m)
    return sorted(matches + extra, key=lambda m: m[1])


# 偵測結果快取：「先分析、讓使用者勾選、再輸出」會對同一份文字跑兩次，第二次直接用第一次的結果，
# 不用再等一次（尤其是 LLM），也確保兩次看到的項目完全一致。
# 資安考量：
#  - 快取只存「類型、起點、終點」，不存原文（個資）；命中時再用「這次上傳的文字」切出原文。
#    （快取的鍵是整份文字的 SHA-1，所以命中就代表文字完全相同，位置一定對得上。）
#  - 每筆最多保留 DEID_CACHE_TTL_SECONDS 秒（預設 600 = 10 分鐘），過期就清掉；設成 0 代表完全不快取。
#  - 只存在記憶體，不寫硬碟，後端重啟就清空。
CACHE_TTL_SECONDS = int(os.environ.get("DEID_CACHE_TTL_SECONDS", "600"))
CACHE_MAX_ENTRIES = 8


class _TTLCache:
    def __init__(self):
        self._d: dict = {}
        self._lock = threading.Lock()

    def _purge(self):
        now = time.time()
        for k in [k for k, (t, _) in self._d.items() if now - t > CACHE_TTL_SECONDS]:
            self._d.pop(k, None)

    def get(self, key):
        if CACHE_TTL_SECONDS <= 0:
            return None
        with self._lock:
            self._purge()
            hit = self._d.get(key)
            return hit[1] if hit else None

    def put(self, key, value):
        if CACHE_TTL_SECONDS <= 0:
            return
        with self._lock:
            self._purge()
            self._d[key] = (time.time(), value)
            while len(self._d) > CACHE_MAX_ENTRIES:
                self._d.pop(next(iter(self._d)))

    def clear(self):
        with self._lock:
            self._d.clear()


def _strip_originals(text: str, matches: list):
    """把 [(類型, 起, 迄, 原文)] 轉成 [(類型, 起, 迄)]；只要有任何一筆原文和 text[起:迄] 不一致，就回傳 None（不快取）。"""
    out = []
    for m in matches:
        t, a, b, w = m[0], m[1], m[2], m[3]
        if text[a:b] != w:
            return None
        out.append((t, a, b))
    return out


def _restore_originals(text: str, slim: list) -> list:
    return [(t, a, b, text[a:b]) for t, a, b in slim]


_DETECT_CACHE = _TTLCache()
_ITEM_SECRET = os.urandom(16)


def _item_id(pii_type: str, original: str) -> str:
    """同類型＋同原文 = 同一個「項目」。用只在本程序有效的密鑰做 HMAC，網頁拿到的 id 無法反推原文。"""
    return hmac.new(_ITEM_SECRET, f"{pii_type}|{original}".encode("utf-8"), hashlib.sha256).hexdigest()[:10]


def _col_specs(sel: Optional[dict], source: str) -> dict:
    """取出使用者替某個來源（例如「表格 1」）的各欄選的處理方式：{欄名: {"method": ...}}。"""
    prefix = f"{source}::"
    specs = {}
    for k in (sel or {}).get("exclude_columns", []):  # 舊版相容：取消勾選 = 不處理
        if k.startswith(prefix):
            specs[k[len(prefix):]] = {"method": "keep"}
    for k, v in ((sel or {}).get("column_methods") or {}).items():
        if k.startswith(prefix) and isinstance(v, dict) and v.get("method") in CUSTOM_METHOD_LABELS:
            specs[k[len(prefix):]] = v
    return specs


def _docx_sel(sel: Optional[dict]) -> Optional[dict]:
    """Word 表格是依位置寫回的，不能刪欄；「刪除整欄」在 Word 裡改成把內容換成 [已刪除]。"""
    if not sel or not sel.get("column_methods"):
        return sel
    cm = {k: ({**v, "method": "redact"} if v.get("method") == "drop" else v) for k, v in sel["column_methods"].items()}
    return {**sel, "column_methods": cm}


_SENS_CACHE = _TTLCache()


def _llm_sensitive_cached(text: str) -> list:
    """本機 LLM 的「敏感內容探索」結果（快取：分析與輸出兩階段只跑一次）。失敗就回傳空清單。"""
    key = hashlib.sha1(text.encode("utf-8")).hexdigest()
    slim = _SENS_CACHE.get(key)
    if slim is not None:
        return _restore_originals(text, slim)
    try:
        from llm_detector import detect_sensitive_with_ollama

        res = []
        for cat, a, b, w in detect_sensitive_with_ollama(text):
            if cat.upper() in DEFAULT_STRATEGY:
                cat = "LLM：" + cat
            res.append((cat, a, b, w))
    except Exception as e:  # noqa: BLE001
        print(f"[app] LLM 敏感內容探索失敗，略過：{e}")
        res = []
    slim = _strip_originals(text, res)
    if slim is not None:  # 原文和位置對不上的（理論上不會發生）就不快取
        _SENS_CACHE.put(key, slim)
    return res


def _detect_with_progress(job_id, text: str, use_llm: bool, base_progress: int = 35, sel: Optional[dict] = None):
    """
    對一段文字做 PII 偵測（含進度回報），回傳 matches：[(類型, 起點, 終點, 原文), ...]。
    job_id 為 None 時（同步 API）不回報進度。
    LLM 輔助偵測那段如果開啟，因為是單一次呼叫、中途真的沒有進度可以回報，
    改用一個背景「心跳」執行緒緩慢把進度往上推，讓使用者知道「還在跑、沒有卡住」，
    而不是讓進度條停在原地一動也不動。
    """
    cache_key = (hashlib.sha1(text.encode("utf-8")).hexdigest(), bool(use_llm))
    want_llm = bool(use_llm and sel and (sel.get("analyze") or sel.get("include_items")))
    slim = _DETECT_CACHE.get(cache_key)
    if slim is not None:
        _set_job(job_id, progress=min(base_progress + 50, 96), stage="套用去識別化策略")
        extra = _llm_sensitive_cached(text) if want_llm else None
        return _merge_suggested(text, _restore_originals(text, slim), sel, extra)

    _set_job(job_id, progress=base_progress, stage="執行正則規則與姓名辨識")

    ticker_stop = threading.Event()
    ticker_thread = None

    if use_llm:
        tick_start = base_progress + 10
        tick_cap = min(base_progress + 45, 95)
        _set_job(job_id, progress=tick_start, stage="本機 LLM 輔助偵測中（第一次使用模型可能較慢，請耐心等候）")

        def _ticker():
            p = tick_start
            while not ticker_stop.wait(2):  # 每 2 秒推進一次，wait() 回傳 True 代表被提早喚醒（已經做完了）
                if p >= tick_cap:
                    break
                p += 1
                _set_job(job_id, progress=p, stage="本機 LLM 輔助偵測中（第一次使用模型可能較慢，請耐心等候）")

        ticker_thread = threading.Thread(target=_ticker, daemon=True)
        ticker_thread.start()

    try:
        matches = detect_text_pii(text, use_llm=use_llm)
        llm_extra = _llm_sensitive_cached(text) if want_llm else None  # 敏感內容探索（也在進度心跳保護下）
    finally:
        if ticker_thread:
            ticker_stop.set()
            ticker_thread.join(timeout=1)

    slim = _strip_originals(text, matches)
    if slim is not None:
        _DETECT_CACHE.put(cache_key, slim)
    _set_job(job_id, progress=min(base_progress + 50, 96), stage="套用去識別化策略")
    return _merge_suggested(text, list(matches), sel, llm_extra)


def _replace_matches(text: str, matches, overrides: dict, sel: Optional[dict] = None):
    """
    把偵測到的片段依策略取代掉，回傳 (去識別化後文字, 報告)。
    sel（使用者的勾選結果）：exclude_items 內的項目不取代、也不列進報告；
    sel["analyze"] 為真時，報告每筆多附一小段前後文，供「確認畫面」辨識是哪一處。
    """
    excl = set((sel or {}).get("exclude_items", []))
    if excl:
        matches = [m for m in matches if _item_id(m[0], m[3]) not in excl]
    result_text = text
    tm = (sel or {}).get("type_methods") or {}  # 使用者替某個類型自訂的遮罩（保留前 n、後 m 字）

    def _apply(pii_type, original, strategy):
        spec = tm.get(pii_type)
        if spec and spec.get("method") == "mask":
            return apply_custom_method(original, spec)[0]
        return apply_strategy(original, strategy, pii_type)

    for pii_type, start, end, original in sorted(matches, key=lambda m: m[1], reverse=True):
        strategy = overrides.get(pii_type, DEFAULT_STRATEGY.get(pii_type, "mask"))
        replacement = _apply(pii_type, original, strategy)
        result_text = result_text[:start] + replacement + result_text[end:]

    report = []
    for pii_type, start, end, original in matches:
        strategy = overrides.get(pii_type, DEFAULT_STRATEGY.get(pii_type, "mask"))
        entry = {
            "type": pii_type,
            "original_preview": (
                _apply(pii_type, original, strategy) if pii_type in tm else mask_preview(original, pii_type, strategy)
            ),
            "strategy": strategy,
            "id": _item_id(pii_type, original),
            **({"suggested": True, **({"likely": True} if pii_type in _SUGGESTED_LIKELY else {})} if is_suggested_type(pii_type) else {}),
        }
        if (sel or {}).get("analyze"):
            if is_suggested_type(pii_type):
                entry["original_preview"] = original[:30]  # 建議項目本來就沒有遮罩預覽可看，確認畫面直接顯示偵測到的原詞
            # 前後文裡的數字與英文字母一律以 • 遮掉，避免相鄰的其他個資在確認畫面上露出來
            ctx = (
                text[max(0, start - 10):start].replace("\n", " ")
                + "［　］"
                + text[end:end + 10].replace("\n", " ")
            )
            entry["ctx"] = re.sub(r"[0-9A-Za-z]", "•", ctx)
        report.append(entry)
    return result_text, report


def _run_text_pipeline(job_id: str, text: str, overrides: dict, use_llm: bool, base_progress: int = 35, sel=None):
    """處理非結構化文字（含進度回報）。邏輯跟 deidentify_text() 一樣。"""
    matches = _detect_with_progress(job_id, text, use_llm, base_progress, sel)
    return _replace_matches(text, matches, overrides, sel)


# ---------------------------------------------------------------------------
# 非結構化檔案「輸入什麼格式就輸出什麼格式」：.docx → .docx、.pdf → .pdf、.txt → .txt
#
# 安全考量（為什麼不是直接改原檔）：
#   Word 的頁首頁尾、文字方塊、註解、文件屬性（作者…）、PDF 內的圖片等位置，目前都沒有做偵測。
#   如果直接在原檔上取代文字，這些沒處理到的地方的個資會原封不動留在輸出檔裡。
#   所以改成「依原文件順序，用處理過的內容重新產生一份新檔」：段落樣式、對齊、粗斜體、表格
#   與表頭合併會保留，但頁首頁尾、文字方塊、圖片、註解不會帶過去，也就不會洩漏。
# ---------------------------------------------------------------------------
def _copy_run_format(src_par, dst_run):
    """把來源段落第一個有字的 run 的字型格式（粗體、斜體、底線、字級）複製到新 run。"""
    src_run = next((r for r in src_par.runs if r.text.strip()), None)
    if src_run is None:
        return
    try:
        dst_run.bold = src_run.bold
        dst_run.italic = src_run.italic
        dst_run.underline = src_run.underline
        if src_run.font.size:
            dst_run.font.size = src_run.font.size
    except Exception:  # noqa: BLE001 - 格式複製失敗不影響內容
        pass


def _cell_str(v) -> str:
    s = "" if v is None else str(v)
    return "" if s.strip().lower() == "nan" else s


def build_deidentified_docx(raw: bytes, overrides: dict, use_llm: bool, k_results: Optional[list] = None, job_id=None, sel=None):
    """
    讀入 .docx，回傳 (去識別化後的 .docx 位元組, 報告, data_type)。
    - 有真正表格物件的（≥2 列且判斷得出表頭）走結構化流程，逐欄偵測；
    - 其他表格（例如只有 1 列）的儲存格文字、以及所有段落，併入同一份文字做非結構化偵測。
    """
    import docx
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    _set_job(job_id, progress=12, stage="解析 Word 文件（表格與段落文字）")
    src = docx.Document(io.BytesIO(raw))

    # 1. 依文件順序收集段落與表格
    items = []
    for child in src.element.body.iterchildren():
        if child.tag == qn("w:p"):
            items.append({"kind": "p", "par": Paragraph(child, src)})
        elif child.tag == qn("w:tbl"):
            items.append({"kind": "t", "tbl": Table(child, src)})

    report = []
    units = []  # 所有要走「文字偵測」的片段（段落、以及沒走結構化的表格儲存格內的段落）

    # 2. 表格：能走結構化的先處理；不能的，把儲存格文字加進 units
    table_items = [it for it in items if it["kind"] == "t"]
    structured_total = 0
    for it in table_items:
        grid = [[c for c in row.cells] for row in it["tbl"].rows]
        rows = [[c.text for c in r] for r in grid]
        it["grid"], it["rows"] = grid, rows
        df = _rows_to_dataframe_with_header_detection(rows) if len(rows) >= 2 else None
        it["df"] = df if (df is not None and not df.empty) else None
        structured_total += it["df"] is not None

    done_tables = 0
    for it in table_items:
        if it["df"] is not None:
            done_tables += 1
            _set_job(job_id, progress=15 + int(35 * done_tables / max(structured_total, 1)), stage=f"處理表格 {done_tables}/{structured_total}")
            result_df, table_report = deidentify_dataframe(it["df"], overrides, _docx_sel(sel), f"表格 {done_tables}")
            result_df, table_report = _apply_k_anonymity(it["df"], result_df, table_report, sel, f"表格 {done_tables}", k_results if k_results is not None else [], allow_delete=False)
            for r in table_report:
                r["source"] = f"表格 {done_tables}"
            report.extend(table_report)
            it["result_df"] = result_df
            it["hrc"] = _detect_header_row_count(it["rows"])
        else:
            # 不走結構化的表格：每個「不重複的儲存格」內的每個段落，各自成為一個文字片段
            it["cell_units"] = {}
            seen = set()
            for r, row in enumerate(it["grid"]):
                for c, cell in enumerate(row):
                    if id(cell._tc) in seen:
                        continue
                    seen.add(id(cell._tc))
                    idxs = []
                    for par in cell.paragraphs:
                        units.append(par.text)
                        idxs.append(len(units) - 1)
                    it["cell_units"][(r, c)] = (cell, idxs)

    for it in items:
        if it["kind"] == "p":
            units.append(it["par"].text)
            it["unit"] = len(units) - 1

    # 3. 文字偵測：所有片段用空行（\n\n）串成一份，只跑一次（LLM 只呼叫一次），
    #    再把偵測到的位置對應回各個片段。用 \n\n 而不是 \n 串接，是因為號碼規則允許數字間有一個
    #    空白字元（含換行），單一換行可能讓兩個相鄰片段的數字被誤接成一組。
    norm_units = [normalize_text_spacing(u) for u in units]
    offsets, pos = [], 0
    for u in norm_units:
        offsets.append(pos)
        pos += len(u) + 2
    joined = "\n\n".join(norm_units)

    out_units = list(norm_units)
    if joined.strip():
        matches = _detect_with_progress(job_id, joined, use_llm, base_progress=55, sel=sel)
        per_unit = {}
        import bisect

        for m in matches:
            ui = bisect.bisect_right(offsets, m[1]) - 1
            if ui < 0 or m[2] > offsets[ui] + len(norm_units[ui]):
                continue  # 跨片段的比對結果（理論上用 \n\n 隔開後幾乎不會發生），略過
            per_unit.setdefault(ui, []).append((m[0], m[1] - offsets[ui], m[2] - offsets[ui], m[3]))
        for ui, ms in per_unit.items():
            out_units[ui], _ = _replace_matches(norm_units[ui], ms, overrides, sel)
        _, text_report = _replace_matches(joined, matches, overrides, sel)
        report.extend(text_report)

    # 4. 產生新的 Word 文件
    _set_job(job_id, progress=96, stage="產生 Word 輸出檔")
    out = docx.Document()
    out.core_properties.author = ""
    out.core_properties.last_modified_by = ""
    out.core_properties.title = ""
    normal = out.styles["Normal"]
    rfonts = normal.element.get_or_add_rPr().get_or_add_rFonts()
    rfonts.set(qn("w:eastAsia"), "Microsoft JhengHei")  # 中文字型，避免顯示成預設的英文字型

    def _write_par(dst_par, src_par, text):
        try:
            dst_par.style = out.styles[src_par.style.name]
        except Exception:  # noqa: BLE001 - 找不到對應樣式就用預設
            pass
        dst_par.alignment = src_par.alignment
        if text:
            _copy_run_format(src_par, dst_par.add_run(text))

    for it in items:
        if it["kind"] == "p":
            _write_par(out.add_paragraph(), it["par"], out_units[it["unit"]])
            continue

        grid, rows = it["grid"], it["rows"]
        nrows, ncols = len(grid), max((len(r) for r in grid), default=0)
        if nrows == 0 or ncols == 0:
            continue
        new_tbl = out.add_table(rows=nrows, cols=ncols)
        new_tbl.style = "Table Grid"

        groups, first_pos = {}, {}
        for r in range(nrows):
            for c in range(len(grid[r])):
                key = id(grid[r][c]._tc)
                groups.setdefault(key, []).append((r, c))
                first_pos.setdefault(key, (r, c))

        for key, positions in groups.items():
            r0, c0 = first_pos[key]
            dst = new_tbl.cell(r0, c0)
            if it["df"] is not None:
                if r0 < it["hrc"]:
                    value = rows[r0][c0]  # 表頭列維持原樣
                    is_header = True
                else:
                    rd = it["result_df"]
                    ri = r0 - it["hrc"]
                    value = _cell_str(rd.iat[ri, c0]) if ri < rd.shape[0] and c0 < rd.shape[1] else rows[r0][c0]
                    is_header = False
                dst.text = value
                if is_header:
                    for par in dst.paragraphs:
                        for run in par.runs:
                            run.bold = True
            else:
                src_cell, idxs = it["cell_units"][(r0, c0)]
                first = True
                for src_par, ui in zip(src_cell.paragraphs, idxs):
                    dst_par = dst.paragraphs[0] if first else dst.add_paragraph()
                    first = False
                    _write_par(dst_par, src_par, out_units[ui])
            # 還原合併儲存格（只還原矩形範圍；其他位置維持空白，合併後不會多出空段落）
            if len(positions) > 1:
                rs = [p[0] for p in positions]
                cs = [p[1] for p in positions]
                try:
                    new_tbl.cell(min(rs), min(cs)).merge(new_tbl.cell(max(rs), max(cs)))
                except Exception:  # noqa: BLE001 - 不規則合併無法還原就維持不合併
                    pass
        out.add_paragraph()  # 表格後留一行，避免跟下一個表格黏在一起

    buf = io.BytesIO()
    out.save(buf)
    return buf.getvalue(), report, ("mixed" if structured_total else "unstructured")


_PDF_FONT_CANDIDATES = [
    os.environ.get("PDF_CJK_FONT_PATH", ""),
    r"C:\Windows\Fonts\msjh.ttc",       # 微軟正黑體
    r"C:\Windows\Fonts\msjh.ttf",
    r"C:\Windows\Fonts\mingliu.ttc",    # 新細明體
    r"C:\Windows\Fonts\kaiu.ttf",       # 標楷體
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]


def _register_pdf_cjk_font():
    """找一個能顯示繁體中文的字型並註冊進 reportlab，回傳字型名稱。找不到就用內建的 CID 繁中字型。"""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont

    for path in _PDF_FONT_CANDIDATES:
        if path and os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("CJKBody", path, subfontIndex=0))
                return "CJKBody"  # TrueType 字型會把用到的字嵌入 PDF，任何閱讀器都能正確顯示
            except Exception:  # noqa: BLE001 - 字型格式 reportlab 不支援（例如 CFF），換下一個
                continue
    pdfmetrics.registerFont(UnicodeCIDFont("MSung-Light"))  # 不嵌入字型，靠閱讀器內建的亞洲字型顯示
    return "MSung-Light"


def render_text_to_pdf(text: str) -> bytes:
    """把（已去識別化的）純文字排成 PDF。沒有安裝 reportlab 會丟出 ImportError。"""
    from xml.sax.saxutils import escape

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    font = _register_pdf_cjk_font()
    style = ParagraphStyle("body", fontName=font, fontSize=11, leading=19, wordWrap="CJK", spaceAfter=4)
    story = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.strip():
            story.append(Paragraph(escape(line), style))
        else:
            story.append(Spacer(1, 8))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title="", author="",
    )
    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# PDF「版面還原」輸出：讓輸出檔跟輸入檔長得一樣（頁數、紙張大小、每個字的位置、字級、顏色、
# 粗體、表格線與色塊、段落間距），只有被遮罩的字不同。
#
# 做法：用 pdfplumber 讀出原檔每一個字的座標，偵測個資後，用 ReportLab「照原座標重新畫一份」。
# 這是重新產生的新檔，原檔的文字內容流不會被帶過來，所以被遮掉的字不會藏在底下（不像在原檔上蓋白框）。
# 還原不了的：圖片、向量圖形（曲線）、超連結、書籤、表單欄位——這些不會出現在輸出檔裡。
# 字型：原檔的內嵌字型是子集，沒辦法重用，所以換成本機同類字型（見 _PDF_FONT_CANDIDATES），
# 每個字仍放在原位置，所以版面不會跑掉，但字形可能略有不同。
# ---------------------------------------------------------------------------
_PDF_CJK_CHARS = re.compile(rf"[{_CJK}]")


def _extract_pdf_layout(raw: bytes):
    """回傳 (頁面資訊 list, 字元 entries list)。entries 依閱讀順序，含「虛擬分隔」（real=False，只用來偵測文字）。"""
    import pdfplumber

    pages, entries = [], []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for pno, page in enumerate(pdf.pages):
            lines, cur = [], None
            for ch in page.chars:
                base = ch["matrix"][5] if ch.get("matrix") else ch["bottom"]
                if cur is None or abs(base - cur["base"]) > max(1.0, 0.4 * ch["size"]) or ch["x0"] < cur["last_x1"] - 2 * ch["size"]:
                    cur = {"chars": [], "base": base, "last_x1": ch["x0"]}
                    lines.append(cur)
                cur["chars"].append(ch)
                cur["last_x1"] = ch["x1"]
            page_right = max((max(c["x1"] for c in l["chars"]) for l in lines), default=0)
            pages.append(
                {
                    "w": float(page.width),
                    "h": float(page.height),
                    "rects": list(page.rects),
                    "lines": list(page.lines),
                    "images": len(page.images),
                }
            )
            for li, line in enumerate(lines):
                chars = line["chars"]
                prev = None
                for ch in chars:
                    if prev is not None:
                        gap = ch["x0"] - prev["x1"]
                        if (
                            gap > 0.25 * ch["size"]
                            and ch["text"].strip()
                            and prev["text"].strip()
                            and not _PDF_CJK_CHARS.match(ch["text"])
                            and not _PDF_CJK_CHARS.match(prev["text"])
                        ):
                            entries.append({"c": " ", "real": False})
                    entries.append(
                        {
                            "c": ch["text"], "real": True, "page": pno, "x0": ch["x0"], "x1": ch["x1"],
                            "base": line["base"], "size": ch["size"], "font": ch.get("fontname", ""),
                            "color": ch.get("non_stroking_color"), "line": (pno, li), "gid": None,
                        }
                    )
                    prev = ch
                last = chars[-1]
                right = max(c["x1"] for c in chars)
                nxt = lines[li + 1]["chars"][0] if li + 1 < len(lines) else None
                end_par = (
                    nxt is None
                    or right < page_right - 1.5 * last["size"]
                    or abs(nxt["size"] - last["size"]) > 0.5
                )
                if end_par:
                    entries.append({"c": "\n", "real": False})
                elif nxt is not None:
                    a, b = last["text"], nxt["text"]
                    if not _PDF_CJK_CHARS.match(a) and not _PDF_CJK_CHARS.match(b) and a.isalnum() and b.isalnum():
                        entries.append({"c": " ", "real": False})
    return pages, entries


def _pdf_color(c):
    if not c:
        return (0, 0, 0)
    try:
        if len(c) == 1:
            return (c[0], c[0], c[0])
        if len(c) == 3:
            return tuple(c)
        if len(c) == 4:  # CMYK → RGB
            k = c[3]
            return tuple((1 - c[i]) * (1 - k) for i in range(3))
    except Exception:  # noqa: BLE001
        pass
    return (0, 0, 0)


def build_deidentified_pdf_layout(raw: bytes, overrides: dict, use_llm: bool, job_id=None, sel=None):
    """回傳 (輸出位元組, 報告, 'pdf', 警告)。沒有文字可抽（掃描檔）會丟 ValueError。"""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfgen import canvas as rl_canvas

    _set_job(job_id, progress=15, stage="讀取 PDF 版面（每個字的位置、字級、顏色）")
    pages, entries = _extract_pdf_layout(raw)
    t0 = "".join(e["c"] for e in entries)
    if not t0.strip():
        raise ValueError("這份 PDF 沒有可擷取的文字（可能是掃描影像檔），目前無法偵測個資，需要先做 OCR 辨識文字")

    # 偵測用的文字 = 整理過空白的版本；normalize 只會「刪除」空白，所以可以用雙指標對回原位置
    t1 = normalize_text_spacing(t0)
    map1, i0 = [], 0
    for ch in t1:
        while i0 < len(t0) and t0[i0] != ch:
            i0 += 1
        map1.append(i0)
        i0 += 1

    matches = _detect_with_progress(job_id, t1, use_llm, 35, sel)
    _, report = _replace_matches(t1, matches, overrides, sel)
    if (sel or {}).get("analyze"):
        return b"", report, "pdf", None  # 只做分析：不需要輸出檔

    # 規劃每一筆要換掉的範圍：前後相同的字保留原樣，只換中間不同的部分
    _set_job(job_id, progress=96, stage="還原版面並產生 PDF")
    groups = []
    for pii_type, s_, e_, original in sorted(matches, key=lambda m: m[1]):
        rep, _ = _replace_matches(original, [(pii_type, 0, len(original), original)], overrides, sel)
        if rep == original:
            continue
        n = min(len(original), len(rep))
        p = 0
        while p < n and original[p] == rep[p]:
            p += 1
        q = 0
        while q < n - p and original[-1 - q] == rep[-1 - q]:
            q += 1
        a, b = s_ + p, e_ - q
        mid = rep[p:len(rep) - q]
        if b <= a:
            a, b, mid = s_, e_, rep
        gid = len(groups)
        lo, hi = map1[a], map1[b - 1]
        idxs = [i for i in range(lo, hi + 1) if entries[i]["real"]]
        for i in idxs:
            entries[i]["gid"] = gid
        groups.append({"idxs": idxs, "mid": mid})

    font = _register_pdf_cjk_font()
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=(pages[0]["w"], pages[0]["h"]))
    c.setTitle("")
    c.setAuthor("")
    c.setSubject("")

    def _draw(text, x0, x1, base, size, color, bold):
        if not text:
            return
        target = max(x1 - x0, 1)
        natural = pdfmetrics.stringWidth(text, font, size)
        if natural > target * 1.02:  # 太寬：縮小字級（最多縮到 55%）塞進原本的位置，不要壓到後面的字
            size = max(size * 0.55, size * target / natural)
            natural = pdfmetrics.stringWidth(text, font, size)
        # 文字繪製模式（Tr）、描邊色、線寬在 PDF 裡會跨文字物件保留，
        # 所以每段文字都要用 saveState/restoreState 包起來並明確指定，否則粗體標題的設定會汙染後面的內文
        c.saveState()
        t = c.beginText()
        t.setFont(font, size)
        t.setTextOrigin(x0, base)
        t.setFillColorRGB(*color)
        if len(text) > 1:
            t.setCharSpace(max(-size * 0.3, min((target - natural) / (len(text) - 1), size * 0.8)))
        if bold:
            t.setTextRenderMode(2)
            c.setStrokeColorRGB(*color)
            c.setLineWidth(max(size * 0.025, 0.15))
        else:
            t.setTextRenderMode(0)
        t.textOut(text)
        c.drawText(t)
        c.restoreState()

    by_page = {}
    for i, e in enumerate(entries):
        if e["real"]:
            by_page.setdefault(e["page"], []).append(i)

    for pno, pg in enumerate(pages):
        c.setPageSize((pg["w"], pg["h"]))
        for r in pg["rects"]:  # 色塊與框線
            try:
                if r.get("fill") and r.get("non_stroking_color") not in (None, ()):
                    c.setFillColorRGB(*_pdf_color(r["non_stroking_color"]))
                    c.rect(r["x0"], pg["h"] - r["bottom"], r["x1"] - r["x0"], r["bottom"] - r["top"], stroke=0, fill=1)
                if r.get("stroke"):
                    c.setStrokeColorRGB(*_pdf_color(r.get("stroking_color")))
                    c.setLineWidth(r.get("linewidth", 0.5) or 0.5)
                    c.rect(r["x0"], pg["h"] - r["bottom"], r["x1"] - r["x0"], r["bottom"] - r["top"], stroke=1, fill=0)
            except Exception:  # noqa: BLE001
                pass
        for ln in pg["lines"]:
            try:
                c.setStrokeColorRGB(*_pdf_color(ln.get("stroking_color")))
                c.setLineWidth(ln.get("linewidth", 0.5) or 0.5)
                c.line(ln["x0"], pg["h"] - ln["top"], ln["x1"], pg["h"] - ln["bottom"])
            except Exception:  # noqa: BLE001
                pass

        # 沒被換掉的字：同一行、同字型、字級顏色相同、位置連續的合成一段畫
        run = []

        def flush():
            if run:
                a, z = run[0], run[-1]
                _draw("".join(r["c"] for r in run), a["x0"], z["x1"], a["base"], a["size"], _pdf_color(a["color"]), "Bold" in a["font"])
                run.clear()

        for i in by_page.get(pno, []):
            e = entries[i]
            if e["gid"] is not None:
                flush()
                continue
            if run and (
                e["line"] != run[-1]["line"] or e["font"] != run[-1]["font"] or abs(e["size"] - run[-1]["size"]) > 0.1
                or e["color"] != run[-1]["color"] or e["x0"] - run[-1]["x1"] > 2 * e["size"]
            ):
                flush()
            run.append(e)
        flush()

        # 被換掉的字：把替換後的字依原本佔的位置（跨行就分行）畫上去
        for g in groups:
            ents = [entries[i] for i in g["idxs"] if entries[i]["page"] == pno]
            if not ents or not g["mid"]:
                continue
            all_n = len(g["idxs"])
            before = sum(1 for i in g["idxs"] if entries[i]["page"] < pno)
            mid, m = g["mid"], len(g["mid"])
            seg_lines = []
            for e in ents:
                if seg_lines and seg_lines[-1][0]["line"] == e["line"]:
                    seg_lines[-1].append(e)
                else:
                    seg_lines.append([e])
            done = before
            for seg in seg_lines:
                s_i = round(m * done / all_n)
                done += len(seg)
                e_i = round(m * done / all_n)
                _draw(mid[s_i:e_i], seg[0]["x0"], seg[-1]["x1"], seg[0]["base"], seg[0]["size"], _pdf_color(seg[0]["color"]), "Bold" in seg[0]["font"])
        c.showPage()
    c.save()

    warn = None
    imgs = sum(pg["images"] for pg in pages)
    if imgs:
        warn = f"原檔有 {imgs} 張圖片，輸出檔不會保留圖片（圖片裡可能有個資，這樣反而不會外流）。"
    return buf.getvalue(), report, "pdf", warn


def build_deidentified_pdf(raw: bytes, overrides: dict, use_llm: bool, job_id=None, sel=None):
    """
    讀入 .pdf，回傳 (輸出位元組, 報告, 副檔名, 警告訊息)。
    PDF 沒有可編輯的段落結構，所以是把去識別化後的文字重新排版成新的 PDF
    （原本的版面、圖片不會保留，圖片裡可能有個資，這樣反而不會漏）。
    """
    try:
        return build_deidentified_pdf_layout(raw, overrides, use_llm, job_id, sel)
    except ImportError:
        layout_note = "伺服器尚未安裝 pdfplumber，輸出檔的版面會與原檔不同。請執行 pip install pdfplumber 後重新啟動，即可還原原本的版面。"
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001 - 版面還原失敗時退回「純文字重排」，不要讓整個流程失敗
        layout_note = f"版面還原失敗（{e}），已改用純文字重新排版。"
    _set_job(job_id, progress=15, stage="擷取 PDF 文字內容")
    text = extract_text_from_pdf(raw)
    if not text.strip():
        raise ValueError(
            "這份 PDF 沒有可擷取的文字（可能是掃描影像檔），目前無法偵測個資，需要先做 OCR 辨識文字"
        )
    result_text, report = _run_text_pipeline(job_id, text, overrides, use_llm, sel=sel)
    _set_job(job_id, progress=97, stage="產生 PDF 輸出檔")
    try:
        return render_text_to_pdf(result_text), report, "pdf", layout_note
    except ImportError:
        return (
            result_text.encode("utf-8"),
            report,
            "txt",
            "伺服器尚未安裝 reportlab，無法輸出 PDF，已改輸出 .txt。請執行 pip install reportlab 後重新啟動。",
        )


def _build_analysis(report: list) -> dict:
    """把一次「試跑」的報告整理成確認畫面要用的項目：文字中的個資（依類型＋原文彙總）與可勾選的表格欄位。"""
    text_items: dict = {}
    columns = []
    for r in report:
        if "id" in r:
            it = text_items.setdefault(
                r["id"],
                {
                    "id": r["id"],
                    "type": r["type"],
                    "preview": r["original_preview"],
                    "strategy": r["strategy"],
                    "context": r.get("ctx", ""),
                    "suggested": bool(r.get("suggested")),
                    "likely": bool(r.get("likely")),
                    "count": 0,
                },
            )
            it["count"] += 1
        elif "column" in r:
            source = r.get("source", "資料表")
            columns.append(
                {
                    "key": f"{source}::{r['column']}",
                    "source": source,
                    "column": r["column"],
                    "detected_type": r["detected_type"],
                    "strategy": r["strategy"],
                    "affected_rows": r["affected_rows"],
                    "scanned": bool(r.get("scanned")),
                    "recognized": r.get("processed") is not False,
                    "samples": r.get("samples", []),
                    "note": r.get("note", ""),
                }
            )
    return {"items": list(text_items.values()), "columns": columns}


def _run_deidentify_job(job_id: str, raw: bytes, filename: str, overrides: dict, use_llm: bool, sel: Optional[dict] = None):
    """在背景執行緒裡實際跑整個去識別化流程，過程中持續更新 JOBS[job_id] 的進度。"""
    try:
        ext = get_ext(filename)
        k_results: list = []
        warning = None
        _set_job(job_id, progress=5, stage="已接收檔案，判斷格式")

        # ---------- 結構化資料 ----------
        if ext in STRUCTURED_EXTS:
            _set_job(job_id, progress=15, stage="讀取資料表")
            df = load_dataframe(raw, ext)
            _set_job(job_id, progress=30, stage="逐欄位偵測並套用去識別化策略")
            result_df, report = deidentify_dataframe(df, overrides, sel, "資料表")
            result_df, report = _apply_k_anonymity(df, result_df, report, sel, "資料表", k_results)
            _set_job(job_id, progress=90, stage="產生輸出檔案")
            output_bytes = dataframe_to_bytes(result_df, ext)
            data_type = "structured"

        # ---------- 非結構化資料（含「其實是表格」的偵測） ----------
        elif ext in PLAIN_TEXT_EXTS:
            text = _decode_bytes_best_effort(raw)
            delimiter = sniff_table_delimiter(text)

            if delimiter:
                _set_job(job_id, progress=15, stage="偵測到內容其實是表格，改走結構化流程")
                df = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str)
                _set_job(job_id, progress=30, stage="逐欄位偵測並套用去識別化策略")
                result_df, report = deidentify_dataframe(df, overrides, sel, "資料表")
                result_df, report = _apply_k_anonymity(df, result_df, report, sel, "資料表", k_results)
                _set_job(job_id, progress=90, stage="產生輸出檔案")
                output_bytes = dataframe_to_bytes(result_df, "csv")
                data_type = "structured"
                ext = "csv"
            else:
                result_text, report = _run_text_pipeline(job_id, normalize_text_spacing(text), overrides, use_llm, sel=sel)
                output_bytes = result_text.encode("utf-8")
                data_type = "unstructured"

        elif ext in DOCX_EXTS:
            output_bytes, report, data_type = build_deidentified_docx(raw, overrides, use_llm, k_results, job_id, sel)

        elif ext in PDF_EXTS:
            output_bytes, report, ext, warning = build_deidentified_pdf(raw, overrides, use_llm, job_id, sel)
            data_type = "unstructured"

        else:
            raise ValueError(f"不支援的檔案格式: .{ext}")

        if (sel or {}).get("analyze"):
            _set_job(
                job_id,
                status="done",
                progress=100,
                stage="分析完成",
                result={
                    "analysis": _build_analysis(report),
                    "filename": filename,
                    "data_type": data_type,
                },
            )
            return

        _set_job(
            job_id,
            status="done",
            progress=100,
            stage="完成",
            result={
                "filename": f"deidentified_{filename.rsplit('.', 1)[0]}.{ext}",
                "data_type": data_type,
                "file_base64": base64.b64encode(output_bytes).decode("utf-8"),
                "report": report,
                "k_anonymity": k_results,
                **({"warning": warning} if warning else {}),
            },
        )

    except Exception as e:  # noqa: BLE001
        _set_job(job_id, status="error", error=f"處理過程發生錯誤: {e}")
    finally:
        _cleanup_job_later(job_id)


@app.post("/api/deidentify/start")
async def deidentify_start(
    file: UploadFile = File(...),
    strategy_overrides: Optional[str] = Form(None),
    use_llm: bool = Form(False),
    analyze_only: bool = Form(False),
    selection: Optional[str] = Form(None),
):
    """接收檔案後立即回傳 job_id，實際處理丟到背景執行緒跑，不會卡住這個請求。"""
    raw = await file.read()
    filename = file.filename

    overrides = {}
    if strategy_overrides:
        try:
            overrides = json.loads(strategy_overrides)
        except json.JSONDecodeError:
            raise HTTPException(400, "strategy_overrides 必須是合法 JSON 字串")

    sel: dict = {}
    if selection:
        try:
            sel = json.loads(selection)
        except json.JSONDecodeError:
            raise HTTPException(400, "selection 必須是合法 JSON 字串")
    if analyze_only:
        sel = {"analyze": True}

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "processing",
            "progress": 0,
            "stage": "已加入處理佇列",
            "result": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_run_deidentify_job,
        args=(job_id, raw, filename, overrides, use_llm, sel),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"job_id": job_id})


@app.get("/api/deidentify/status/{job_id}")
async def deidentify_status(job_id: str):
    """前端定期呼叫這支，取得目前處理進度；status 變成 done 時，result 欄位會附上完整結果。"""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "找不到這個工作編號（可能已經處理完很久、被清除，或伺服器重啟過）")
    return JSONResponse(job)


@app.post("/api/deidentify")
async def deidentify_endpoint(
    file: UploadFile = File(...),
    strategy_overrides: Optional[str] = Form(None),
    use_llm: bool = Form(False),
):
    ext = get_ext(file.filename)
    raw = await file.read()
    warning = None

    overrides = {}
    if strategy_overrides:
        try:
            overrides = json.loads(strategy_overrides)
        except json.JSONDecodeError:
            raise HTTPException(400, "strategy_overrides 必須是合法 JSON 字串")

    try:
        # ---------- 結構化資料 ----------
        if ext in STRUCTURED_EXTS:
            df = load_dataframe(raw, ext)
            result_df, report = deidentify_dataframe(df, overrides)
            output_bytes = dataframe_to_bytes(result_df, ext)
            data_type = "structured"

        # ---------- 非結構化資料（含「其實是表格」的偵測） ----------
        elif ext in PLAIN_TEXT_EXTS:
            text = _decode_bytes_best_effort(raw)
            delimiter = sniff_table_delimiter(text)

            if delimiter:
                # 內容其實是表格（例如從 Excel 複製貼上存成 .txt），改走結構化流程
                df = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str)
                result_df, report = deidentify_dataframe(df, overrides)
                output_bytes = dataframe_to_bytes(result_df, "csv")
                data_type = "structured"
                ext = "csv"
            else:
                result_text, report = deidentify_text(normalize_text_spacing(text), overrides, use_llm=use_llm)
                output_bytes = result_text.encode("utf-8")
                data_type = "unstructured"

        elif ext in DOCX_EXTS:
            output_bytes, report, data_type = build_deidentified_docx(raw, overrides, use_llm)

        elif ext in PDF_EXTS:
            output_bytes, report, ext, warning = build_deidentified_pdf(raw, overrides, use_llm)
            data_type = "unstructured"

        else:
            raise HTTPException(400, f"不支援的檔案格式: .{ext}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"處理過程發生錯誤: {e}")

    return JSONResponse(
        {
            "filename": f"deidentified_{file.filename.rsplit('.', 1)[0]}.{ext}",
            "data_type": data_type,
            "file_base64": base64.b64encode(output_bytes).decode("utf-8"),
            "report": report,
            **({"warning": warning} if warning else {}),
        }
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}
