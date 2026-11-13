# -*- coding: utf-8 -*-
"""
llm_detector.py
用本機 Ollama 模型當作規則型偵測（正則 + CKIP NER）之外的輔助偵測層，
專門抓那些沒有固定格式、規則抓不到的 PII，例如：
  - 職稱＋姓名的組合（「王經理」「陳醫師」）
  - 角色描述後面接的名字（「負責人：張三」）
  - 公司/機構名稱

因為呼叫的是本機 Ollama（預設 http://localhost:11434），文字不會離開這台機器，
不會有資料外流給第三方雲端服務的疑慮。

事前準備：
    1. 安裝 Ollama：https://ollama.com
    2. 下載一個模型：ollama pull gemma3:4b（可換成你機器跑得動的其他模型）
    3. 確保 Ollama 服務有在跑（安裝完通常會自動啟動；也可手動執行 `ollama serve`）
    4. pip install requests（通常 fastapi 專案已經有這個依賴）

使用方式：
    這一層預設關閉（因為本機模型比正則慢很多），需要時才由呼叫端傳入 use_llm=True 開啟。
    詳見 detection.py 的 detect_text_pii(text, use_llm=True)。
"""

import json
from typing import List, Tuple

import requests

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:4b"  # 請依你本機實際 `ollama pull` 的模型名稱調整

# 逾時採「指數退避」：第一次嘗試用 OLLAMA_TIMEOUT_BASE 秒，
# 若逾時就重試，且每次重試的逾時秒數乘以 2（a = a * 2），
# 最多重試 OLLAMA_MAX_RETRIES 次，逾時秒數不超過 OLLAMA_TIMEOUT_CAP。
OLLAMA_TIMEOUT_BASE = 120  # 第一次嘗試的逾時秒數
OLLAMA_MAX_RETRIES = 3  # 重試次數上限（不含第一次嘗試，總嘗試次數 = 1 + 這個值）
OLLAMA_TIMEOUT_CAP = 180  # 逾時秒數上限，避免無限翻倍等太久

_ollama_available = None  # None=尚未檢查過；True/False=檢查結果快取，避免每次都重新連線測試


def _check_ollama_available() -> bool:
    """檢查本機 Ollama 服務是否有在跑，結果會快取，不用每次呼叫都重新檢查一次。"""
    global _ollama_available
    if _ollama_available is None:
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            _ollama_available = resp.status_code == 200
        except Exception:
            _ollama_available = False
        if not _ollama_available:
            print(
                "[llm_detector] 偵測不到本機 Ollama 服務（http://localhost:11434），"
                "將略過 LLM 輔助偵測。請確認已安裝並啟動 Ollama。"
            )
    return _ollama_available


PII_PROMPT_TEMPLATE = """你是個資偵測助手，請找出下面文字中「規則比對容易漏掉」的個人識別資訊，
特別是：
- 職稱＋姓名的組合（例如「王經理」「陳醫師」）
- 角色描述後面接的名字（例如「負責人：張三」「聯絡窗口為李四」）
- 公司或機構名稱
- 職業描述（即使沒有搭配姓名出現，例如「他是一名軟體工程師」「目前從事護理工作」中的職業名稱本身）

不需要重複找手機號碼、Email、身分證號、地址、銀行帳號這類已經有固定格式規則在處理的類型。

請只回傳一個 JSON 陣列，每個元素格式為：
{{"type": "類型代碼（英文大寫，例如 TITLE_NAME、ORG、OCCUPATION）", "text": "原文中出現的片段，必須逐字照抄，不可改寫"}}

如果沒有找到任何東西，回傳空陣列 []。不要輸出任何其他文字、說明或 markdown 標記。

文字內容：
{text}
"""


def detect_pii_with_ollama(text: str) -> List[Tuple[str, int, int, str]]:
    """
    呼叫本機 Ollama 做輔助 PII 偵測，回傳格式跟 detection.py 的 detect_text_pii 一致：
    [(類型, 起始位置, 結束位置, 原文片段), ...]

    任何失敗情況（Ollama 沒開、逾時、回傳不是合法 JSON、模型「幻覺」出原文沒有的片段）
    都會被安靜地忽略掉，不會讓整個去識別化流程掛掉——頂多就是這次沒有 LLM 輔助偵測到的結果。
    """
    if not text or not text.strip():
        return []
    if not _check_ollama_available():
        return []

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": PII_PROMPT_TEMPLATE.format(text=text),
        "stream": False,
        "format": "json",  # 讓 Ollama 強制輸出合法 JSON，減少解析失敗機率
        "options": {"temperature": 0, "num_predict": 400}, # 偵測任務不需要創意發揮，降低亂猜/幻覺機率
        "keep_alive": "30m",  
    }

    timeout = OLLAMA_TIMEOUT_BASE
    raw_output = None
    last_error = None

    for attempt in range(1, OLLAMA_MAX_RETRIES + 2):  # 1 次初始嘗試 + OLLAMA_MAX_RETRIES 次重試
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            raw_output = resp.json().get("response", "").strip()
            last_error = None
            break
        except requests.exceptions.Timeout as e:
            last_error = e
            print(
                f"[llm_detector] Ollama 第 {attempt} 次嘗試逾時（timeout={timeout}s），"
                + ("即將以雙倍逾時時間重試..." if attempt <= OLLAMA_MAX_RETRIES else "已達重試上限，放棄本次 LLM 輔助偵測。")
            )
            timeout = min(timeout * 2, OLLAMA_TIMEOUT_CAP)
        except Exception as e:  # noqa: BLE001
            # 非逾時類型的錯誤（連線失敗、HTTP 錯誤等）重試通常沒有幫助，直接放棄
            last_error = e
            print(f"[llm_detector] Ollama 呼叫或回應解析失敗，本次略過 LLM 輔助偵測：{e}")
            break

    if raw_output is None:
        if last_error is not None:
            print(f"[llm_detector] 多次嘗試後仍失敗，略過 LLM 輔助偵測：{last_error}")
        return []

    try:
        items = json.loads(raw_output)
    except Exception as e:  # noqa: BLE001
        print(f"[llm_detector] Ollama 回應解析失敗，本次略過 LLM 輔助偵測：{e}")
        return []

    if not isinstance(items, list):
        return []

    results: List[Tuple[str, int, int, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        pii_type = str(item.get("type", "LLM_OTHER")).upper().strip() or "LLM_OTHER"
        snippet = str(item.get("text", "")).strip()
        if not snippet:
            continue
        idx = text.find(snippet)
        if idx == -1:
            # 模型「幻覺」出原文根本沒有的片段，直接跳過，避免誤植/誤刪不存在的內容
            continue
        results.append((pii_type, idx, idx + len(snippet), snippet))
    return results


# ---------------------------------------------------------------------------
# 「敏感內容探索」：不限定類型，請模型自己判斷文字裡有哪些內容可能需要遮罩
# （宗教、政治傾向、性取向、婚姻狀況、疾病與用藥、所得與信用、生物特徵、家族關係、各種編號…）。
# 結果只會當成「建議項目」交給使用者勾選，不會自動遮罩。
# ---------------------------------------------------------------------------
SENSITIVE_PROMPT_TEMPLATE = """你是個資與敏感資料審查助手。請找出下面文字中「可能需要遮罩或去識別化」的內容，
類別不限，包括但不限於：性別（文字中描述某人是男或女，請回報那個「男」或「女」字）、疾病與診斷、用藥與過敏、病史與家族病史、心理與精神狀況、基因與生物特徵、
宗教信仰、政治傾向、性取向、婚姻與家庭狀況、薪資所得與財務、信用評分、犯罪或訴訟紀錄、
各種證號與編號、帳號與密碼、可定位個人的細節。

規則：
- 每一筆回報要盡量短（只含敏感本身的字詞，例如「藏傳佛教格魯派」、「158 萬元」），不要整句。
- text 必須逐字照抄原文，不可改寫、不可翻譯。
- category 用 2～8 個中文字說明這是什麼（例如「宗教信仰」「所得」「過敏藥物」）。
- 不需要重複回報手機號碼、Email、身分證號、一般地址這類格式固定的資料。

只回傳一個 JSON 陣列，每個元素格式：{{"category": "宗教信仰", "text": "藏傳佛教格魯派"}}
沒有找到就回傳 []。不要輸出其他文字或 markdown。

文字內容：
{text}
"""

SENSITIVE_CHUNK_CHARS = 1200  # 本機小模型一次看太長會漏，所以依段落切成這個長度左右分段送


def _split_chunks(text: str, size: int = SENSITIVE_CHUNK_CHARS):
    chunks, buf, start, pos = [], "", 0, 0
    for para in text.split("\n"):
        if buf and len(buf) + len(para) > size:
            chunks.append((start, buf))
            buf, start = "", pos
        if not buf:
            start = pos
        buf += para + "\n"
        pos += len(para) + 1
    if buf.strip():
        chunks.append((start, buf))
    # 單一段落太長（例如 PDF 整頁一段）再硬切
    out = []
    for st, c in chunks:
        if len(c) <= size * 2:
            out.append((st, c))
        else:
            for i in range(0, len(c), size):
                out.append((st + i, c[i:i + size]))
    return out


def detect_sensitive_with_ollama(text: str) -> List[Tuple[str, int, int, str]]:
    """回傳 [(模型自己判斷的類別, 起點, 終點, 原文), ...]；失敗一律回傳 []，不影響主流程。"""
    if not text or not text.strip() or not _check_ollama_available():
        return []
    results, seen = [], set()
    for offset, chunk in _split_chunks(text):
        if not chunk.strip():
            continue
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": SENSITIVE_PROMPT_TEMPLATE.format(text=chunk),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        timeout, raw_output = OLLAMA_TIMEOUT_BASE, None
        for attempt in range(1, OLLAMA_MAX_RETRIES + 2):
            try:
                resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=timeout)
                resp.raise_for_status()
                raw_output = resp.json().get("response", "").strip()
                break
            except requests.exceptions.Timeout:
                timeout = min(timeout * 2, OLLAMA_TIMEOUT_CAP)
            except Exception as e:  # noqa: BLE001
                print(f"[llm_detector] 敏感內容探索失敗，略過這一段：{e}")
                break
        if not raw_output:
            continue
        try:
            items = json.loads(raw_output)
            if isinstance(items, dict):  # 有些模型會包成 {"results": [...]}
                items = next((v for v in items.values() if isinstance(v, list)), [])
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category", "") or item.get("type", "")).strip()[:10] or "LLM 建議"
            snippet = str(item.get("text", "")).strip()
            if not snippet:
                continue
            # 同一片段在文件裡每個出現的位置都列出來（模型通常只回報一次）
            start = 0
            while True:
                idx = text.find(snippet, start)
                if idx == -1:
                    break
                key = (idx, idx + len(snippet))
                if key not in seen:
                    seen.add(key)
                    results.append((category, idx, idx + len(snippet), snippet))
                start = idx + len(snippet)
    return results
