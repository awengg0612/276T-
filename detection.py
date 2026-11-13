# -*- coding: utf-8 -*-
"""
pii_detector.py
負責辨識文字/欄位中的個人識別資訊 (PII)。
可依專案需求擴充 PATTERNS 與 COLUMN_KEYWORDS。

姓名偵測（非結構化文字）：
    使用 CKIP Transformers 做中文 NER，辨識出 PERSON 實體。
    安裝：pip install ckip-transformers
    注意：
      - 底層是 BERT 模型，第一次執行會自動從 HuggingFace 下載權重（需要網路，約數百 MB），
        之後會快取在本機，之後執行不用再下載。
      - 模型採「延遲載入」：伺服器啟動時不會載入，直到第一次真的呼叫姓名偵測才載入，
        避免拖慢 uvicorn 啟動速度。
      - 若套件未安裝或模型載入失敗，會印出警告並跳過姓名偵測（不影響其他 PII 類型偵測）。
"""

import re
from typing import List, Tuple, Optional, Dict

# ---------------------------------------------------------------------------
# 地址規則（獨立組裝，方便維護）
# ---------------------------------------------------------------------------
# 結構：縣市（只認真正的縣市名稱）＋「區/鄉/鎮/市」及（或）路街 ＋ 段 巷 弄 號 樓
# 與舊版差異：
#   1. 縣市改成明確名單。舊版是「任意兩個中文字＋市」，會把「遠離城市」誤判成「離城市」、
#      「聯絡人何美玲小姐市話」誤判成「聯絡市」。
#   2. 縣市後面至少要有「區鄉鎮市」或「路街」才算地址，單獨一個「台北市」不算（資訊量低、容易是雜訊）。
#   3. 補上巷、弄、樓：舊版「三段86巷12號4樓」只抓到「三段」，後面的門牌沒被遮罩。
#   4. 路名不會跨過「的、在、於、是、位、並」這類連接字，避免「台北市政府位於信義路」整句被吞掉。
_CITY = (
    r"(?:(?:台北|臺北|新北|桃園|台中|臺中|台南|臺南|高雄|基隆|新竹|嘉義)市"
    r"|(?:台北|臺北|桃園|台中|臺中|台南|臺南|高雄|新竹|苗栗|彰化|南投|雲林|嘉義"
    r"|屏東|宜蘭|花蓮|台東|臺東|澎湖|金門|連江)縣)"
)
_DISTRICT = r"[一-鿿]{1,3}?(?:區|鄉|鎮|市)"
_ROAD = r"(?:(?![的在於是位並])[一-鿿0-9０-９]){1,8}(?:路|街|大道)"
_NUM = r"[0-9０-９]+"
_CNUM = r"[0-9０-９一二三四五六七八九十]+"
_ADDRESS_TAIL = (
    rf"(?:{_CNUM}段)?"
    rf"(?:{_CNUM}巷)?"
    rf"(?:{_CNUM}弄)?"
    rf"(?:{_NUM}(?:之{_NUM})?號)?"
    rf"(?:{_CNUM}(?:樓|[Ff])(?:之{_NUM})?)?"
)
_ADDRESS_REGEX = rf"{_CITY}(?:{_DISTRICT}(?:{_ROAD})?|{_ROAD}){_ADDRESS_TAIL}"


_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"  # IPv4 的一段（0~255）

# ---------------------------------------------------------------------------
# 1. 正則表達式規則（用於非結構化文字 & 結構化欄位內容抽樣比對）
# ---------------------------------------------------------------------------
PATTERNS: Dict[str, re.Pattern] = {
    # 注意：這裡刻意不用 \b 當邊界——Python 的正則表達式預設把中文字也算進
    # 「單字字元」，所以像「帳號699519144071」「身分證號A123456789」這種
    # 中文字緊貼數字/英文、中間沒有空格的寫法，用 \b 會完全抓不到。
    # 改用「前後不是數字/英文字母」的 lookaround 判斷，中文字前後緊貼也能正確比對。
    "TW_ID": re.compile(r"(?<![A-Za-z0-9])[A-Z][12]\d{8}(?![A-Za-z0-9])"),   # 台灣身分證字號
    "TW_ARC": re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}\d{8}(?![A-Za-z0-9])"),   # 居留證號（簡化版）
    # 手機：支援國內 09 開頭（0912-345-678），也支援國際格式 +886 開頭（+886-912-345-678，
    # 國際格式會把國內的第一個 0 省略，所以是 +886 接 9 開頭）
    "MOBILE": re.compile(
        r"(?<!\d)(?:09\d{2}|\+886[-\s]?9\d{2})(?:[-\s]?\d{3}){2}(?!\d)"
    ),
    # 市話：0 + 1~2碼區碼（區碼可以用括號包住：(02) 2368-9988、（02）2368-9988）+ 本地號碼（本地號碼內部可以再用空格/破折號分段，
    # 例如「02 1234 5678」），也支援國際格式 +886 開頭（+886-2-1234-5678）
    # + 選用的分機（轉123 / 分機123 / ext.123 / #123）
    "PHONE": re.compile(
        r"(?<!\d)(?:[(（]?0\d{1,2}[)）]?|[(（]?\+886[)）]?[-\s]?[(（]?\d{1,2}[)）]?)(?:[-\s]?\d){5,9}"
        r"(?:\s*(?:轉|分機|[Ee][Xx][Tt]\.?|#)\s*\d{1,5})?"
        r"(?!\d)"
    ),
    "EMAIL": re.compile(r"(?<![A-Za-z0-9_.+-])[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+(?![A-Za-z0-9_.+-])"),
    "CREDIT_CARD": re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?<=\d)(?!\d)"),
    "DATE": re.compile(r"(?<!\d)(19|20)\d{2}[-/年]\d{1,2}[-/月]\d{1,2}日?(?!\d)"),
    "BANK_ACCOUNT": re.compile(r"(?<!\d)\d{10,16}(?!\d)"),
    # 地址：縣市（必要）+ 鄉鎮市區（可省略）+ 路街巷弄（可省略）+ 段（可省略）+ 門牌號（可省略）
    # 用完整結構比對，取代舊版「隨便一個『市/區/號』字就往前後抓一段」的作法，
    # 避免「帳號」「身分證字號」這類詞裡的「號」字被誤判成地址。
    # IPv4：四段各 0~255（192.168.1.1），前後不能再接數字或小數點，避免抓到版本號之類的長串
    "IP": re.compile(rf"(?<![\d.]){_OCTET}(?:\.{_OCTET}){{3}}(?!\d)(?!\.\d)"),
    # MAC：AA:BB:CC:DD:EE:FF／AA-BB-CC-DD-EE-FF，以及 Cisco 寫法 aabb.ccdd.eeff
    "MAC": re.compile(
        r"(?<![0-9A-Fa-f:\-])(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f])(?!:[0-9A-Fa-f])"
        r"|(?<![0-9A-Fa-f.])(?:[0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4}(?![0-9A-Fa-f.])"
    ),
    "ADDRESS": re.compile(_ADDRESS_REGEX),
}

# ---------------------------------------------------------------------------
# 1a. 「標籤＋值」型態的規則（只用在自由文字，不能拿去驗證結構化欄位的儲存格）
# ---------------------------------------------------------------------------
# 這些規則靠「前面有沒有職業/職稱這種標籤字樣」來判斷後面那段是不是要遮的值，
# 用的是「向前查看」（lookbehind），單獨拿一個儲存格內容去比對時，
# 前面沒有任何文字可看，lookbehind 一定會判斷失敗——所以這組規則刻意不放進上面的
# PATTERNS，避免被 app.py 的結構化格式驗證誤用，導致每一筆職業資料都被判成「無效資料」。
TEXT_ONLY_PATTERNS: Dict[str, re.Pattern] = {
    # 職業／職稱：後面接冒號（全形或半形），值就是同一行剩下的內容
    # （lookbehind 只放固定 3 碼寬的「職業:」/「職稱:」，選用的空白留在主要比對式裡
    #   一起被抓進去；反正職業預設策略是整段換成 [REDACTED]，多一個空白字元不影響結果）
    "OCCUPATION": re.compile(r"(?<=職業[:：]|職稱[:：]) ?[^\n\r]+"),
    # 護照號碼：9 碼以內的英數字，單獨出現沒有辨識度，一定要有「護照」「Passport」標籤才抓。
    # 只遮罩標籤後面的值（具名群組 v），標籤本身保留。
    "PASSPORT": re.compile(
        r"(?:護照(?:號碼|號|編號)?|[Pp]assport(?:\s*(?:No\.?|Number|#))?)\s*[:：]?\s*(?P<v>[A-Za-z0-9]{6,9})(?![A-Za-z0-9])"
    ),
    # 健保卡號：12 碼數字，跟銀行帳號／信用卡長得像，同樣一定要有「健保卡」等標籤才抓。
    "NHI_CARD": re.compile(
        r"(?:健保卡(?:號碼|號)?|健保(?:卡)?(?:號碼|號)|NHI(?:\s*(?:card|No\.?|Number))?)\s*[:：]?\s*(?P<v>\d{4}[-\s]?\d{4}[-\s]?\d{4})(?!\d)",
        re.I,
    ),
}

# 只用來驗證結構化儲存格格式的規則（不能拿去在自由文字裡搜尋，否則誤判太多）
VALIDATION_PATTERNS: Dict[str, re.Pattern] = {
    "PASSPORT": re.compile(r"[A-Za-z0-9]{6,9}"),
    "NHI_CARD": re.compile(r"\d{4}[-\s]?\d{4}[-\s]?\d{4}"),
}

# ---------------------------------------------------------------------------
# 1a. 姓名偵測（CKIP NER，延遲載入）
# ---------------------------------------------------------------------------
_ner_chunker = None
_ner_load_failed = False


def _get_ner_chunker():
    """延遲載入 CKIP NER 模型。第一次呼叫才真的載入，之後重複使用同一個實例。"""
    global _ner_chunker, _ner_load_failed
    if _ner_chunker is None and not _ner_load_failed:
        try:
            from ckip_transformers.nlp import CkipNerChunker
            # model="albert-base" 速度較快；準確率要求更高可改成 "bert-base"（較慢、較準）
            _ner_chunker = CkipNerChunker(model="albert-base")
        except Exception as e:  # noqa: BLE001
            print(f"[detection.py] CKIP NER 模型載入失敗，將略過姓名偵測：{e}")
            _ner_load_failed = True
    return _ner_chunker


def detect_names_ner(text: str) -> List[Tuple[str, int, int, str]]:
    """使用 CKIP NER 辨識文字中的人名（PERSON 實體），回傳格式與 detect_text_pii 一致。"""
    if not text or not text.strip():
        return []

    chunker = _get_ner_chunker()
    if chunker is None:
        return []

    results: List[Tuple[str, int, int, str]] = []
    try:
        ner_result = chunker([text])  # 輸入為 list[str]，回傳 list[list[NerToken]]
        for token in ner_result[0]:
            if token.ner == "PERSON":
                results.append(("NAME", token.idx[0], token.idx[1], token.word))
    except Exception as e:  # noqa: BLE001
        print(f"[detection.py] CKIP NER 執行時發生錯誤，本次略過姓名偵測：{e}")
    return results

# ---------------------------------------------------------------------------
# 2. 結構化資料：欄位名稱關鍵字對照表（比對欄位名稱以決定整欄的處理方式）
# ---------------------------------------------------------------------------
COLUMN_KEYWORDS: Dict[str, List[str]] = {
    "NAME": ["姓名", "name", "客戶姓名", "聯絡人", "負責人"],
    "TW_ID": ["身分證字號", "身分證", "id_number", "national_id", "統一編號"],
    "NHI_CARD": ["健保卡", "健保號", "健保", "nhi", "health_insurance"],
    "PASSPORT": ["護照", "passport"],
    # 注意：關鍵字是「包含」比對，單獨的 "ip"、"mac" 會誤中 description／zip／machine，
    # 所以這裡只放完整詞組；欄名剛好就是 IP 或 MAC 的情況在 match_column_by_name 另外處理
    "IP": ["ip位址", "ip位置", "ip地址", "ip_address", "ip address", "ipaddress", "ipv4", "來源ip", "目的ip", "客戶端ip", "使用者ip"],
    "MAC": ["mac位址", "mac位置", "mac地址", "mac_address", "mac address", "macaddress", "網卡位址", "實體位址"],
    "MOBILE": ["手機", "手機號碼", "mobile", "cellphone"],
    "PHONE": ["電話", "市話", "phone", "tel"],
    "EMAIL": ["email", "e-mail", "信箱", "電子郵件"],
    "ADDRESS": ["地址", "住址", "address"],
    "BIRTHDAY": ["生日", "出生日期", "birthday", "birth_date", "dob"],
    "CREDIT_CARD": ["信用卡", "卡號", "credit_card", "card_number"],
    "BANK_ACCOUNT": ["帳號", "銀行帳號", "account_number", "bank_account"],
    "OCCUPATION": ["職業", "職稱", "occupation", "job_title", "job title", "position"],
}


def match_column_by_name(column_name: str) -> Optional[str]:
    """
    依欄位名稱（忽略大小寫、忽略欄名裡的空白）比對出可能的 PII 類型。

    有些 Word 表格的表頭習慣在字中間加空白排版（例如「姓 名」「服 務 單 位」），
    這種空白如果不先去掉，"姓名" 這個關鍵字就永遠對不上「姓 名」，導致整欄完全沒被偵測到。
    """
    name = re.sub(r"[\s　]+", "", column_name.strip().lower())

    # pandas 對沒有文字的欄位會自動編號成「Unnamed: 4」這種欄名，英文關鍵字 "name"
    # 剛好是 "unnamed" 的子字串，字串比對會誤判成姓名欄，導致整欄被假名化覆蓋掉
    # ——這種自動產生的佔位欄名本身不代表任何意義，直接跳過、不參與關鍵字比對。
    if re.fullmatch(r"unnamed[:_]?\s*\d*", name):
        return None

    if name in ("ip", "mac"):  # 欄名剛好就是 IP／MAC（不能用「包含」比對，會誤中 zip、machine）
        return name.upper()

    for pii_type, keywords in COLUMN_KEYWORDS.items():
        for kw in keywords:
            kw_normalized = re.sub(r"[\s　]+", "", kw.lower())
            if kw_normalized in name:
                return pii_type
    return None


def guess_column_by_sample(sample_values: List[str], sample_size: int = 20) -> Optional[str]:
    """
    當欄位名稱無法判斷時，抽樣欄位內容以正則表達式比對，
    若命中率超過閾值則視為該類型。

    注意：這裡刻意用 fullmatch（整個儲存格內容都要符合格式）而不是 search
    （內容裡有一部分像就算數）。例如「臺北市交通管制工程處」這種服勤單位名稱，
    開頭剛好長得像地址的「臺北市」，如果只用 search 會被誤判成地址整欄去遮罩/防呆，
    但其實整格內容跟地址格式對不起來，改用 fullmatch 可以避免這種誤判。
    """
    # 先統一轉成字串再過濾空值/NaN，不要在轉換之前就用 `in (None, "", "nan")` 判斷——
    # pandas 欄位裡缺值的儲存格可能是 Python 的 float('nan')，不會等於字串 "nan"，
    # 用轉換後的字串判斷才不會漏掉這些缺值，也避免 "nan" 這種佔位字串混進樣本裡
    # 影響命中率計算。
    values = []
    for v in sample_values[:sample_size]:
        sv = str(v).strip() if v is not None else ""
        if sv and sv.lower() != "nan":
            values.append(sv)
    if not values:
        return None

    best_type, best_ratio = None, 0.0
    for pii_type, pattern in PATTERNS.items():
        hits = sum(1 for v in values if pattern.fullmatch(v))
        ratio = hits / len(values)
        if ratio > best_ratio:
            best_type, best_ratio = pii_type, ratio

    return best_type if best_ratio >= 0.6 else None


def detect_text_pii(
    text: str,
    use_llm: bool = False,
    use_ner: bool = True,
    only_types: Optional[set] = None,
) -> List[Tuple[str, int, int, str]]:
    """
    偵測一段非結構化文字中的 PII，回傳 (類型, 起始位置, 結束位置, 原始字串)。

    use_ner=False 時不跑 CKIP 姓名辨識（模型比正則慢很多，逐格掃描大量儲存格時用）。
    only_types 有給的話，只保留這些類型的結果（在移除重疊區間「之前」就先過濾，
    所以不會出現「被一個不要的類型蓋掉、想要的類型反而沒抓到」的情況）。

    use_llm=True 時，會額外呼叫本機 Ollama（見 llm_detector.py）做輔助偵測，
    抓正則/NER 漏掉的職稱+姓名組合、公司名稱等。預設關閉，因為本機模型比規則慢很多，
    且需要事先安裝、啟動 Ollama——沒開的話這段就直接跳過，不影響其他規則正常運作。
    """
    results: List[Tuple[str, int, int, str]] = []

    # 「職業：」「護照號碼：」「健保卡號：」這種有明確標籤的欄位，不用等 LLM，直接用規則抓，
    # 快又穩定。先於通用規則加入：同一段文字同時符合標籤規則和銀行帳號（12 碼數字）等
    # 通用規則時，排序後保留先加入的，標籤規則比較準，要讓它優先。
    for pii_type, pattern in TEXT_ONLY_PATTERNS.items():
        for m in pattern.finditer(text):
            if "v" in pattern.groupindex and m.group("v") is not None:
                results.append((pii_type, m.start("v"), m.end("v"), m.group("v")))
            else:
                results.append((pii_type, m.start(), m.end(), m.group()))

    for pii_type, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            results.append((pii_type, m.start(), m.end(), m.group()))

    # 姓名：交給 CKIP NER 辨識（規則型正則無法可靠判斷自由文字中的人名）
    if use_ner:
        results.extend(detect_names_ner(text))

    # 選用：本機 Ollama 輔助偵測（職稱+姓名、公司名稱等規則型方法抓不到的情況）
    if use_llm:
        try:
            from llm_detector import detect_pii_with_ollama
            results.extend(detect_pii_with_ollama(text))
        except Exception as e:  # noqa: BLE001
            print(f"[detection.py] LLM 輔助偵測發生錯誤，本次略過：{e}")

    if only_types is not None:
        results = [r for r in results if r[0] in only_types]

    # 依起始位置排序，並移除重疊區間（保留較早偵測到的規則）
    results.sort(key=lambda x: x[1])
    filtered: List[Tuple[str, int, int, str]] = []
    last_end = -1
    for item in results:
        if item[1] >= last_end:
            filtered.append(item)
            last_end = item[2]
    return filtered
