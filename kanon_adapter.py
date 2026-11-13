"""
kanon_adapter.py —— 把 ANJANA（現成的 k 匿名套件）接到去識別平台的「轉接層」

k 匿名的演算法本身完全由 ANJANA 的 anjana.anonymity.k_anonymity 執行（泛化 + 抑制），
k 值的計算由 pyCANON 執行。這個檔案只負責三件事：
  1. 依欄位內容自動建立 ANJANA 需要的「泛化層級」（hierarchies）：
       數字 → 每 N 一組；日期 → 年月 → 年；代碼（郵遞區號）→ 逐步遮後幾碼；其他 → 直接 *
  2. 呼叫 ANJANA，並處理它的回傳（被抑制的列、失敗時回傳空表）
  3. 整理成平台報告要用的結果

安裝：  python -m pip install anjana     （需要 Python 3.10~3.12；它會連 pandas、numpy 一起鎖定版本）
"""
from __future__ import annotations

import contextlib
import io
import re
from typing import Dict, List, Optional, Sequence

import pandas as pd

try:
    from anjana.anonymity import k_anonymity as _anjana_k_anonymity
    import pycanon.anonymity as _pc
    AVAILABLE = True
    IMPORT_ERROR = ""
except Exception as _e:  # noqa: BLE001  套件沒裝或版本不相容時，平台其他功能照常運作
    AVAILABLE = False
    IMPORT_ERROR = str(_e)

_DATE_RE = re.compile(r"^\s*(\d{4})\s*[-/.年]?\s*(\d{1,2})?")
_CODE_NAME_RE = re.compile(r"郵遞|區號|zip|postal|代碼|代號|code", re.I)
_BLANK = {"", "nan", "None", "NaN"}
MAX_SET_LABEL = 3  # 文字欄位的不同值不超過這個數量時，最上層用「值1/值2」標籤；超過就用 *


# ---- 台灣縣市 → 區域（依國土計畫四大區域；離島另列）。想改分法直接改這張表 ----
REGION_OF_COUNTY = {
    "北部": ["台北市", "新北市", "基隆市", "桃園市", "新竹市", "新竹縣", "苗栗縣"],
    "中部": ["台中市", "彰化縣", "南投縣", "雲林縣"],
    "南部": ["嘉義市", "嘉義縣", "台南市", "高雄市", "屏東縣"],
    "東部": ["宜蘭縣", "花蓮縣", "台東縣"],
}
# 只要值裡出現這些字就算離島（寫縣市或鄉鎮都行）；離島優先於上面的縣市（例如「台東縣蘭嶼鄉」→ 離島）
ISLAND_WORDS = ["澎湖", "金門", "連江", "馬祖", "綠島", "蘭嶼", "琉球", "小琉球", "東沙", "南沙", "釣魚台"]


# 英文寫法（小寫比對，包含關係；寫「Taipei City」「taipei, taiwan」都認得）。
# 「tai」「tp」這類縮寫有歧義（台北／台中／台南／台東都可能），無法判斷，所以不收。
EN_REGION = {
    "離島": ["penghu", "kinmen", "lienchiang", "matsu", "green island", "orchid island", "lanyu", "xiaoliuqiu", "liuqiu"],
    "北部": ["new taipei", "taipei", "keelung", "taoyuan", "hsinchu", "miaoli"],
    "中部": ["taichung", "changhua", "nantou", "yunlin"],
    "南部": ["chiayi", "tainan", "kaohsiung", "pingtung"],
    "東部": ["yilan", "ilan", "hualien", "taitung"],
}


def region_of(value: str) -> str:
    """縣市／地址 → 北部、中部、南部、東部、離島；對不到回傳空字串。支援中文與英文寫法。"""
    low = value.strip().lower()
    if low and low.isascii():  # 英文寫法
        for region in ("離島", "北部", "中部", "南部", "東部"):  # 離島優先
            if any(w in low for w in EN_REGION[region]):
                return region
        return ""
    v = value.replace("臺", "台").replace(" ", "")
    if any(w in v for w in ISLAND_WORDS):
        return "離島"
    for region, counties in REGION_OF_COUNTY.items():
        for c in counties:
            if v.startswith(c[:-1]) or c in v:  # 「台北市」「台北」「台北市松山區…」都認得
                return region
    return ""


def _is_county_column(s: pd.Series) -> bool:
    v = _nonblank(s)
    return len(v) > 0 and (v.map(lambda x: region_of(x) != "").mean() >= 0.8)


def _clean(series: pd.Series) -> pd.Series:
    """全部轉成字串；空值統一成空字串（ANJANA 的層級比對是用字串相等）。"""
    return series.map(lambda v: "" if pd.isna(v) or str(v).strip() in _BLANK else str(v).strip())


def _nonblank(s: pd.Series) -> pd.Series:
    return s[s != ""]


def _is_numeric(s: pd.Series) -> bool:
    v = _nonblank(s)
    return len(v) > 0 and pd.to_numeric(v.str.replace(",", ""), errors="coerce").notna().all()


def _is_date(s: pd.Series) -> bool:
    v = _nonblank(s)
    return len(v) > 0 and v.map(lambda x: bool(_DATE_RE.match(x)) and len(x) >= 6 and bool(re.search(r"[-/.年]", x))).all()


def _is_code(s: pd.Series) -> bool:
    v = _nonblank(s)
    return len(v) > 0 and v.str.len().nunique() == 1 and v.str.len().iloc[0] >= 3 and v.str.fullmatch(r"[0-9A-Za-z]+").all()


def _bucket(x: str, width: int, offset: int = 0) -> str:
    if x == "":
        return "*"
    n = float(x.replace(",", ""))
    lo = int((n - offset) // width * width + offset)
    return f"{lo}–{lo + width - 1}"


def _date_part(x: str, parts: int) -> str:
    m = _DATE_RE.match(x)
    if x == "" or not m:
        return "*"
    return m.group(1) if (parts == 1 or not m.group(2)) else f"{m.group(1)}-{int(m.group(2)):02d}"


def _map_fixed(col: pd.Series, hier: dict, level: int) -> pd.Series:
    """把原始值對應到階層裡指定的那一層（依唯一值對照表）。"""
    base = [str(x) for x in hier[0].tolist()]
    to = dict(zip(base, hier[level].tolist()))
    return col.map(lambda v: to.get("" if pd.isna(v) else str(v).strip(), "*"))


def _cut_hierarchy(hier: dict, start: int) -> dict:
    """取階層從第 start 層開始的部分，當作新的階層（新的第 0 層 = 原第 start 層的不重複值）。"""
    g = hier[start].tolist()
    uniq = list(dict.fromkeys(g))
    first = {v: g.index(v) for v in uniq}
    return {j - start: pd.Series([hier[j].iloc[first[v]] for v in uniq], dtype=object) for j in range(start, len(hier))}


def build_hierarchy(series: pd.Series, name: str = "", kind: Optional[str] = None,
                    widths: Optional[Sequence[int]] = None):
    """回傳 (hierarchy dict, 各層說明 list)。
    hierarchy = {0: [唯一原值...], 1: [對應的第 1 層泛化值...], ..., 最後一層: 全部 '*'}（ANJANA 的格式）。
    只放「不重複的值」，比逐列放快很多。"""
    s = _clean(series)
    uniq = list(dict.fromkeys(s.tolist()))
    if kind is None:
        if _is_code(s) and _CODE_NAME_RE.search(name):
            kind = "code"
        elif _is_county_column(s):
            kind = "region"
        elif _is_numeric(s):
            kind = "numeric"
        elif _is_date(s):
            kind = "date"
        elif _is_code(s):
            kind = "code"
        else:
            kind = "category"

    levels = [("原值", list(uniq))]
    if kind == "region":
        levels.append(("區域（北部／中部／南部／東部／離島）", [region_of(u) or "*" for u in uniq]))
    elif kind == "numeric":
        nums = pd.to_numeric(_nonblank(s).str.replace(",", ""), errors="coerce")
        span = float(nums.max() - nums.min()) if len(nums) else 0
        for w in (list(widths) if widths else [w for w in (2, 5, 10, 20, 50, 100, 500, 1000) if w < max(span, 3)]):
            levels.append((f"每 {w} 一組", [_bucket(u, w) for u in uniq]))
    elif kind == "date":
        levels.append(("年月", [_date_part(u, 2) for u in uniq]))
        levels.append(("年", [_date_part(u, 1) for u in uniq]))
    elif kind == "code":
        n = int(_nonblank(s).str.len().iloc[0]) if len(_nonblank(s)) else 0
        all_digits = all(u.isdigit() for u in uniq if u)
        for keep in range(n - 1, 0, -1):
            if all_digits:  # 純數字代碼（郵遞區號）：泛化成「區間」，例如 10050 → 10000–10099
                width = 10 ** (n - keep)
                levels.append((f"區間（每 {width} 一組）",
                               [(f"{u[:keep]}{'0' * (n - keep)}–{u[:keep]}{'9' * (n - keep)}") if u else "*" for u in uniq]))
            else:  # 含英文字母的代碼無法表示成數字區間，只能遮掉後面幾碼
                levels.append((f"只留前 {keep} 碼", [(u[:keep] + "*" * (len(u) - keep)) if u else "*" for u in uniq]))
    # 最上層：文字類別且種類很少（例如性別）時，用「男/女」這種標籤代表「全部」，比單純的 * 更清楚；其他欄位仍是 *
    top, top_name = "*", "全部遮成 *"
    if kind == "category":
        vals = [u for u in uniq if u]
        if 1 < len(vals) <= MAX_SET_LABEL:
            order = {"男": 0, "女": 1}
            vals = sorted(vals, key=lambda v: (order.get(v, 9), v))
            top, top_name = "/".join(vals), f"合併成「{'/'.join(vals)}」"
    levels.append((top_name, [top] * len(uniq)))

    # 去掉和上一層完全一樣的層級（例如全是空值時）
    out, names = [], []
    for nm, vals in levels:
        if out and vals == out[-1]:
            continue
        out.append(vals)
        names.append(nm)
    # 用 object 型別的 Series：新版 pandas 會把字串轉成 StringArray，ANJANA 的型別檢查會拒絕
    return {i: pd.Series(v, dtype=object) for i, v in enumerate(out)}, names


def run_k_anonymity(df: pd.DataFrame, quasi: Sequence[str], k: int = 5, suppress_pct: float = 10,
                    ident: Sequence[str] = (), kinds: Optional[Dict[str, str]] = None,
                    widths: Optional[Dict[str, Sequence[int]]] = None,
                    force_top: Sequence[str] = ()) -> dict:
    """
    對 df 的準識別碼欄位做 k 匿名。不修改傳入的 df。
    回傳 dict：
      ok            是否達成
      df            泛化後的 quasi 欄位（只有 quasi 欄；列數可能因抑制而減少）
      kept_index    保留下來的列在輸入 df 的位置（0 起算）
      k_before / k_after / target_k / suppressed_rows / original_rows
      levels        {欄位: 最後採用的泛化說明}
      error         失敗原因（ok=False 時）
    """
    if not AVAILABLE:
        return {"ok": False, "error": f"無法載入 anjana：{IMPORT_ERROR}"}
    quasi = list(quasi)
    try:  # pandas 3 預設把字串欄位轉成專用字串型別，ANJANA（鎖定 pandas 2.3.x）內部的型別檢查會拒絕；關掉即可
        pd.set_option("future.infer_string", False)
    except Exception:  # noqa: BLE001
        pass
    miss = [c for c in quasi if c not in df.columns]
    if miss or not quasi:
        return {"ok": False, "error": f"找不到準識別碼欄位：{miss}" if miss else "沒有選任何準識別碼欄位"}
    if len(df) < k:
        return {"ok": False, "error": f"資料只有 {len(df)} 筆，少於 k={k}，不可能達成"}

    # 一律用 object 型別：ANJANA 內部有型別檢查（要 numpy.ndarray），新版 pandas 的字串欄位會被拒絕
    sub = pd.DataFrame({c: _clean(df[c]).astype(object).values for c in quasi}, dtype=object)
    hierarchies, names = {}, {}
    for c in quasi:
        hierarchies[c], names[c] = build_hierarchy(sub[c], c, (kinds or {}).get(c), (widths or {}).get(c))

    k_before = int(_pc.k_anonymity(sub, quasi))

    # 「固定泛化」的欄位：先把整欄換成指定那一層的值（縣市→區域、性別→男/女、其他→*），
    # 再把「該層以上」的階層交給 ANJANA。這樣固定後的值仍然算進分組（縣市變成北部／南部，
    # 同區域的人才算同一組），ANJANA 若 k 還不夠，仍可從那一層繼續往上泛化。
    forced = [c for c in quasi if c in set(force_top)]
    start = {c: 0 for c in quasi}
    for c in forced:
        start[c] = next((i for i, n in enumerate(names[c]) if n.startswith("區域")), len(names[c]) - 1)
        sub[c] = _map_fixed(sub[c], hierarchies[c], start[c]).astype(object).values
        hierarchies[c] = _cut_hierarchy(hierarchies[c], start[c])
        names[c] = names[c][start[c]:]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):  # ANJANA 會 print 進度，不要弄髒後端輸出
        anon = _anjana_k_anonymity(sub.copy(), [], quasi, int(k), float(suppress_pct), hierarchies)

    if anon is None or len(anon) == 0:
        return {"ok": False, "k_before": k_before, "target_k": k, "original_rows": len(df),
                "error": "在允許刪除的比例內，無法達到這個 k（可調高允許刪除的比例、調低 k，或減少準識別碼欄位）"}

    # 有抑制列時，ANJANA 會 reset_index() 出一個 'index' 欄，記錄保留列在輸入中的位置
    if "index" in anon.columns and "index" not in quasi:
        kept = [int(i) for i in anon["index"].tolist()]
        anon = anon.drop(columns=["index"])
    else:
        kept = list(range(len(anon)))
    anon = anon.reset_index(drop=True)[quasi]

    levels = {}
    for c in quasi:
        vals = set(anon[c].astype(str))
        lv = 0
        for i in range(len(names[c])):
            if vals.issubset(set(hierarchies[c][i].tolist())):
                lv = i
                break
        levels[c] = names[c][lv] + ("（固定）" if c in forced and lv == 0 else "")

    k_after = int(_pc.k_anonymity(anon, quasi))
    return {
        "ok": k_after >= k, "df": anon, "kept_index": kept,
        "k_before": k_before, "k_after": k_after, "target_k": int(k),
        "suppressed_rows": len(df) - len(anon), "original_rows": len(df),
        "levels": levels, "quasi_columns": quasi,
    }
