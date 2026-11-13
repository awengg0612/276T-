# 去識別化平台

在本機偵測並去識別化檔案中的個人資料（PII）。資料不會上傳到外部服務，LLM 使用本機 Ollama。

## 功能

- **結構化資料**（xlsx、csv）：逐欄選擇處理方式（刪除、雜湊、遮蔽），並可做 k 匿名泛化（ANJANA + pyCANON）。
- **非結構化文件**（docx、文字型 pdf）：規則 + 中文 NER（CKIP，選用）+ 本機 LLM 偵測，輸出去識別化後的檔案。
- 每次處理後提供報表，列出各欄位或項目的處理方式。

## 檔案結構

| 檔案 | 用途 |
| --- | --- |
| `app.py` | FastAPI 後端 |
| `去識別平台.html` | 前端操作頁面 |
| `登入.html` | 登入頁 |
| `detection.py` | 規則式偵測 |
| `llm_detector.py` | 呼叫本機 Ollama 的 LLM 偵測 |
| `kanon_adapter.py` | k 匿名介面（ANJANA） |
| `deidentify_api.py` | 遮罩／雜湊等基本去識別函式 |

## 安裝

必須使用 **Python 3.12**。

```bash
py -3.12 -m venv venv312
venv312\Scripts\activate
pip install -r requirements.txt
# 選用：中文 NER
pip install ckip-transformers torch
```

確認 k 匿名套件可用：

```bash
python -c "import anjana.anonymity, pycanon.anonymity; print('OK')"
```

## 啟動

```bash
python -m uvicorn app:app --reload --port 8000
```

再用瀏覽器開啟 `登入.html`（或直接開 `去識別平台.html`）。前端預設連線 `http://localhost:8000`。

### LLM（選用）

安裝 [Ollama](https://ollama.com) 後拉取模型：

```bash
ollama pull qwen2.5:7b
```

| 環境變數 | 說明 | 預設 |
| --- | --- | --- |
| `OLLAMA_MODEL` | Ollama 模型名稱（須與 `ollama list` 相符） | 見 `llm_detector.py` |
| `DEID_CACHE_TTL_SECONDS` | 偵測結果暫存秒數，0 為關閉 | 600 |
| `PDF_CJK_FONT_PATH` | PDF 輸出使用的中文字型路徑 | 自動尋找系統字型 |

## 安全注意事項

- API **沒有身分驗證**，CORS 為 `allow_origins=["*"]`，**只適合本機使用**。要給多人或內網使用前，請先加上驗證並限縮 CORS。
- 目前 `登入.html` 只在前端檢查帳號密碼格式，不是真正的驗證。
- 啟動時不要使用 `--host 0.0.0.0`。
- 本專案不包含任何測試資料；請勿把真實個資檔提交到版本庫。

## 已知限制

- xlsx 只讀取第一個工作表。
- 不支援掃描型 PDF（無文字層）。
- 不支援 l-diversity，只保證 k 匿名。
- 英文縣市縮寫、混用的性別寫法需先在原始資料統一。

## 授權

（請自行選擇，例如 MIT；使用的 ANJANA、pyCANON 等套件各有自己的授權條款。）
