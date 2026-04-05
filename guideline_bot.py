"""
消化器分野 ガイドライン更新通知 Bot
====================================
- PubMed から消化器関連のガイドライン・推奨・コンセンサスを検索
- 主要学会 (AGA, ECCO, ACG, ESGE, JSGE 等) の新規発表を検出
- Gemini API で内容を日本語要約
- Discord Webhook で通知
"""

import os
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import requests
import google.generativeai as genai

# ============================================================
# 設定
# ============================================================
GUIDELINE_WEBHOOK_URL = os.environ["GUIDELINE_WEBHOOK_URL"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"

NOTIFIED_FILE = Path(__file__).parent / "notified_guideline_pmids.json"

# 直近何日分を検索するか（週1回実行なら7、毎日なら1）
SEARCH_DAYS = 7

MAX_RESULTS = 30

# ============================================================
# 検索クエリ
# Publication Type + MeSH + 学会名 で幅広く捕捉
# ============================================================
SEARCH_QUERIES = [
    # ---- Publication Type ベース ----
    # ガイドライン (消化器全般)
    (
        '("Practice Guideline"[Publication Type] OR "Guideline"[Publication Type]) '
        'AND ("Gastroenterology"[MeSH] OR "Gastrointestinal Diseases"[MeSH] '
        'OR "Liver Diseases"[MeSH] OR "Pancreatic Diseases"[MeSH] '
        'OR "Biliary Tract Diseases"[MeSH])'
    ),
    # コンセンサス・推奨 (タイトルベース)
    (
        '("consensus"[Title] OR "guideline"[Title] OR "recommendation"[Title] '
        'OR "position statement"[Title] OR "clinical practice"[Title] OR "guidance"[Title]) '
        'AND ("gastro*"[Title/Abstract] OR "hepat*"[Title/Abstract] '
        'OR "inflammatory bowel"[Title/Abstract] OR "endoscop*"[Title/Abstract] '
        'OR "pancrea*"[Title/Abstract] OR "colorectal"[Title/Abstract] '
        'OR "liver"[Title/Abstract] OR "biliary"[Title/Abstract] '
        'OR "esophag*"[Title/Abstract] OR "celiac"[Title/Abstract])'
    ),
    # ---- 主要学会名ベース ----
    # AGA (American Gastroenterological Association)
    (
        '("American Gastroenterological Association"[Affiliation] '
        'OR "AGA Clinical Practice"[Title]) '
        'AND ("guideline"[Title] OR "update"[Title] OR "recommendation"[Title])'
    ),
    # ACG (American College of Gastroenterology)
    (
        '("American College of Gastroenterology"[Affiliation] OR "ACG"[Title]) '
        'AND ("guideline"[Title] OR "recommendation"[Title] OR "clinical"[Title])'
    ),
    # ECCO (European Crohn's and Colitis Organisation)
    (
        '("ECCO"[Title] OR "European Crohn"[Affiliation]) '
        'AND ("guideline"[Title] OR "consensus"[Title] OR "recommendation"[Title])'
    ),
    # ESGE (European Society of Gastrointestinal Endoscopy)
    (
        '("ESGE"[Title] OR "European Society of Gastrointestinal Endoscopy"[Affiliation]) '
        'AND ("guideline"[Title] OR "recommendation"[Title] OR "position statement"[Title])'
    ),
    # JSGE (日本消化器病学会) / JGES (日本消化器内視鏡学会)
    (
        '("Japanese Society of Gastroenterology"[Affiliation] '
        'OR "Japan Gastroenterological Endoscopy"[Affiliation] '
        'OR "JSGE"[Title] OR "JGES"[Title]) '
        'AND ("guideline"[Title] OR "recommendation"[Title] OR "consensus"[Title])'
    ),
    # AASLD (American Association for the Study of Liver Diseases)
    (
        '("AASLD"[Title] OR "American Association for the Study of Liver"[Affiliation]) '
        'AND ("guideline"[Title] OR "guidance"[Title] OR "recommendation"[Title])'
    ),
    # EASL (European Association for the Study of the Liver)
    (
        '("EASL"[Title] OR "European Association for the Study of the Liver"[Affiliation]) '
        'AND ("guideline"[Title] OR "recommendation"[Title] OR "position statement"[Title])'
    ),
    # BSG (British Society of Gastroenterology)
    (
        '("British Society of Gastroenterology"[Affiliation] OR "BSG"[Title]) '
        'AND ("guideline"[Title] OR "recommendation"[Title] OR "consensus"[Title])'
    ),
]

# ============================================================
# PubMed E-utilities
# ============================================================
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pubmed(query: str, reldate: int) -> list[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": MAX_RESULTS,
        "datetype": "edat",
        "reldate": reldate,
        "retmode": "json",
        "sort": "date",
    }
    resp = requests.get(ESEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_articles(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []

    # PubMed API は一度に最大200件
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    resp = requests.get(EFETCH_URL, params=params, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    articles = []

    for article_elem in root.findall(".//PubmedArticle"):
        pmid = _text(article_elem, ".//PMID")
        title = _full_text(article_elem, ".//ArticleTitle")

        # Abstract
        abstract_parts = []
        for at in article_elem.findall(".//AbstractText"):
            label = at.get("Label", "")
            text = "".join(at.itertext()).strip()
            if label:
                abstract_parts.append(f"[{label}] {text}")
            else:
                abstract_parts.append(text)
        abstract = "\n".join(abstract_parts)

        if not abstract:
            abstract_node = article_elem.find(".//Abstract")
            if abstract_node is not None:
                abstract = "".join(abstract_node.itertext()).strip()

        # ガイドラインはabstractがない場合もあるが、タイトルだけでも通知する
        journal = _full_text(article_elem, ".//Journal/Title")

        authors = []
        for author in article_elem.findall(".//Author")[:3]:
            last = _text(author, "LastName")
            fore = _text(author, "ForeName")
            if last:
                authors.append(f"{last} {fore}".strip())
        if len(article_elem.findall(".//Author")) > 3:
            authors.append("et al.")

        doi = ""
        for aid in article_elem.findall(".//ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text or ""

        # Publication Type を取得
        pub_types = []
        for pt in article_elem.findall(".//PublicationType"):
            if pt.text:
                pub_types.append(pt.text.strip())

        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "authors": ", ".join(authors),
            "doi": doi,
            "pub_types": pub_types,
        })

    return articles


def _text(elem, path: str) -> str:
    node = elem.find(path)
    if node is not None and node.text:
        return node.text.strip()
    return ""


def _full_text(elem, path: str) -> str:
    node = elem.find(path)
    if node is not None:
        return "".join(node.itertext()).strip()
    return ""


# ============================================================
# Gemini API で要約生成
# ============================================================
def summarize_guideline(article: dict) -> dict:
    model = genai.GenerativeModel(GEMINI_MODEL)

    abstract_section = article["abstract"] if article["abstract"] else "（Abstractなし。タイトルから推測してください。）"

    prompt = f"""あなたは消化器内科の専門医向けにガイドラインの更新情報を伝える医学ライターです。
以下の論文はガイドライン・推奨・コンセンサスステートメントです。
内容を日本語で分かりやすく要約してください。

## 出力フォーマット（厳守）

TITLE_JA: （日本語タイトル。例: 「AGA ガイドライン：潰瘍性大腸炎の管理 2026年改訂」）

SOCIETY: （発行学会名。例: AGA, ECCO, ACG, ESGE, JSGE, AASLD, EASL, BSG, その他）

SUMMARY: （3〜5文。主な推奨事項の変更点、新しい推奨のポイント、前版からの重要な変更点を含める。）

KEY_CHANGES: （箇条書きで2〜4点。最も重要な変更・追加された推奨を簡潔に。各項目は1文で。）

## 論文情報
タイトル: {article['title']}
ジャーナル: {article['journal']}
著者: {article['authors']}
Publication Type: {', '.join(article['pub_types'])}

Abstract:
{abstract_section}
"""

    response = model.generate_content(prompt)
    text = response.text

    # パース
    title_ja = ""
    society = "その他"
    summary = ""
    key_changes = ""

    lines = text.split("\n")
    current_section = None
    section_lines = {"SUMMARY": [], "KEY_CHANGES": []}

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("TITLE_JA:"):
            title_ja = stripped.replace("TITLE_JA:", "").strip()
            current_section = None
        elif stripped.startswith("SOCIETY:"):
            society = stripped.replace("SOCIETY:", "").strip()
            current_section = None
        elif stripped.startswith("SUMMARY:"):
            content = stripped.replace("SUMMARY:", "").strip()
            if content:
                section_lines["SUMMARY"].append(content)
            current_section = "SUMMARY"
        elif stripped.startswith("KEY_CHANGES:"):
            content = stripped.replace("KEY_CHANGES:", "").strip()
            if content:
                section_lines["KEY_CHANGES"].append(content)
            current_section = "KEY_CHANGES"
        elif current_section and stripped:
            section_lines[current_section].append(stripped)

    summary = " ".join(section_lines["SUMMARY"]).strip()
    key_changes = "\n".join(section_lines["KEY_CHANGES"]).strip()

    return {
        "title_ja": title_ja or article["title"],
        "society": society,
        "summary": summary or "（要約を生成できませんでした）",
        "key_changes": key_changes,
    }


# ============================================================
# 学会ごとの絵文字・色
# ============================================================
SOCIETY_CONFIG = {
    "AGA":   {"emoji": "🇺🇸", "color": 0x1F4E79},
    "ACG":   {"emoji": "🇺🇸", "color": 0x2E86C1},
    "ECCO":  {"emoji": "🇪🇺", "color": 0x27AE60},
    "ESGE":  {"emoji": "🇪🇺", "color": 0x16A085},
    "EASL":  {"emoji": "🇪🇺", "color": 0x8E44AD},
    "AASLD": {"emoji": "🇺🇸", "color": 0xC0392B},
    "JSGE":  {"emoji": "🇯🇵", "color": 0xE74C3C},
    "JGES":  {"emoji": "🇯🇵", "color": 0xE74C3C},
    "BSG":   {"emoji": "🇬🇧", "color": 0x2C3E50},
}

DEFAULT_CONFIG = {"emoji": "📋", "color": 0xF39C12}


# ============================================================
# Discord 通知
# ============================================================
def send_discord_notification(article: dict, guideline: dict):
    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/"
    doi_url = f"https://doi.org/{article['doi']}" if article["doi"] else ""

    config = DEFAULT_CONFIG
    for key, val in SOCIETY_CONFIG.items():
        if key.lower() in guideline["society"].lower():
            config = val
            break

    links = f"[PubMed]({pubmed_url})"
    if doi_url:
        links += f"  |  [Full Text]({doi_url})"

    fields = [
        {
            "name": "📝 概要",
            "value": guideline["summary"][:1024],
            "inline": False,
        },
    ]

    if guideline["key_changes"]:
        fields.append({
            "name": "🔑 主な変更点",
            "value": guideline["key_changes"][:1024],
            "inline": False,
        })

    fields.extend([
        {
            "name": "📄 原著",
            "value": f"**{article['title'][:200]}**\n"
                     f"_{article['journal']}_  |  {article['authors']}",
            "inline": False,
        },
        {
            "name": "🔗 リンク",
            "value": links,
            "inline": False,
        },
    ])

    embed = {
        "title": f"{config['emoji']} {guideline['title_ja']}"[:256],
        "url": pubmed_url,
        "color": config["color"],
        "fields": fields,
        "footer": {
            "text": f"{guideline['society']}  |  PMID: {article['pmid']}",
        },
        "timestamp": datetime.utcnow().isoformat(),
    }

    payload = {
        "username": "Guideline Bot",
        "embeds": [embed],
    }

    resp = requests.post(GUIDELINE_WEBHOOK_URL, json=payload, timeout=15)
    resp.raise_for_status()
    print(f"[Discord] ガイドライン通知: PMID {article['pmid']}")


# ============================================================
# 重複排除
# ============================================================
def load_notified_pmids() -> set[str]:
    if NOTIFIED_FILE.exists():
        data = json.loads(NOTIFIED_FILE.read_text())
        return set(data.get("pmids", []))
    return set()


def save_notified_pmids(pmids: set[str]):
    recent = sorted(pmids)[-2000:]
    NOTIFIED_FILE.write_text(json.dumps({"pmids": recent}, indent=2))


# ============================================================
# メイン処理
# ============================================================
def main():
    print(f"=== Guideline Bot 実行: {datetime.now().isoformat()} ===")

    notified = load_notified_pmids()

    # 全クエリで検索して候補を集める
    all_pmids = []
    for i, query in enumerate(SEARCH_QUERIES):
        print(f"[Search {i+1}/{len(SEARCH_QUERIES)}] {query[:60]}...")
        pmids = search_pubmed(query, reldate=SEARCH_DAYS)
        all_pmids.extend(pmids)
        time.sleep(0.5)  # rate limit 対策

    # 重複除去 & 未通知
    seen = set()
    new_pmids = []
    for p in all_pmids:
        if p not in notified and p not in seen:
            new_pmids.append(p)
            seen.add(p)

    print(f"[Filter] 新規 {len(new_pmids)} 件")

    if not new_pmids:
        print("新規ガイドラインなし。終了。")
        return

    # Abstract 取得
    articles = fetch_articles(new_pmids)
    print(f"[PubMed] {len(articles)} 件取得")

    if not articles:
        print("論文データなし。終了。")
        return

    # 各ガイドラインを要約して通知
    count = 0
    for article in articles:
        try:
            guideline = summarize_guideline(article)
            send_discord_notification(article, guideline)
            notified.add(article["pmid"])
            count += 1
            time.sleep(2)  # Discord rate limit 対策
        except Exception as e:
            print(f"[Error] PMID {article['pmid']}: {e}")

    save_notified_pmids(notified)
    print(f"=== 完了: {count} 件通知 ===")


if __name__ == "__main__":
    main()
