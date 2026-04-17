from __future__ import annotations

import re


def detect_language(question: str) -> str:
    """
    Returns "tr" if the question likely contains Turkish characters, else "en".
    """
    if re.search(r"[çğıöşüİı]", question):
        return "tr"
    # Keyword fallback for ASCII-only Turkish questions.
    words = set(re.findall(r"\b\w+\b", question.lower()))
    tr_keywords = {
        "nerede",
        "dogdu",
        "doğdu",
        "dogmustur",
        "doğmuştur",
        "yonetmeni",
        "yönetmeni",
        "filminin",
        "oynadigi",
        "oynadığı",
        "kac",
        "kaç",
        "hangi",
        "kim",
        "olan",
        "nedir",
        "ne",
        "nasil",
        "nasıl",
    }
    if words & tr_keywords:
        return "tr"
    return "en"


def language_instruction(lang: str) -> str:
    # Backwards-compatible default: used for "final answers".
    return language_instruction_final(lang)


def language_instruction_final(lang: str) -> str:
    if lang == "tr":
        return (
            "Cevabını yalnızca Türkçe ver. Başka dil kullanma. "
            "ÖNCE tek satırda olabildiğince kısa nihai cevabı (ideali: tek isim öbeği) yaz. "
            "Gerekirse ikinci cümlede kısa bir bağlam ver. "
            "Madde işareti, liste veya JSON yazma. "
            "Bilgi notlarda varsa 'Bilinmiyor' deme; notlarda geçen varlık adını (özgün haliyle, "
            "İngilizce veya Türkçe olsa da) koru."
        )
    return (
        "Answer only in English. Do not use any other language. "
        "FIRST output the shortest possible final answer on its own line — ideally a single "
        "noun phrase. Optionally add one short sentence of context. "
        "No bullet points, no JSON. "
        "If the note contains the answer, do NOT say \"Unknown\" — return the entity name verbatim."
    )


def language_instruction_working(lang: str) -> str:
    """
    For intermediate steps (KG summary, notes, query expansion).
    Allows multi-sentence text so the pipeline can actually reason/summarize.
    """
    if lang == "tr":
        return (
            "Yanıtını yalnızca Türkçe ver. "
            "Bu bir ara-adım çıktısı olabilir; gerektiğinde birkaç cümle yazabilirsin. "
            "JSON yazma."
        )
    return (
        "Answer only in English. "
        "This may be an intermediate output; you may write a few sentences if needed. "
        "Do not output JSON."
    )


def wrap_prompt(prompt: str, *, lang: str, style: str = "final") -> str:
    """
    style:
      - "final": strict one-line final answer (default; keeps existing behavior)
      - "work":  allow multi-sentence intermediate outputs
    """
    instr = language_instruction_final(lang) if style == "final" else language_instruction_working(lang)
    return f"{instr}\n\n{prompt}"

