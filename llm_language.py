from __future__ import annotations

import re


def detect_language(question: str) -> str:
    """
    Returns "tr" if the question likely contains Turkish characters, else "en".
    """
    if re.search(r"[çğıöşüİı]", question):
        return "tr"
    return "en"


def language_instruction(lang: str) -> str:
    if lang == "tr":
        return (
            "Cevabını yalnızca Türkçe ver. Başka dil kullanma. "
            "SADECE nihai cevabı yaz: tek satır, mümkünse tek kelime/ifade. "
            "Açıklama, gerekçe, madde işareti, JSON veya ek metin yazma. "
            "Eğer bilgi yetersizse tek satırda \"Bilinmiyor\" yaz."
        )
    # default English
    return (
        "Answer only in English. Do not use any other language. "
        "Output ONLY the final answer: one line, ideally a single word/phrase. "
        "No explanation, no bullets, no JSON, no extra text. "
        "If information is insufficient, output exactly: \"Unknown\"."
    )


def wrap_prompt(prompt: str, *, lang: str) -> str:
    return f"{language_instruction(lang)}\n\n{prompt}"

