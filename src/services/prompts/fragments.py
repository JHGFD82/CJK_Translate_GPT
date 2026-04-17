"""Raw prompt text fragments referenced by the prompt spec classes.

All string content lives here.  No logic, no conditionals — just named
constants.  Variables use str.format() placeholders, documented inline.
"""

# ---------------------------------------------------------------------------
# Shared injection blocks
# ---------------------------------------------------------------------------

# Placeholders: {note}
ADDITIONAL_INSTRUCTIONS = "\n\nADDITIONAL INSTRUCTIONS:\n{note}"
ADDITIONAL_NOTES = "\n\nADDITIONAL NOTES:\n{note}"


# ---------------------------------------------------------------------------
# Translation — system prompt sections
# ---------------------------------------------------------------------------

# Placeholders: {source}, {target}
TRANSLATION_ROLE = (
    "Follow the instructions carefully. Please act as a professional translator from {source} "
    "to {target}. I will provide you with text from a document, and your task is "
    "to translate it from {source} to {target}. Please only output the translation and do not "
    "output any irrelevant content. If there are garbled characters or other non-standard text "
    "content, delete the garbled characters."
)

# Placeholders: {target}
TRANSLATION_CONTEXT_SPEC = (
    'You may be provided with "--Context: " which includes either the document\'s abstract or '
    "text from the previous page for context. You will also be provided with \"--Current Page: \" "
    "which includes the text of the current page. Only output the {target} translation of "
    'the "--Current Page: ". Do not output the context, nor the "--Context: " and "--Current Page: " '
    "labels."
)

# Output-format variants — keyed by canonical format group
TRANSLATION_FORMATTING: dict[str, str] = {
    "file": (
        "Use proper paragraph breaks and standard text formatting suitable for file output. "
        "Use actual line breaks (not \\n characters) to separate paragraphs and sections naturally."
    ),
    "console": (
        'You can format and line break the output yourself using "\\n" for line breaks in console output.'
    ),
}

# Numbered-content block — included in the system prompt only when numbered content is detected
TRANSLATION_NUMBERED_SYSTEM = (
    "IMPORTANT: Pay special attention to numbered lists, citations, and footnotes.\n"
    "Preserve ALL numbering exactly as it appears in the source text. This includes:\n"
    "\u2022 Arabic numerals: 1, 2, 3... or 1), 2), 3)...\n"
    "\u2022 Numbers in brackets: [1], [2], [3]... or (1), (2), (3)...\n"
    "\u2022 Chinese numerals: \u4e00\u3001\u4e8c\u3001\u4e09... or \uff08\u4e00\uff09\u3001\uff08\u4e8c\uff09\u3001\uff08\u4e09\uff09...\n"
    "\u2022 Japanese/Korean numbering: \u2460, \u2461, \u2462... or \uff11\u3001\uff12\u3001\uff13...\n"
    "\u2022 Japanese reference format: 14\u3000author\u300ctitle\u300d\u2192 should become \"14. Author, 'Title'\"\n"
    "\n"
    "CRITICAL DISTINCTION - DO NOT ADD NUMBERING:\n"
    "\n"
    "- If the source text has section headings WITHOUT numbers, do NOT add numbers to them\n"
    "- Only preserve numbering that already exists in the source\n"
    '- Section titles like "\u80cc\u666f" or "\u7d50\u8ad6" should remain as "Background" or "Conclusion" without numbers\n'
    "\n"
    "CRITICAL FOR BIBLIOGRAPHY/REFERENCES: If you encounter numbered reference lists or bibliography\n"
    '(like "1. Author Title, Publisher" format), preserve the exact numbering format. Do NOT convert\n'
    "numbered references into paragraph form. Keep each reference as a separate numbered item.\n"
    "\n"
    'CRITICAL: When you see Japanese reference format like "14\u3000\u677e\u4e0b\u5b89\u96c4\u76e3\u4fee\u6c38\u6843\u5143\u826f\u300c\u798f\u5ca1\u8529\u300d",\n'
    'translate it to proper English reference format like "14. Supervised by Matsushita Yasuo, Higaki Motoyoshi, \'Fukuoka Domain\'".\n'
    'DO NOT output just the number "14" by itself - always include the full reference text with proper formatting.'
)


# ---------------------------------------------------------------------------
# Translation — user prompt sections
# ---------------------------------------------------------------------------

# Placeholders: {source}, {target}
TRANSLATION_USER_BASE = (
    'Translate only the {source} text of the "--Current Page: " to {target}, without outputting any other\n'
    'content, and without outputting anything related to "--Context: ", if provided.'
)

# Included in the user prompt only when numbered content is detected
TRANSLATION_NUMBERED_USER = (
    "CRITICAL: Preserve all numbering systems exactly as they appear in the source "
    "(1, 2, 3... or [1], [2]... or \u2460, \u2461... etc.).\n"
    "DO NOT ADD numbering to headings or sections that are not numbered in the source text.\n"
    "\n"
    'CRITICAL FOR REFERENCES: When translating reference entries like "14\u3000\u677e\u4e0b\u5b89\u96c4\u76e3\u4fee\u6c38\u6843\u5143\u826f\u300c\u798f\u5ca1\u8529\u300d",\n'
    "translate the COMPLETE reference including author names, titles, and formatting. Output should be\n"
    '"14. Author Name, \'Title\'" NOT just the isolated number "14". Always translate the full reference text.\n'
    "\n"
    "NUMBERING CONTINUATION - VERY IMPORTANT:\n"
    '- If the context shows "Previous numbering ended with: X. Reference", you MUST continue numbering from X+1 for any new numbered items on the current page.\n'
    "- Do NOT restart numbering from 1 - always continue the sequence from the previous page.\n"
    '- Example: If context shows "Previous numbering ended with: 25. Some Reference", and current page has more numbered items, they should be numbered 26, 27, 28, etc.\n'
    "- This applies ONLY to numbered reference lists, NOT to section headings.\n"
    "\n"
    "SECTION HEADINGS: If the source has section headings without numbers, translate them as headings without adding numbers."
)

TRANSLATION_FOOTNOTE_RULE = (
    'IMPORTANT: Only add a "Footnotes:" section if there is actual explanatory footnote text at the bottom\n'
    "of the page. Do NOT add \"Footnotes:\" for simple citation numbers like (38), (39) within paragraphs."
)

TRANSLATION_NO_META_COMMENTARY = (
    'Do not provide any prompts to the user, for example: "This is the translation of the current page.":'
)


# ---------------------------------------------------------------------------
# OCR — system prompt sections
# ---------------------------------------------------------------------------

OCR_SYSTEM_BASE = (
    "You are an expert OCR assistant specializing in text extraction from images containing "
    "Chinese, Japanese, Korean, and English.\n"
    "\n"
    "Your task is to transcribe all legibly visible text from the image exactly as it appears, "
    "preserving layout, orientation (horizontal or vertical), and structure as closely as possible."
)

OCR_RULES = (
    "RULES:\n"
    "- Extract ONLY text that is actually visible in the image \u2014 do NOT add, invent, or hallucinate any content\n"
    "- Do NOT repeat text unless it genuinely appears multiple times in the image\n"
    "- Do NOT translate \u2014 output text in its original language and script exactly as shown\n"
    "- Do NOT add commentary, analysis, disclaimers, or assumptions\n"
    "- Preserve original formatting, line breaks, numbering, symbols, and special characters\n"
    "- If text is partially obscured or unclear, extract what you can; note any unreadable sections with a "
    'single brief line at the end (e.g., "[Some text unclear due to image quality]")'
)

# Placeholders: {target}
OCR_USER_BASE = (
    "Transcribe all legibly visible text from this image exactly as it appears in {target}."
)

OCR_USER_RULES = (
    "CRITICAL RULES FOR THIS IMAGE:\n"
    "- Output ONLY text that is genuinely visible \u2014 do NOT invent, fill in, or hallucinate any characters or words\n"
    "- Do NOT translate \u2014 preserve the original script and language exactly as shown, even in mixed-language content\n"
    "- Include ALL text elements: body text, headings, captions, page numbers, table contents, labels, and marginalia\n"
    "- Preserve line breaks, paragraph spacing, and structural layout as faithfully as plain text allows\n"
    "- Reproduce punctuation, symbols, and special characters exactly as they appear\n"
    "- If a section of text is partially obscured or too degraded to read, extract what you can and note the gap "
    'with a single brief marker (e.g., "[text unclear]") \u2014 do not skip the surrounding legible text\n'
    "- Do not add commentary, disclaimers, or explanatory notes outside of the above illegibility marker"
)

OCR_REFINEMENT_BASE = (
    "Review the transcription above carefully against this image."
    "\n\n"
    "Correct any errors you find: wrong or missing characters, extra or hallucinated text, "
    "misread characters, or formatting issues. "
    "If the transcription is already accurate, return it unchanged.\n\n"
    "Return ONLY the corrected transcription \u2014 no commentary, no explanation, no preamble."
)

# Shared vertical-text block used by both OCR and image-translation services.
# OCR uses different wording for the column direction hint; they are kept separate below.
OCR_VERTICAL_BLOCK = (
    "TEXT ORIENTATION:\n"
    "The majority of text in this image is vertical \u2014 written top-to-bottom, "
    "with columns ordered right-to-left. Read and transcribe each column from top to bottom, "
    "proceeding from the rightmost column to the leftmost."
)

OCR_VERTICAL_REINFORCEMENT = (
    "ORIENTATION REMINDER: Text is vertical \u2014 transcribe each column top-to-bottom, "
    "proceeding right-to-left across columns."
)

# ---------------------------------------------------------------------------
# Image translation — system prompt sections
# ---------------------------------------------------------------------------

# Placeholders: {source}, {target}
IMAGE_TRANSLATION_ROLE = (
    "You are an expert reader and translator specialising in {source} text found in images."
)

# Placeholders: {source}, {target}
IMAGE_TRANSLATION_FORMAT_SPEC = (
    "Your task is to perform two operations on the image:\n"
    "1. TRANSCRIBE all visible {source} text exactly as it appears.\n"
    "2. TRANSLATE that transcribed text into fluent, accurate {target}.\n"
    "\n"
    "You MUST return your response in EXACTLY this format, with the section headers on their own lines:\n"
    "\n"
    "[TRANSCRIPT]\n"
    "<transcribed {source} text, preserving original layout and line breaks>\n"
    "\n"
    "[TRANSLATION]\n"
    "<{target} translation of the transcribed text>"
)

# Placeholders: {target}
IMAGE_TRANSLATION_TRANSCRIPTION_RULES = (
    "TRANSCRIPTION RULES:\n"
    "- Reproduce text exactly as it appears in the image \u2014 do not correct, modernise, or alter characters.\n"
    "- Preserve line breaks, punctuation, numbering, and overall structure.\n"
    "- Use surrounding context and translation target to resolve ambiguous or partially obscured characters; "
    "mark genuinely unreadable text with [unclear] inline rather than a trailing summary.\n"
    "- Do not skip any text, including headers, captions, inscriptions, or marginal notes."
)

# Placeholders: {target}
IMAGE_TRANSLATION_TRANSLATION_RULES = (
    "TRANSLATION RULES:\n"
    "- Produce a fluent, scholarly {target} translation.\n"
    "- Preserve the structure of the original (line breaks, stanzas, numbered items, etc.).\n"
    "- For classical or archaic language, prefer a literary translation over a literal one.\n"
    "- Do not add explanatory notes, commentary, or translator remarks."
)

IMAGE_TRANSLATION_VERTICAL_BLOCK = (
    "TEXT ORIENTATION:\n"
    "The majority of text in this image is vertical \u2014 written top-to-bottom, "
    "with columns ordered right-to-left. Read each column from top to bottom, "
    "proceeding from the rightmost column to the leftmost."
)

IMAGE_TRANSLATION_VERTICAL_NOTE = (
    " The text is predominantly vertical (top-to-bottom, right-to-left columns)."
)

# ---------------------------------------------------------------------------
# Language-pair-specific notes — injected automatically by TranslationPromptSpec
# when the source/target combination matches.  Keyed by (source_language, target_language).
# Both directions of a pair should normally have entries.
# ---------------------------------------------------------------------------

# Source-style notes — injected automatically when a source-style flag is set.
# Each entry is a complete block of instructions placed after the language-pair
# note  (if any) and before the user's explicit system_note.

KANBUN_NOTE = (
    "The source text is kanbun (漢文) — Classical Chinese written for Japanese "
    "kundoku (訓読) reading. Apply the following conventions:\n"
    "- Reconstruct word order according to kundoku conventions: Japanese verb-final "
    "syntax, not the Subject-Verb-Object order of Classical Chinese.\n"
    "- Expand implicit grammatical elements (particles, verb endings, auxiliary "
    "words) that are absent in the kanbun but required by kundoku reading.\n"
    "- Preserve kanbun punctuation markers (返り点 kaeriten, 送り仮名 okurigana) "
    "as context clues — do not reproduce them literally in the translation.\n"
    "- Use the register appropriate to classical Japanese scholarly prose when "
    "producing Japanese output, or fluent academic prose for English output.\n"
    "- Proper nouns, reign names, and place names should follow established "
    "Sinological or Japanese historical conventions (e.g. Heian, not 'Hei-an')."
)

KANBUN_SCRIPT_NOTE = (
    "The text is kanbun (漢文) — Classical Chinese written in kanji only. "
    "Hiragana and katakana appear only as annotations (送り仮名 okurigana, "
    "振り仮名 furigana) beside the main characters, not as independent text. "
    "Transcribe all kanji and all kana annotations exactly as they appear — "
    "do NOT omit small or lightly printed kana, as they are critical annotations."
)

KANBUN_OCR_NOTE = (
    "The image contains kanbun (漢文) — Classical Chinese text annotated for "
    "Japanese kundoku (訓読) reading. Transcribe the image faithfully, preserving "
    "all annotations exactly as they appear:\n"
    "- Transcribe all 返り点 (kaeriten: レ点, 一二三点, 上中下点, etc.) as they appear "
    "beside or below the main characters.\n"
    "- Transcribe all 送り仮名 (okurigana) — small kana written beside the main "
    "characters — exactly as they appear.\n"
    "- Preserve 句読点 (punctuation marks) and any 訓点 (kunten) annotations.\n"
    "- Do NOT reorder characters, expand grammar, or interpret kundoku conventions; "
    "transcribe the text exactly as written on the page."
)

LANGUAGE_PAIR_NOTES: dict[tuple[str, str], str] = {
    ("Japanese", "Korean"): (
        "Japanese and Korean both have elaborate honorific systems that do not map "
        "one-to-one. Apply the following conventions:\n"
        "- Preserve the formality register of the source: formal/polite text should "
        "become formal/polite in the target (e.g. 합쇼체 or 해요체 in Korean; "
        "丁寧語 / teineigo in Japanese).\n"
        "- Translate title suffixes appropriately: Japanese 様/さん/先生 → Korean "
        "님/선생님, and vice versa.\n"
        "- Do not flatten honorific speech to plain speech (반말 / タメ口) unless "
        "the source explicitly uses an informal register."
    ),
    ("Korean", "Japanese"): (
        "Korean and Japanese both have elaborate honorific systems that do not map "
        "one-to-one. Apply the following conventions:\n"
        "- Preserve the formality register of the source: formal/polite text should "
        "become formal/polite in the target (丁寧語 / teineigo in Japanese; "
        "합쇼체 or 해요체 in Korean).\n"
        "- Translate title suffixes appropriately: Korean 님/선생님 → Japanese "
        "様/さん/先生, and vice versa.\n"
        "- Do not flatten honorific speech to plain speech (タメ口 / 반말) unless "
        "the source explicitly uses an informal register."
    ),
}


# ---------------------------------------------------------------------------
# Script guidance dictionaries
# (moved from constants.py — text lives here, services import from here)
# ---------------------------------------------------------------------------

# Keyed by target language name; used by OcrPromptSpec
OCR_SCRIPT_GUIDANCE: dict[str, str] = {
    "Chinese": (
        "The text uses Chinese characters (hanzi/\u6f22\u5b57). "
        "Transcribe each character exactly as it appears."
    ),
    "Simplified Chinese": (
        "The text uses Simplified Chinese characters (\u7b80\u4f53\u5b57). "
        "Transcribe each character exactly in its simplified form \u2014 "
        "do NOT convert to or substitute traditional variants."
    ),
    "Traditional Chinese": (
        "The text uses Traditional Chinese characters (\u7e41\u9ad4\u5b57). "
        "Transcribe each character exactly in its traditional form \u2014 "
        "do NOT convert to or substitute simplified variants."
    ),
    "Japanese": (
        "The text uses Japanese script, which combines kanji (Chinese-derived characters), "
        "hiragana, katakana, and possibly r\u014dmaji. "
        "Reproduce all scripts exactly as written. "
        "Some kanji may be Japanese-specific forms (kokuji) not found in standard Chinese \u2014 "
        "transcribe them faithfully and do NOT substitute simplified or traditional Chinese variants. "
        "Hiragana printed at very small sizes may be omitted only if completely illegible."
    ),
    "Korean": (
        "The text uses Korean script (hangul/\ud55c\uae00), possibly mixed with hanja (\u6f22\u5b57) or Latin text. "
        "Transcribe all scripts exactly as they appear."
    ),
    "English": "The text uses the Latin alphabet.",
}

# Keyed by source language name; used by ImageTranslationPromptSpec.
# Uses "source text" phrasing and includes translation-context notes.
IMAGE_TRANSLATION_SCRIPT_GUIDANCE: dict[str, str] = {
    "Chinese": (
        "The source text uses Chinese characters (hanzi/\u6f22\u5b57). "
        "Transcribe each character exactly as it appears."
    ),
    "Simplified Chinese": (
        "The source text uses Simplified Chinese characters (\u7b80\u4f53\u5b57). "
        "Transcribe each character exactly in its simplified form \u2014 "
        "do NOT convert to or substitute traditional variants."
    ),
    "Traditional Chinese": (
        "The source text uses Traditional Chinese characters (\u7e41\u9ad4\u5b57). "
        "Transcribe each character exactly in its traditional form \u2014 "
        "do NOT convert to or substitute simplified variants."
    ),
    "Japanese": (
        "The source text uses Japanese script, which combines kanji (Chinese-derived characters), "
        "hiragana, katakana, and possibly r\u014dmaji. "
        "Reproduce all scripts exactly as written. "
        "Some kanji may be Japanese-specific forms (kokuji) not found in standard Chinese \u2014 "
        "transcribe them faithfully and do NOT substitute simplified or traditional Chinese variants. "
        "Use kanji ambiguity resolution via translation context before committing to a transcript."
    ),
    "Korean": (
        "The source text uses Korean script (hangul/\ud55c\uae00), possibly mixed with hanja (\u6f22\u5b57) or Latin text. "
        "Transcribe all scripts exactly as they appear."
    ),
    "English": "The source text uses the Latin alphabet.",
}
