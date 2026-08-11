from __future__ import annotations

from typing import Any


TARGET_LANGUAGES = ["bn", "sw", "te", "ne", "hi"]


LOW_RESOURCE_ZERO_SHOT_PROMPTS: dict[str, dict[str, str]] = {
    "bn": {
        "system": "আপনি বহুনির্বাচনী প্রশ্নের উত্তর দেন। লুকানো চিন্তা বন্ধ আছে। শুধু JSON আউটপুট দিন।",
        "language": "ভাষা",
        "subject": "বিষয়",
        "question": "প্রশ্ন",
        "options": "বিকল্প",
        "return_json": 'শুধু JSON ফেরত দিন: {"answer":"A"}। উত্তর A, B, C, অথবা D-এর একটি হতে হবে।',
        "repair_system": "আপনি বহুনির্বাচনী উত্তরের অক্ষর বের করেন বা বেছে নেন। লুকানো চিন্তা বন্ধ আছে। শুধু JSON আউটপুট দিন।",
        "prior_response": "আগের মডেল উত্তর, সম্ভবত কাটা বা ভুল ফরম্যাটে:",
        "repair_instruction": 'যদি চূড়ান্ত উত্তর থাকে, সেটি বের করুন। না থাকলে বর্তমান প্রশ্ন ও বিকল্প থেকে সেরা উত্তর বেছে নিন। শুধু JSON ফেরত দিন: {"answer":"A"}। উত্তর A, B, C, অথবা D-এর একটি হতে হবে।',
    },
    "sw": {
        "system": "Unajibu maswali ya chaguo nyingi. Kufikiri kwa siri kumezimwa. Toa JSON pekee.",
        "language": "Lugha",
        "subject": "Somo",
        "question": "Swali",
        "options": "Chaguo",
        "return_json": 'Rudisha JSON pekee: {"answer":"A"}. Jibu liwe mojawapo ya A, B, C, au D.',
        "repair_system": "Unatoa au kuchagua herufi ya jibu la swali la chaguo nyingi. Kufikiri kwa siri kumezimwa. Toa JSON pekee.",
        "prior_response": "Jibu la awali la modeli, huenda limekatwa au lina muundo mbaya:",
        "repair_instruction": 'Ikiwa jibu la mwisho lipo, litoe. Ikiwa halipo, chagua jibu bora kutoka kwenye swali na machaguo ya sasa. Rudisha JSON pekee: {"answer":"A"}. Jibu liwe mojawapo ya A, B, C, au D.',
    },
    "te": {
        "system": "మీరు బహుళ ఎంపిక ప్రశ్నలకు సమాధానం ఇస్తారు. దాచిన ఆలోచన నిలిపివేయబడింది. JSON మాత్రమే ఇవ్వండి.",
        "language": "భాష",
        "subject": "విషయం",
        "question": "ప్రశ్న",
        "options": "ఎంపికలు",
        "return_json": 'JSON మాత్రమే ఇవ్వండి: {"answer":"A"}. సమాధానం A, B, C లేదా D లో ఒకటి కావాలి.',
        "repair_system": "మీరు బహుళ ఎంపిక సమాధాన అక్షరాన్ని తీసుకుంటారు లేదా ఎంచుకుంటారు. దాచిన ఆలోచన నిలిపివేయబడింది. JSON మాత్రమే ఇవ్వండి.",
        "prior_response": "మునుపటి మోడల్ స్పందన, కత్తిరించబడి ఉండవచ్చు లేదా తప్పు రూపంలో ఉండవచ్చు:",
        "repair_instruction": 'చివరి ఉద్దేశించిన సమాధానం ఉంటే దాన్ని తీసుకోండి. లేకపోతే ప్రస్తుత ప్రశ్న మరియు ఎంపికల నుంచి ఉత్తమ సమాధానాన్ని ఎంచుకోండి. JSON మాత్రమే ఇవ్వండి: {"answer":"A"}. సమాధానం A, B, C లేదా D లో ఒకటి కావాలి.',
    },
    "ne": {
        "system": "तपाईं बहुविकल्पीय प्रश्नहरूको उत्तर दिनुहुन्छ। लुकेको सोच बन्द गरिएको छ। JSON मात्र आउटपुट दिनुहोस्।",
        "language": "भाषा",
        "subject": "विषय",
        "question": "प्रश्न",
        "options": "विकल्पहरू",
        "return_json": 'JSON मात्र फर्काउनुहोस्: {"answer":"A"}। उत्तर A, B, C, वा D मध्ये एक हुनुपर्छ।',
        "repair_system": "तपाईं बहुविकल्पीय उत्तरको अक्षर निकाल्नुहुन्छ वा छान्नुहुन्छ। लुकेको सोच बन्द गरिएको छ। JSON मात्र आउटपुट दिनुहोस्।",
        "prior_response": "अघिल्लो मोडेल प्रतिक्रिया, सम्भवतः काटिएको वा गलत ढाँचामा:",
        "repair_instruction": 'यदि अन्तिम अभिप्रेत उत्तर छ भने निकाल्नुहोस्। नभए वर्तमान प्रश्न र विकल्पहरूबाट सबैभन्दा राम्रो उत्तर छान्नुहोस्। JSON मात्र फर्काउनुहोस्: {"answer":"A"}। उत्तर A, B, C, वा D मध्ये एक हुनुपर्छ।',
    },
    "hi": {
        "system": "आप बहुविकल्पीय प्रश्नों के उत्तर देते हैं। छिपी हुई सोच बंद है। केवल JSON आउटपुट दें।",
        "language": "भाषा",
        "subject": "विषय",
        "question": "प्रश्न",
        "options": "विकल्प",
        "return_json": 'केवल JSON लौटाएँ: {"answer":"A"}। उत्तर A, B, C, या D में से एक होना चाहिए।',
        "repair_system": "आप बहुविकल्पीय उत्तर का अक्षर निकालते या चुनते हैं। छिपी हुई सोच बंद है। केवल JSON आउटपुट दें।",
        "prior_response": "पिछला मॉडल उत्तर, संभवतः कटा हुआ या गलत प्रारूप में:",
        "repair_instruction": 'यदि अंतिम इच्छित उत्तर मौजूद है, तो उसे निकालें। यदि नहीं, तो वर्तमान प्रश्न और विकल्पों से सबसे अच्छा उत्तर चुनें। केवल JSON लौटाएँ: {"answer":"A"}। उत्तर A, B, C, या D में से एक होना चाहिए।',
    },
}


def low_resource_zero_shot_prompt_parts(row: dict[str, Any]) -> dict[str, str]:
    language = str(row.get("language") or "")
    if language not in LOW_RESOURCE_ZERO_SHOT_PROMPTS:
        supported = ", ".join(TARGET_LANGUAGES)
        raise ValueError(f"unsupported low-resource language {language!r}; supported low-resource language values: {supported}")
    return LOW_RESOURCE_ZERO_SHOT_PROMPTS[language]


def localized_question_block(row: dict[str, Any], prompt: dict[str, str]) -> str:
    options = row.get("options") or {}
    if isinstance(options, dict):
        options_text = "\n".join(f"{label}. {text}" for label, text in options.items())
    else:
        options_text = "\n".join(str(item) for item in options)
    return (
        f"{prompt['language']}: {row.get('language', '')}\n"
        f"{prompt['subject']}: {row.get('subject', '')}\n"
        f"{prompt['question']}:\n{row.get('question', '')}\n\n"
        f"{prompt['options']}:\n{options_text}"
    )


def zero_shot_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    prompt = low_resource_zero_shot_prompt_parts(row)
    return [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": localized_question_block(row, prompt) + "\n\n" + prompt["return_json"]},
    ]


def zero_shot_answer_repair_prompt(row: dict[str, Any], prior_response: str) -> list[dict[str, str]]:
    prompt = low_resource_zero_shot_prompt_parts(row)
    return [
        {"role": "system", "content": prompt["repair_system"]},
        {
            "role": "user",
            "content": (
                localized_question_block(row, prompt)
                + f"\n\n{prompt['prior_response']}\n{prior_response[-2500:]}\n\n"
                + prompt["repair_instruction"]
            ),
        },
    ]
