import argparse
import json
import os
import re
from typing import Any, Dict, List

DATE_RE = re.compile(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})")

MONTH_MAP = {
    "january": "01", "jan": "01",
    "february": "02", "feb": "02",
    "march": "03", "mar": "03",
    "april": "04", "apr": "04",
    "may": "05",
    "june": "06", "jun": "06",
    "july": "07", "jul": "07",
    "august": "08", "aug": "08",
    "september": "09", "sep": "09",
    "october": "10", "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "dec": "12"
}


def parse_date(date_str: str) -> str:
    if not date_str:
        return ""

    match = DATE_RE.search(date_str.lower())
    if match:
        day = match.group(1).zfill(2)
        month_name = match.group(2).lower()
        year = match.group(3)
        month = MONTH_MAP.get(month_name, "00")
        return f"{year}/{month}/{day}"
    return date_str


def extract_evidence_prefix(evidence_list):
    prefixes = []
    for ev in evidence_list:
        if isinstance(ev, str) and ":" in ev:
            prefix = ev.split(":", 1)[0]
            if prefix not in prefixes:
                prefixes.append(prefix)
    return prefixes


def build_context(session_messages: List[Dict[str, Any]]) -> str:
    parts = []
    for msg in session_messages:
        speaker = msg.get("speaker", "")
        text = msg.get("text", "")

        blip_caption = msg.get("blip_caption", "")
        if blip_caption:
            text = f"{text} (attached is {blip_caption})"
        parts.append(f'"{speaker}": "{text}"')

    return "".join(parts)


def process_group(group_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    conversation = group_data.get("conversation", {})
    qa_list = group_data.get("qa", [])

    corpus_records = []
    session_num = 1

    while True:
        session_key = f"session_{session_num}"
        date_key = f"session_{session_num}_date_time"

        if session_key not in conversation:
            break

        session_messages = conversation.get(session_key, [])
        session_date = conversation.get(date_key, "")

        session_time = parse_date(session_date)
        context = build_context(session_messages)

        session_id = f"D{session_num}"

        corpus_records.append({
            "session_time": session_time,
            "context": context,
            "session_id": session_id,
        })

        session_num += 1

    question_records = []
    for qa_item in qa_list:
        question = qa_item.get("question", "")
        answer = qa_item.get("answer", "")
        if not answer:
            answer = "Context insufficient to answer"
        else:
            if not isinstance(answer, str):
                answer = str(answer)

        evidence = qa_item.get("evidence", [])
        category = qa_item.get("category", "")

        origin = extract_evidence_prefix(evidence)
        if len(origin) == 1:
            origin = origin[0]

        question_records.append({
            "question": question,
            "answer": answer,
            "question_type": str(category),
            "origin": origin,
            "label": answer
        })

    return {
        "corpus": corpus_records,
        "question": question_records,
    }


def write_ndjson(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write NDJSON file (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Process LOCOMO dataset into group folders with corpus and question files"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input LOCOMO JSON file"
    )
    parser.add_argument(
        "--outdir", "-o",
        required=True,
        help="Output directory for group folders"
    )
    args = parser.parse_args()

    in_path = args.input
    out_root = args.outdir

    # Create output directory
    os.makedirs(out_root, exist_ok=True)

    # Load data
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of groups.")

    total_groups = 0
    total_sessions = 0
    total_questions = 0

    # Process each group
    for idx, group_data in enumerate(data, start=1):
        if not isinstance(group_data, dict):
            continue

        # Create group folder
        group_folder = os.path.join(out_root, f"group_{idx}")
        os.makedirs(group_folder, exist_ok=True)

        # Process the group
        result = process_group(group_data)
        corpus_records = result["corpus"]
        question_records = result["question"]

        # Write files
        corpus_out = os.path.join(group_folder, "Corpus.json")
        question_out = os.path.join(group_folder, "Question.json")

        write_ndjson(corpus_out, corpus_records)
        write_ndjson(question_out, question_records)

        total_groups += 1
        total_sessions += len(corpus_records)
        total_questions += len(question_records)

        print(f"[OK] {group_folder} -> {len(corpus_records)} sessions, {len(question_records)} questions")

    print(
        f"\nDone. Processed {total_groups} groups with {total_sessions} sessions and {total_questions} questions in total.")
    print(f"Output directory: {out_root}")


if __name__ == "__main__":
    main()