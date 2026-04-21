"""
Parse history.txt and merge with test results for reinforcement learning dataset.
"""
import json
import re
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent.parent  # project root
HISTORY = BASE / "Cricket Data" / "statistical_analysis" / "history_backup_before_clear.txt"
RL_FILE = BASE / "Cricket Data" / "statistical_analysis" / "reinforcement_learning_queries.json"
OUTPUT = BASE / "Cricket Data" / "statistical_analysis" / "reinforcement_learning_dataset.json"

# Parse history.txt
history_text = HISTORY.read_text(encoding="utf-8")
pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] Query Type: (\w+)\nQuestion: (.*?)\nAnswer: (.*?)\n-{40,}'
matches = re.findall(pattern, history_text, re.DOTALL)

history_entries = []
for ts, qtype, question, answer in matches:
    answer = answer.strip()
    is_failure = (
        "could not generate" in answer.lower() or
        answer == "" or
        len(answer) < 20
    )
    history_entries.append({
        "source": "user_history",
        "timestamp": ts,
        "query_type": qtype,
        "question": question.strip(),
        "answer": answer,
        "status": "FAIL" if is_failure else "PASS",
        "answer_length": len(answer),
    })

# Load test results
with open(RL_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

test_entries = []
for r in test_data["results"]:
    test_entries.append({
        "source": "automated_test",
        "timestamp": r["timestamp"],
        "query_type": r["query_type"],
        "question": r["question"],
        "answer": r["answer"],
        "status": r["status"],
        "answer_length": r["answer_length"],
    })

# Combine
all_entries = history_entries + test_entries

# Stats
total = len(all_entries)
passed = sum(1 for e in all_entries if e["status"] == "PASS")
failed = sum(1 for e in all_entries if e["status"] != "PASS")
hist_pass = sum(1 for e in history_entries if e["status"] == "PASS")
hist_fail = sum(1 for e in history_entries if e["status"] != "PASS")
test_pass = sum(1 for e in test_entries if e["status"] == "PASS")
test_fail = sum(1 for e in test_entries if e["status"] != "PASS")

dataset = {
    "description": "Reinforcement learning dataset for Cricket World Cup RAG Assistant",
    "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "statistics": {
        "total_entries": total,
        "total_passed": passed,
        "total_failed": failed,
        "history_entries": len(history_entries),
        "history_passed": hist_pass,
        "history_failed": hist_fail,
        "test_entries": len(test_entries),
        "test_passed": test_pass,
        "test_failed": test_fail,
    },
    "failed_queries": [e for e in all_entries if e["status"] != "PASS"],
    "all_entries": all_entries,
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Parsed {len(history_entries)} history entries ({hist_pass} pass, {hist_fail} fail)")
print(f"Loaded {len(test_entries)} test entries ({test_pass} pass, {test_fail} fail)")
print(f"Total: {total} entries → {OUTPUT}")
print(f"Failed queries saved: {failed}")
