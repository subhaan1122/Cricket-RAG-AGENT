"""
Complex Query Test Suite for Cricket World Cup RAG Assistant
Runs difficult queries, saves results for reinforcement learning.
"""
import json
import sys
import os
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path so backend modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from main import CricketChatbot

# ── Complex test queries covering all categories ──
TEST_QUERIES = [
    # Player career profiles (previously failed)
    ("player", "Tell me the complete performance of MS Dhoni across all World Cups from 2003 to 2023"),
    ("player", "How did Virat Kohli perform in each World Cup edition? Give tournament-by-tournament stats"),
    ("player", "Compare Ricky Ponting and Sachin Tendulkar's World Cup careers with runs in each edition"),
    ("player", "What are Kumar Sangakkara's batting stats across all World Cups he played in?"),
    
    # Death overs and phase statistics (previously failed)
    ("statistical", "Who are the best death overs bowlers in World Cup history from 2003 to 2023?"),
    ("statistical", "What are the death overs batting and bowling statistics across World Cup tournaments?"),
    ("statistical", "Which bowlers have the best economy rate in the powerplay overs in World Cup history?"),
    
    # World records (previously failed)
    ("statistical", "Has any team scored over 400 in a World Cup match? List all 400+ team scores"),
    ("statistical", "What is the fastest century in World Cup history and who scored it?"),
    ("statistical", "What is the highest individual score ever in a World Cup match?"),
    ("statistical", "List the top 5 highest team totals in World Cup history 2003-2023"),
    
    # Captaincy comparisons (previously failed)
    ("comparative", "Compare Eoin Morgan and Kane Williamson as World Cup captains with detailed stats"),
    ("comparative", "Which captain has the best win percentage in World Cup history from 2003 to 2023?"),
    
    # Memorable moments (previously failed)
    ("tournament", "What were the most memorable moments of the 2015 World Cup?"),
    ("tournament", "What were the key highlights and memorable events of the 2007 World Cup?"),
    ("tournament", "Describe the most dramatic moments from the 2023 World Cup"),
    
    # 2019 Final specifics (previously hallucinated)
    ("match_specific", "How did the 2019 World Cup final end? Explain the Super Over and boundary count rule"),
    ("match_specific", "What was the overthrow controversy in the 2019 World Cup final?"),
    
    # Venue statistics (previously failed)
    ("statistical", "What are the average scores at major World Cup venues? Which ground has the highest average first innings score?"),
    
    # Team aggregate records
    ("statistical", "What is India's win-loss record across all World Cups from 2003 to 2023?"),
    ("comparative", "Which team has the worst win-loss ratio in World Cups from 2003 to 2023 with minimum 20 matches?"),
    ("statistical", "What is Australia's World Cup record? How many titles have they won from 2003-2023?"),
    
    # Awards and milestones
    ("tournament", "Who won the Player of the Tournament award in each World Cup from 2003 to 2023?"),
    ("statistical", "Who were the top run scorers (Golden Bat winners) in each World Cup edition?"),
    
    # Head-to-head
    ("comparative", "What is India vs Pakistan head to head record in World Cups from 2003 to 2023?"),
    
    # Upsets and controversies
    ("tournament", "What were the biggest upsets in World Cup history from 2003 to 2023?"),
    ("tournament", "Tell me about Bob Woolmer's death during the 2007 World Cup"),
    
    # Complex multi-part queries
    ("comparative", "Compare Glenn Maxwell's 201* vs Afghanistan in 2023 with other great World Cup knocks"),
    ("statistical", "How many centuries did Rohit Sharma score in the 2019 World Cup? Is that a record?"),
    ("match_specific", "Describe the 2011 World Cup final. How did Dhoni finish the match?"),
]

def main():
    print("=" * 70)
    print("CRICKET WORLD CUP RAG ASSISTANT — COMPLEX QUERY TEST SUITE")
    print("=" * 70)
    print(f"Testing {len(TEST_QUERIES)} complex queries...")
    print()

    chatbot = CricketChatbot()
    chatbot.initialize()

    results = []
    passed = 0
    failed = 0

    for i, (qtype, query) in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] Testing: {query[:80]}...")
        try:
            result = chatbot.ask(query)
            answer = result.get("answer", "")
            
            # Check if it failed to generate a response
            is_failure = (
                "could not generate" in answer.lower() or
                "i don't have" in answer.lower() or
                answer.strip() == "" or
                len(answer) < 20
            )
            
            status = "FAIL" if is_failure else "PASS"
            if is_failure:
                failed += 1
                print(f"  ❌ FAIL — No useful response generated")
            else:
                passed += 1
                print(f"  ✅ PASS — Got {len(answer)} char response")
            
            results.append({
                "query_number": i,
                "query_type": qtype,
                "question": query,
                "answer": answer,
                "status": status,
                "answer_length": len(answer),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "retrieval_score": result.get("confidence", None),
            })
            
        except Exception as e:
            failed += 1
            print(f"  ❌ ERROR — {str(e)[:100]}")
            results.append({
                "query_number": i,
                "query_type": qtype,
                "question": query,
                "answer": f"ERROR: {str(e)}",
                "status": "ERROR",
                "answer_length": 0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        
        # Small delay to avoid rate limiting
        time.sleep(1)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total queries:  {len(TEST_QUERIES)}")
    print(f"Passed:         {passed}")
    print(f"Failed:         {failed}")
    print(f"Pass rate:      {passed/len(TEST_QUERIES)*100:.1f}%")
    print()

    # Save results for reinforcement learning
    rl_file = Path(__file__).parent / "Cricket Data" / "statistical_analysis" / "reinforcement_learning_queries.json"
    with open(rl_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_queries": len(TEST_QUERIES),
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{passed/len(TEST_QUERIES)*100:.1f}%",
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {rl_file}")

    # Also save just the queries for future use
    queries_file = Path(__file__).parent / "Cricket Data" / "statistical_analysis" / "complex_test_queries.json"
    with open(queries_file, "w", encoding="utf-8") as f:
        json.dump({
            "description": "Complex test queries for Cricket World Cup RAG Assistant reinforcement learning",
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "queries": [{"type": t, "question": q} for t, q in TEST_QUERIES]
        }, f, indent=2, ensure_ascii=False)
    print(f"Queries saved to: {queries_file}")

    # Print failed queries for review
    if failed > 0:
        print("\n" + "=" * 70)
        print("FAILED QUERIES:")
        print("=" * 70)
        for r in results:
            if r["status"] != "PASS":
                print(f"  [{r['query_number']}] {r['question'][:80]}")
                print(f"      → {r['answer'][:120]}")
                print()

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
