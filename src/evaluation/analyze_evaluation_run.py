import pandas as pd
import json
import os

def load_golden_dataset(golden_path):
    """Load the golden dataset and return a dict mapping question to metadata."""
    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)
    # Map question text to metadata (type, difficulty, id)
    question_map = {}
    for item in golden:
        question_map[item["question"]] = {
            "question_id": item.get("question_id"),
            "question_type": item.get("question_type"),
            "difficulty": item.get("difficulty"),
        }
    return question_map

def analyze_evaluation_run(csv_path, golden_path):
    # Load evaluation results
    df = pd.read_csv(csv_path)
    # Load golden dataset for metadata
    question_map = load_golden_dataset(golden_path)

    # Add question_type and difficulty columns
    df["question_type"] = df["user_input"].map(lambda q: question_map.get(q, {}).get("question_type", "Unknown"))
    df["difficulty"] = df["user_input"].map(lambda q: question_map.get(q, {}).get("difficulty", "Unknown"))

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print(f"\n=== MODEL EVALUATION SUMMARY ({os.path.basename(csv_path)}) ===\n")
    print(f"Domande totali: {len(df)}")
    print("Metriche disponibili:", ", ".join(metrics))
    print()

    # Overall averages
    print(">> Medie globali:")
    for m in metrics:
        print(f"  {m}: {df[m].mean():.3f}")
    print()

    # By question type
    print(">> Medie per tipo di domanda:")
    for qtype in sorted(df["question_type"].unique()):
        sub = df[df["question_type"] == qtype]
        print(f"  {qtype} ({len(sub)}):")
        for m in metrics:
            print(f"    {m}: {sub[m].mean():.3f}")
    print()

    # By difficulty
    print(">> Medie per difficoltà:")
    for diff in sorted(df["difficulty"].unique()):
        sub = df[df["difficulty"] == diff]
        print(f"  {diff} ({len(sub)}):")
        for m in metrics:
            print(f"    {m}: {sub[m].mean():.3f}")
    print()

    # Best and worst questions
    print(">> Migliore e peggiore domanda (per faithfulness):")
    if len(df) > 0:
        best = df.sort_values("faithfulness", ascending=False).iloc[0]
        worst = df.sort_values("faithfulness", ascending=True).iloc[0]
        print("  Migliore:")
        print(f"    Q: {best['user_input']}\n    Faithfulness: {best['faithfulness']:.3f}")
        print("  Peggiore:")
        print(f"    Q: {worst['user_input']}\n    Faithfulness: {worst['faithfulness']:.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze a RAG evaluation run.")
    parser.add_argument("--csv", required=True, help="Path to evaluation CSV file")
    parser.add_argument("--golden", required=True, help="Path to golden_dataset.json")
    args = parser.parse_args()

    analyze_evaluation_run(args.csv, args.golden)

#python src/evaluation/analyze_evaluation_run.py --csv data/evaluation_results/evaluation_run_rawData_gemini_20250916_150517.csv --golden data/golden_dataset.json    