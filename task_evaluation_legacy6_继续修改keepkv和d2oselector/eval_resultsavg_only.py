import argparse
import json
from LongBench.eval import collect_all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("-e", action="store_true", help="evaluate LongBench-E")
    args = parser.parse_args()

    collect_all_results(args.pred_path.rstrip("/") + "/", args.out_path, args_e=args.e)

    with open(args.out_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    scores = [v for v in result.values() if isinstance(v, (int, float))]
    result["avg"] = round(sum(scores) / len(scores), 2)

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()