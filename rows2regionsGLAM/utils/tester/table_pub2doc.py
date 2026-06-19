import re
from pathlib import Path
from typing import Dict


MAP_PATTERN = re.compile(r"mAP@IoU\[0\.50:0\.95\]\s*:\s*([0-9]*\.?[0-9]+)")


def _extract_last_map(log_path: Path) -> float:
    last_match = None

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = MAP_PATTERN.search(line)
            if match:
                last_match = float(match.group(1))

    if last_match is None:
        raise ValueError(f"В файле {log_path} метрика mAP не найдена")

    return last_match


def collect_maps(log_dir: str) -> Dict[str, Dict[str, float]]:

    log_dir = Path(log_dir)

    files = {
        ("PubLayNet", "PubLayNet"): log_dir / "log_train_pub_test_pub.txt",
        ("DocLayNet", "PubLayNet"): log_dir / "log_train_pub_test_doc.txt",
        ("DocLayNet", "DocLayNet"): log_dir / "log_train_doc_test_doc.txt",
        ("PubLayNet", "DocLayNet"): log_dir / "log_train_doc_test_pub.txt",
    }

    table = {
        "PubLayNet": {},
        "DocLayNet": {},
    }

    for (test_ds, train_ds), path in files.items():
        table[test_ds][train_ds] = _extract_last_map(path)

    return table


def print_map_table(table: Dict[str, Dict[str, float]]) -> None:

    header = f"{'test\\train':<12} | {'PubLayNet':<10} | {'DocLayNet':<10}"
    sep = "-" * len(header)

    print(header)
    print(sep)

    for test_ds in ["PubLayNet", "DocLayNet"]:
        pub = table[test_ds].get("PubLayNet", float("nan"))
        doc = table[test_ds].get("DocLayNet", float("nan"))

        print(f"{test_ds:<12} | {pub:<10.6f} | {doc:<10.6f}")
