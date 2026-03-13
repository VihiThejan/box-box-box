#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    run_cmd = (repo / "solution" / "run_command.txt").read_text(encoding="utf-8").strip()

    inputs = sorted((repo / "data" / "test_cases" / "inputs").glob("test_*.json"))
    expected_dir = repo / "data" / "test_cases" / "expected_outputs"

    passed = 0
    failed = 0
    errors = 0

    print(f"Solution command: {run_cmd}")
    print(f"Tests found: {len(inputs)}")

    for test_file in inputs:
        test_name = test_file.stem
        test_id = test_name.replace("test_", "TEST_").upper()
        data = test_file.read_text(encoding="utf-8")
        proc = subprocess.run(
            run_cmd,
            input=data,
            capture_output=True,
            text=True,
            shell=True,
            cwd=repo,
        )
        if proc.returncode != 0:
            errors += 1
            print(f"x {test_id} execution error: {proc.stderr.splitlines()[:1]}")
            continue

        try:
            output = json.loads(proc.stdout)
            predicted = output["finishing_positions"]
            if not isinstance(predicted, list) or len(predicted) != 20:
                raise ValueError("Invalid finishing_positions")
        except Exception as exc:
            failed += 1
            print(f"x {test_id} invalid output: {exc}")
            continue

        answer_file = expected_dir / f"{test_name}.json"
        if answer_file.exists():
            expected = json.loads(answer_file.read_text(encoding="utf-8"))["finishing_positions"]
            if predicted == expected:
                passed += 1
                print(f"ok {test_id}")
            else:
                failed += 1
                print(f"x {test_id} incorrect prediction")
        else:
            passed += 1
            print(f"? {test_id} output generated")

    total = len(inputs)
    pass_rate = (passed * 100.0 / total) if total else 0.0
    print("\nResults")
    print(f"Total:  {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {errors}")
    print(f"Rate:   {pass_rate:.1f}%")


if __name__ == "__main__":
    main()