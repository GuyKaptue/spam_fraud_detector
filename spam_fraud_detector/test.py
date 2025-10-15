from main import SpamFraudDetector

tasks = ["spam", "fraud"]
for task in tasks:
    print(f"\n\n=== Running Benchmark for {task.upper()} Detection ===")
    detector = SpamFraudDetector(task=task)
    detector.run()
    print(f"=== Completed Benchmark for {task.upper()} Detection ===\n\n")
