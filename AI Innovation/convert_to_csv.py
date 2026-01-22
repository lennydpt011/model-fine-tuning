import pandas as pd

# 1. Read your existing .jsonl file
print("Reading .jsonl file...")
df = pd.read_json("logbook_validation_100.jsonl", lines=True)

# 2. Save it as a readable CSV
print("Saving as CSV...")
df.to_csv("logbook_validation_100_readable.csv", index=False)

print("Done! Look for 'logbook_validation_100_readable.csv' in your folder.")