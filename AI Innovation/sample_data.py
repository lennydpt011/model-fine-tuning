import pandas as pd
from datasets import load_dataset
import re

# 1. Load the finance-alpaca dataset
print("Loading gbharti/finance-alpaca dataset...")
dataset = load_dataset("gbharti/finance-alpaca", split='train')
df = dataset.to_pandas()

# 2. Refined Keywords from Logbook
# Specific Instruments
instruments = ['apple', 'aapl', 'tesla', 'tsla', 'bitcoin', 'btc', 'eth', '401k', 'ira', 'pension', 'etf', 'security', 'bond']
# Action-Oriented Verbs
actions = [r'\bbuy\b', r'\bsell\b', r'\bhold\b', r'\binvest in\b', r'\bdivest from\b', r'\bswitch to\b', r'\ballocate\b', r'\bshort\b']
# Personal Circumstances
personal = ['i am', 'my age', 'my goal', 'my salary', 'risk tolerance', 'retirement', 'my house', 'given you are']

# 3. Decision Boundary Filtering Logic
def identify_logbook_category(row):
    text = (str(row['instruction']) + " " + str(row['input'])).lower()
    
    has_instrument = any(inst in text for inst in instruments)
    has_action = any(re.search(pattern, text) for pattern in actions)
    has_personal = any(p in text for p in personal)
    
    # Category: Explicit Advice (Meets all 3 Logbook prerequisites)
    if (has_personal or has_action) and has_instrument:
        return "Advice"
    # Category: Edge Case (Personalized but no specific instrument/explicit action)
    elif has_personal and not (has_instrument or has_action):
        return "Edge Case"
    # Category: Educational/Guidance
    else:
        return "Guidance"

df['potential_category'] = df.apply(identify_logbook_category, axis=1)

# 4. Stratified Sampling for the "Golden 100"
# Targeting high-entropy boundaries with 100 high-quality examples
advice_samples = df[df['potential_category'] == "Advice"].sample(40, random_state=42)
edge_samples = df[df['potential_category'] == "Edge Case"].sample(30, random_state=42)
guidance_samples = df[df['potential_category'] == "Guidance"].sample(30, random_state=42)

golden_set = pd.concat([advice_samples, edge_samples, guidance_samples])

# 5. Export to JSONL for Fine-tuning/Validation
golden_set.to_json("logbook_validation_100.jsonl", orient="records", lines=True)

print(f"Success! Created validation set with {len(golden_set)} samples.")
print("- 40 Advice samples (Targeting Specificity/Action)")
print("- 30 Edge Case samples (Targeting subtle Personalization)")
print("- 30 Guidance samples (Targeting Objective/Factual tone)")