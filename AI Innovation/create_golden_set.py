import pandas as pd
from datasets import load_dataset
import re
import sys

# 1. SETUP KEYWORDS (Same as before)
keywords = {
    "instruments": [
        'apple', 'aapl', 'tesla', 'tsla', 'bitcoin', 'btc', 'eth', 'stock', 'share', 'dividend',
        '401k', 'ira', 'roth', 'pension', 'etf', 'mutual fund', 'index fund', 's&p 500',
        'insurance', 'annuity', 'bond', 'treasury', 'security', 'option', 'future', 'derivative',
        'mortgage', 'real estate', 'property', 'refinance', 'equity'
    ],
    "actions": [
        r'\bbuy\b', r'\bsell\b', r'\bhold\b', r'\binvest\b', r'\bshort\b', 
        r'\bswitch\b', r'\ballocate\b', r'\bbalance\b', r'\bchurn\b',
        r'\bconsider\b', r'\byou should\b', r'\boptimal\b', r'\bbest choice\b'
    ],
    "personalization": [
        'i am', 'my age', 'my goal', 'my salary', 'my income', 
        'risk tolerance', 'retirement', 'years old', 'have \$', 
        'my portfolio', 'my debt', 'spouse', 'children', 'dependents'
    ]
}

def get_category(row):
    # Only look at instruction and output
    text = (str(row['instruction']) + " " + str(row['output'])).lower()
    
    has_instrument = any(re.search(rf'\b{re.escape(word)}\b', text) for word in keywords["instruments"])
    has_action = any(re.search(pattern, text) for pattern in keywords["actions"])
    has_personal = any(word in text for word in keywords["personalization"])
    
    if has_instrument and (has_action or has_personal):
        return "Advice"
    elif has_personal and not has_instrument:
        return "Edge Case"
    elif not has_personal and not has_action:
        return "Guidance"
    else:
        return "Unsure"

def main():
    print("--- Starting Clean Generation ---")

    # 1. Load Data
    print("1. Loading dataset...")
    dataset = load_dataset("gbharti/finance-alpaca", split='train')
    df = dataset.to_pandas()

    # 2. Filter: Remove rows that have 'Input' content
    print("2. Removing reading comprehension tasks...")
    # We keep rows where 'input' is empty or None
    df_clean = df[df['input'].astype(str).str.strip() == ''].copy()
    print(f"   > Dropped {len(df) - len(df_clean)} rows.")
    print(f"   > Remaining: {len(df_clean)} rows.")

    # 3. Categorize
    print("3. Categorizing...")
    df_clean['presumed_label'] = df_clean.apply(get_category, axis=1)

    # 4. Sample 100
    print("4. Selecting 100 samples...")
    try:
        advice = df_clean[df_clean['presumed_label'] == "Advice"].sample(40, random_state=42)
        
        # Get Edge Cases (Priority)
        edge_pool = df_clean[df_clean['presumed_label'] == "Edge Case"]
        edge_count = min(30, len(edge_pool))
        edge = edge_pool.sample(edge_count, random_state=42)
        
        guidance = df_clean[df_clean['presumed_label'] == "Guidance"].sample(30, random_state=42)
        
        # Combine
        golden_set = pd.concat([advice, edge, guidance])

        # 5. STRICT COLUMN SELECTION (This fixes your issue)
        # We create a new table keeping ONLY these 3 columns.
        final_table = golden_set[['instruction', 'output', 'presumed_label']].copy()
        
    except Exception as e:
        sys.exit(f"Error: {e}")

    # 6. Save
    print("5. Saving files...")
    final_table.to_json("logbook_validation_100.jsonl", orient="records", lines=True)
    final_table.to_csv("logbook_validation_100_readable.csv", index=False)
    
    print("\nSUCCESS!")
    print(f"Files saved. Columns are: {list(final_table.columns)}")

if __name__ == "__main__":
    main()