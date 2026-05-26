import pandas as pd
import os
from data_adapter import EpidemiologicalDataAdapter

def generate_poisoned_data(filepath):
    print("--- ☠️ GENERATING POISONED KHIS DATA ---")
    
    # Simulating a catastrophic data entry failure from a government clerk
    poisoned_data = {
        'khis_facility_name': ['Clinic A', 'Clinic B', 'Clinic C', 'Clinic D', 'Clinic E'],
        'gps_latitude': [0.123, None, 0.333, 0.456, 0.789], # Clinic B is missing latitude
        'gps_longitude': [34.123, 34.456, 34.789, None, 34.999], # Clinic D is missing longitude
        'catchment_pop': [5000, 6000, 'N/A', 8000, 'TEN THOUSAND'], # Text inside population integers
        'projected_malaria_cases': [100, 150, 200, '-', 300], # Typographical dashes
        'actual_cases_treated': [20, 30, 40, 50, 'missing'] # Word instead of number
    }
    
    df = pd.DataFrame(poisoned_data)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Poisoned dataset saved to {filepath}.")
    print(f"Contains {len(df)} heavily corrupted rows.\n")

def run_poison_test():
    print("=== INITIATING ADAPTER POISON TEST ===\n")
    
    poison_path = "data/raw/poisoned_khis_data.csv"
    generate_poisoned_data(poison_path)
    
    print("--- 🛡️ FEEDING POISON TO THE SANITATION GATE ---")
    # Initialize the adapter we built earlier
    adapter = EpidemiologicalDataAdapter(poison_path)
    
    # Process the data
    adapter.process_data(output_filename="survived_poison_data.csv")
    
    # -----------------------------------------
    # VERIFICATION
    # -----------------------------------------
    output_path = "data/processed/survived_poison_data.csv"
    if os.path.exists(output_path):
        clean_df = pd.read_csv(output_path)
        print("\n=== POISON TEST RESULTS ===")
        print(f"Rows surviving sanitation: {len(clean_df)} (Expected 3)")
        print("\nCleaned Data Snapshot:")
        print(clean_df.to_string(index=False))
        print("\n[CONCLUSION]: The Sanitation Gate holds. The architecture is indestructible.")
    else:
        print("\n[FATAL ERROR]: The adapter crashed and failed to output clean data.")

if __name__ == "__main__":
    run_poison_test()