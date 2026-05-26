import pandas as pd
import os

class EpidemiologicalDataAdapter:
    """
    The Sanitation Gate: Ingests raw, unformatted government or field data, 
    cleans missing coordinates, and outputs a mathematically standardized 
    CSV for the Topological Causal GNN.
    """
    def __init__(self, raw_filepath):
        self.raw_filepath = raw_filepath
        
        # This is the "Blueprint". When you finally get the KHIS data, 
        # you just update these keys to match their exact column names.
        self.column_mapping = {
            'khis_facility_name': 'admin_unit',
            'gps_latitude': 'lat',
            'gps_longitude': 'long',
            'catchment_pop': 'population',
            'projected_malaria_cases': 'expected_cases',
            'actual_cases_treated': 'reported_cases',
            'facility_tier': 'facility_tier' # ADDED for categorical methodology
        }

    def process_data(self, output_filename="clean_khis_data.csv"):
        print(f"--- Initializing Data Adapter for {self.raw_filepath} ---")
        
        if not os.path.exists(self.raw_filepath):
            print(f"Error: Could not find raw data at {self.raw_filepath}.")
            print("Action: Place your actual KHIS or KEMRI .csv in the data/raw/ folder.")
            return

        try:
            # 1. Ingest the dirty data
            df = pd.read_csv(self.raw_filepath)
            original_count = len(df)
            print(f"Ingested {original_count} raw facility records.")

            # 2. Rename the columns to match the Engine's strict schema
            # It only renames the columns we care about and ignores the rest
            df = df.rename(columns=self.column_mapping)

            # 3. Filter down to ONLY the essential columns
            # Safely capture mapped columns that actually exist in the raw file
            essential_cols = [col for col in self.column_mapping.values() if col in df.columns]
            
            # Ensure the strictly required mathematical/topology columns are present
            required_cols = ['admin_unit', 'lat', 'long', 'population', 'expected_cases', 'reported_cases']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"FATAL ERROR: The raw data is missing required mapped columns: {missing_cols}")
                return
                
            df = df[essential_cols]

            # 4. The Topology Check: Drop facilities with missing GPS coordinates
            # A node without a location breaks the spatial graph
            df = df.dropna(subset=['lat', 'long'])
            dropped_coords = original_count - len(df)
            if dropped_coords > 0:
                print(f"Sanitation: Dropped {dropped_coords} facilities missing GPS coordinates.")

            # 5. Type Casting: Ensure numbers are actually numbers, not text
            numeric_cols = ['lat', 'long', 'population', 'expected_cases', 'reported_cases']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # 6. Categorical Encoding (The Fix for Non-Numeric Data)
            # Converts text like "Level 4" into mathematical 1s and 0s for the GNN
            if 'facility_tier' in df.columns:
                print("Sanitation: One-hot encoding categorical facility tiers.")
                df = pd.get_dummies(df, columns=['facility_tier'], drop_first=True)

            # 7. Save the perfectly formatted data for the GNN
            output_path = os.path.join("data", "processed", output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            
            print(f"\n[SUCCESS] Data Adapter Complete.")
            print(f"Output saved to {output_path} ({len(df)} clean facilities ready for modeling).")

        except Exception as e:
            print(f"An error occurred during sanitation: {e}")

if __name__ == "__main__":
    # Example usage for when you finally get the data
    adapter = EpidemiologicalDataAdapter("data/raw/messy_government_data.csv")
    adapter.process_data()