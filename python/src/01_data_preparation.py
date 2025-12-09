import pandas as pd
import numpy as np
import janitor
import os
import sys

# Add src to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import clean_names, get_buffer_percent

def load_data():
    # definitions
    url1 = "https://zenodo.org/records/17237416/files/hmdb_keep_v4_python_blessed.txt?download=1"
    url2 = "https://zenodo.org/records/17237416/files/omics1799_data.txt?download=1"
    url3 = "https://zenodo.org/records/17237416/files/validation3990_data.txt?download=1"
    url4 = "https://zenodo.org/records/17237416/files/ovca_metabolomics.tsv?download=1"

    print("Loading data...")
    try:
        metabolites_classification = pd.read_csv(url1, sep="\t").clean_names()
        omics1799_data = pd.read_csv(url2, sep="\t").clean_names()
        validation3990_data = pd.read_csv(url3, sep="\t").clean_names()
        validation_ovca = pd.read_csv(url4, sep="\t").clean_names()
    except Exception as e:
        print(f"Error loading data from URLs: {e}")
        # Fallback to local if available or fail
        raise e

    # Validation OVCA specific processing
    # rename(row_retention_time = rt, row_m_z = m_z, hmdb = hmdb_id)
    validation_ovca = validation_ovca.rename(columns={
        "rt": "row_retention_time",
        "m_z": "row_m_z",
        "hmdb_id": "hmdb"
    })
    
    # mutate(hmdb1 = str_extract(hmdb, "(\\d+)")) -> python regex
    # mutate(hmdb = paste0("HMDB00", hmdb1))
    # Note: R code: str_extract(hmdb, "(\\d+)") extracts the first sequence of digits.
    # paste0("HMDB00", hmdb1)
    
    def fix_hmdb(h):
        if pd.isna(h): return h
        match = re.search(r"(\d+)", str(h))
        if match:
            return f"HMDB00{match.group(1)}"
        return h
        
    import re
    validation_ovca['hmdb'] = validation_ovca['hmdb'].apply(fix_hmdb)

    return metabolites_classification, omics1799_data, validation3990_data, validation_ovca

def process_data(omics_data, classification_data, is_validation=False):
    # Filter RT
    # filter(row_retention_time > 1 & row_retention_time < 14)
    clean_data = omics_data[
        (omics_data['row_retention_time'] > 1) & 
        (omics_data['row_retention_time'] < 14)
    ].copy()

    if not is_validation:
         # Minimize overlap in identified metabolites
         # 1. Identified (flag == 1): Give priority to negative mode
         
         # Split row_id to get charge
         # separate(row_id, into = c("charge", "id"))
         # row_id example: 'neg_00001', 'pos_00001'? Needs verification from R exploration logic which implies row_id contains charge.
         # Actually R code: separate(row_id, into = c("charge", "id"))
         
         # We need to maintain row_id or at least be able to separate it.
         # Let's add temporary columns
         clean_data[['charge', 'metabolite_id']] = clean_data['row_id'].str.split('_', n=1, expand=True)
         
         # 1. Identified
         identified = clean_data[clean_data['non_heavy_identified_flag'] == 1].copy()
         # sort by id, then charge (neg < pos, so negative comes first alphabetically? neg vs pos. n vs p. n comes before p. correct.)
         identified = identified.sort_values(by=['metabolite_id', 'charge'])
         # distinct id, keep first (which is negative if both exist)
         identified = identified.drop_duplicates(subset=['metabolite_id'], keep='first')
         
         # 2. Non-identified
         non_identified = clean_data[clean_data['non_heavy_identified_flag'] == 0].copy()
         non_identified = non_identified.sort_values(by=['metabolite_id', 'charge'])
         non_identified = non_identified.drop_duplicates(subset=['metabolite_id'], keep='first')
         
         # Bind
         clean_data = pd.concat([identified, non_identified], ignore_index=True)
         # distinct id again? The R script does `bind_rows(...) %>% distinct(id, .keep_all = TRUE)`
         # This handles if an ID was in both identified and non-identified? Unlikely but possible.
         clean_data = clean_data.drop_duplicates(subset=['metabolite_id'], keep='first')

    else:
        # Validation logic (same as above for validation3990)
        clean_data[['charge', 'metabolite_id']] = clean_data['row_id'].str.split('_', n=1, expand=True)

        identified = clean_data[clean_data['non_heavy_identified_flag'] == 1].copy()
        identified = identified.sort_values(by=['metabolite_id', 'charge'])
        identified = identified.drop_duplicates(subset=['metabolite_id'], keep='first')

        non_identified = clean_data[clean_data['non_heavy_identified_flag'] == 0].copy()
        non_identified = non_identified.sort_values(by=['metabolite_id', 'charge'])
        non_identified = non_identified.drop_duplicates(subset=['metabolite_id'], keep='first')
        
        clean_data = pd.concat([identified, non_identified], ignore_index=True)
        clean_data = clean_data.drop_duplicates(subset=['metabolite_id'], keep='first')

    # Annotation
    # mutate(hmdb = str_remove(hmdb, "^\\|"))
    # separate_wider_delim ...
    # This implies hmdb column might have pipes.
    
    def clean_hmdb(h):
         if pd.isna(h): return h
         h = str(h).lstrip('|')
         # take first part before '|'
         return h.split('|')[0]

    clean_data['hmdb_clean'] = clean_data['hmdb'].apply(clean_hmdb)
    
    # Left join with classification
    # by = c("hmdb" = "accession")
    clean_data = pd.merge(
        clean_data, 
        classification_data, 
        left_on='hmdb_clean', 
        right_on='accession', 
        how='left'
    )
    
    # Create is_lipids
    # R: row_id == "neg_00049" ~ "No", str_detect(taxonomy_super_class, "Lipids") ~ "Yes", else "No"
    
    def check_lipid(row):
        if row['row_id'] == 'neg_00049':
            return 'No'
        if pd.notna(row['taxonomy_super_class']) and 'Lipids' in row['taxonomy_super_class']:
            return 'Yes'
        return 'No'

    clean_data['is_lipids'] = clean_data.apply(check_lipid, axis=1)
    
    # Calculate buffer percent
    # 4.615 * row_retention_time + 20 (already imported function for this logic, but vectorizing is better)
    # clean_data['buffer_percent'] = clean_data['row_retention_time'].apply(get_buffer_percent) # Using util if we want
    
    # Vectorized:
    clean_data['buffer_percent'] = np.nan
    mask = (clean_data['row_retention_time'] >= 0) & (clean_data['row_retention_time'] <= 13)
    clean_data.loc[mask, 'buffer_percent'] = 4.615 * clean_data.loc[mask, 'row_retention_time'] + 20
    
    return clean_data

def main():
    metabolites_classification, omics1799, validation3990, validation_ovca = load_data()
    
    print("Processing Discovery Data...")
    full_data = process_data(omics1799, metabolites_classification, is_validation=False)
    
    print("Processing Validation Data (3990)...")
    validation3990_processed = process_data(validation3990, metabolites_classification, is_validation=True)
    
    # Validation OVCA logic
    # It has inner join in R, and simplified lipid check
    print("Processing Validation OVCA...")
    validation_ovca_processed = validation_ovca[
        (validation_ovca['row_retention_time'] > 1) & 
        (validation_ovca['row_retention_time'] < 14)
    ].copy()
    
    # inner join
    validation_ovca_processed = pd.merge(
        validation_ovca_processed,
        metabolites_classification,
        left_on='hmdb',
        right_on='accession',
        how='inner'
    )
    
    validation_ovca_processed['is_lipids'] = validation_ovca_processed['taxonomy_super_class'].apply(
        lambda x: 'Yes' if pd.notna(x) and 'Lipids' in x else 'No'
    )
    
    mask = (validation_ovca_processed['row_retention_time'] >= 0) & (validation_ovca_processed['row_retention_time'] <= 13)
    validation_ovca_processed.loc[mask, 'buffer_percent'] = 4.615 * validation_ovca_processed.loc[mask, 'row_retention_time'] + 20
    
    # Save Outputs
    os.makedirs('data/processed', exist_ok=True)
    
    # Create Final minimal data including 2 predictors (m/z and gradient concentration) and the outcome (is_lipids)
    # two_predictor_data <- full_data %>% filter(non_heavy_identified_flag == 1) %>% select(hmdb, row_m_z, buffer_percent, is_lipids)
    
    two_predictor_data = full_data[full_data['non_heavy_identified_flag'] == 1][
        ['hmdb', 'row_m_z', 'buffer_percent', 'is_lipids']
    ]
    
    print("Saving two_predictor_data...")
    two_predictor_data.to_parquet('data/processed/two_predictor_data.parquet')
    full_data.to_parquet('data/processed/full_data.parquet')
    validation3990_processed.to_parquet('data/processed/validation3990_data.parquet')
    validation_ovca_processed.to_parquet('data/processed/validation_ovca.parquet')
    
    print("Done.")

if __name__ == "__main__":
    main()
