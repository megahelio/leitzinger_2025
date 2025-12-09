import pandas as pd
import re
import janitor

def clean_names(df):
    """
    Clean column names to snake_case using pyjanitor logic or manual regex.
    """
    return df.clean_names()

def get_buffer_percent(retention_time):
    """
    Calculate buffer concentration based on retention time.
    Formula from R script: 4.615 * row_retention_time + 20 for RT <= 13
    """
    # R logic:
    # row_retention_time >= 0 & row_retention_time <= 13 ~ 4.615 * row_retention_time + 20
    # The R script only assigned this to rows matching the condition, leaving others as NA? 
    # In R case_when defaults to NA if not matched.
    if 0 <= retention_time <= 13:
        return 4.615 * retention_time + 20
    return None
