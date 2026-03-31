# This is the final version of the data_processor.py script.
# Its only purpose now is to create a correctly-sized sample file.

import lightkurve as lk
import matplotlib.pyplot as plt
from wotan import flatten

print("--- CREATING CORRECTLY-SIZED SAMPLE DATA (V22) ---")

if __name__ == '__main__':
    test_star_id = "KIC 11904151"
    
    search_result = lk.search_lightcurve(test_star_id, mission='Kepler', author='Kepler')
    lc_collection = search_result.download_all()
    light_curve = lc_collection.stitch().remove_nans()
    
    # CORRECTED LINE: We select exactly 3197 points to match the AI model's expectation.
    light_curve = light_curve[:3197]
    print(f"--- Using a subset of {len(light_curve)} data points to match the AI ---")

    # Flatten the light curve using the fast wotan method
    flattened_lc, _ = flatten(light_curve.time.value, light_curve.flux.value, window_length=0.5, return_trend=True)
    flattened_lc = lk.LightCurve(time=light_curve.time.value, flux=flattened_lc)

    # Save the correctly-sized, cleaned data to a CSV file
    flattened_lc.to_csv('sample_data.csv', overwrite=True)
    print("--- Successfully saved new sample_data.csv with 3197 features. ---")