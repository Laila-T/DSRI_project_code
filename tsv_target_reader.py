import numpy as np

def load_targets_from_tsv(tsv_path):
    targets = []
    with open(tsv_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                value = float(parts[1])
                targets.append(value)
    return np.array(targets)

def save_targets_per_dataset(tsv_path, feature_file_paths_lh, feature_file_paths_rh, output_paths):
    #Load all targets from tsv
    all_targets = load_targets_from_tsv(tsv_path)

    #Calculate sample counts per dataset
    sample_counts = []
    for lh_path, rh_path in zip(feature_file_paths_lh, feature_file_paths_rh):
        X_lh = np.load(lh_path)
        X_rh = np.load(rh_path)
        assert X_lh.shape[0] == X_rh.shape[0], "Mismatch in samples between LH and RH features"
        sample_counts.append(X_lh.shape[0])

    #Splitting targets according to sample counts
    start = 0
    for count, out_path in zip(sample_counts, output_paths):
        subset = all_targets[start:start+count]
        np.save(out_path, subset)
        print(f"Saved {count} targets to {out_path}")
        start += count

if __name__ == "__main__":
    #Paths to the features
    feature_lh_files = [
        'C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2015_data\\features_LH.npy',
        'C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2016_Session1\\features_LH.npy',
        'C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2016_Session2\\features_LH.npy',
        'C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2016_Session3\\features_LH.npy'
    ]

    feature_rh_files = [
        'C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2015_data\\features_RH.npy',
        'C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2016_Session1\\features_RH.npy',
        'C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2016_Session2\\features_RH.npy',
        'C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2016_Session3\\features_RH.npy'
    ]

    output_target_files = [
        'C:\\Users\\Laila\\InceptionTime\\targets_2015.npy',
        'C:\\Users\\Laila\\InceptionTime\\targets_2016_Session1.npy',
        'C:\\Users\\Laila\\InceptionTime\\targets_2016_Session2.npy',
        'C:\\Users\\Laila\\InceptionTime\\targets_2016_Session3.npy',
    ]

    targets_tsv_path = 'C:\\Users\\Laila\\InceptionTime\\skill_scores.tsv'

    save_targets_per_dataset(targets_tsv_path, feature_lh_files, feature_rh_files, output_target_files)
