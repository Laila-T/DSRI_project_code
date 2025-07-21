import os
import numpy as np
import xml.etree.ElementTree as ET
from scipy.ndimage import zoom
from classifiers.inception import Classifier_INCEPTION
import pandas as pd

def load_cvc_sample(filename, target_time_steps=250):
    with open(filename, 'r') as f:
        content = "<root>" + f.read() + "</root>"
    root = ET.fromstring(content)

    left, right = [], []
    for log in root.iter('log'):
        t = log.get('transform')
        if not t: continue
        vals = list(map(float, t.strip().split()[:12]))
        if log.get('DeviceName') == 'LeftHandToReference':
            left.append(vals)
        elif log.get('DeviceName') == 'RightHandToReference':
            right.append(vals)

    lh = np.array(left) 
    rh = np.array(right)

    def resample(arr):
        orig = arr.shape[0]
        if orig == target_time_steps:
            return arr
        factor = target_time_steps / orig
        return zoom(arr, zoom=[factor, 1], order=1)

    #Assign the resampled arrays to variables
    lh_rs = resample(lh)
    rh_rs = resample(rh)

    return lh_rs, rh_rs


def load_all_cvc(directory, target_time_steps=250):
    X_lh, X_rh = [], []
    for file_name in sorted(os.listdir(directory)):
        if not file_name.endswith('.xml'): continue
        lh, rh = load_cvc_sample(os.path.join(directory, file_name), target_time_steps)
        X_lh.append(lh) 
        X_rh.append(rh)
    return (
        np.stack(X_lh, axis=0),  #(N, 250, T)
        np.stack(X_rh, axis=0)   #(N, 250, T)
    )



def extract_two_feature_sets(cvc_dir, weights_path, output_dir):
    X_lh, X_rh = load_all_cvc(cvc_dir, target_time_steps=250)
    print("LH  shape:", X_lh.shape)
    print("RH  shape:", X_rh.shape)
    
    input_shape = (250, 12)
    nb_classes = 6
    classifier = Classifier_INCEPTION(
        output_directory=output_dir,
        input_shape=input_shape,
        nb_classes=nb_classes,
        verbose=False,
        build=True)
    print("Model expects  shape:", classifier.model.input_shape)
    #load weights from pretrained model
    classifier.model.load_weights(weights_path)

    feat_ext = classifier.build_feature_extractor(input_shape=input_shape)

    feats_lh = feat_ext.predict(X_lh, batch_size=32)
    feats_rh = feat_ext.predict(X_rh, batch_size=32)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'features_LH.npy'), feats_lh)
    np.save(os.path.join(output_dir, 'features_RH.npy'), feats_rh)


    print('Features extracted from:', cvc_dir,'\nWere saved to:', output_dir)


if __name__ == "__main__":
    cvc_directory_2015 = 'C:\\Users\\Laila\\InceptionTime\\DSRI 2025\\2015-TrackingData'
    pretrained_weights = 'C:\\Users\\Laila\\InceptionTime\\Results_Laila\\results\\inception\\TSC_itr_4MotionSenseHAR\\best_model.keras'
    output_directory = 'C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2015_data'
    extract_two_feature_sets(cvc_directory_2015, pretrained_weights, output_directory)

    cvc_directory_2016_Session1 = 'C:\\Users\\Laila\\InceptionTime\\DSRI 2025\\2016-TrackingData\\Session1'
    output_directorySession1 = 'C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2016_Session1'
    extract_two_feature_sets(cvc_directory_2016_Session1, pretrained_weights, output_directorySession1)

    cvc_directory_2016_Session2 = 'C:\\Users\\Laila\\InceptionTime\\DSRI 2025\\2016-TrackingData\\Session2'
    output_directorySession2 = 'C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2016_Session2'
    extract_two_feature_sets(cvc_directory_2016_Session2, pretrained_weights, output_directorySession2)

    cvc_directory_2016_Session3 = 'C:\\Users\\Laila\\InceptionTime\\DSRI 2025\\2016-TrackingData\\Session3'
    output_directorySession3 = 'C:\\Users\\Laila\\InceptionTime\\cvc_extracted_features\\2016_Session3'
    extract_two_feature_sets(cvc_directory_2016_Session3, pretrained_weights, output_directorySession3)
    
    
    
