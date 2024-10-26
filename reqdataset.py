import os
import random

# Paths
path_to_protocol = r'C:\Users\Serilda\Desktop\VPD\protocols\ASVspoof2021.LA.cm.train.trl.txt'  # Protocol file
path_to_features = r'C:\Users\Serilda\Desktop\VPD\feature_extracted'  # Folder with extracted features
output_protocol = r'C:\Users\Serilda\Desktop\VPD\protocols\subset_protocol.txt'  # New protocol for the 6000 samples

# Feature types you're checking (LFCC and MFCC)
feature_types = ['LFCC', 'MFCC']

# Step 1: Read protocol file
with open(path_to_protocol, 'r') as f:
    protocol_data = [line.strip().split() for line in f.readlines()]  # Load metadata

# Step 2: Select random 6000 samples
random.shuffle(protocol_data)  # Shuffle the protocol to randomize selection

selected_samples = []
for entry in protocol_data:
    filename = entry[1]  # Get the filename from protocol
    features_exist = all(
        os.path.exists(os.path.join(path_to_features, f'{filename}_{feature_type}.csv')) 
        for feature_type in feature_types
    )

    if features_exist:
        selected_samples.append(entry)

    if len(selected_samples) >= 6000:  # Stop when you have 6000 valid samples
        break

# Step 3: Write the selected 6000 samples to a new protocol file
with open(output_protocol, 'w') as f:
    for sample in selected_samples:
        f.write(' '.join(sample) + '\n')

print(f"Selected {len(selected_samples)} samples and saved to {output_protocol}")
