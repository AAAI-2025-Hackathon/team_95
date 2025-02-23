import tensorflow as tf

# Function to inspect the features in a TFRecord file
def inspect_tfrecord(file_path):
    # Create a TFRecordDataset
    raw_dataset = tf.data.TFRecordDataset(file_path)

    # Iterate through the first few records to inspect features
    for raw_record in raw_dataset.take(1):  # Inspect only the first record
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print("Features in the TFRecord file:")
        for feature_name, feature in example.features.feature.items():
            print(f"Feature: {feature_name}")
            # print(f"  Data type: {feature.WhichOneof('kind')}")
            # if feature.HasField('float_list'):
            #     print(f"  Values: {feature.float_list.value}")
            # elif feature.HasField('int64_list'):
            #     print(f"  Values: {feature.int64_list.value}")
            # elif feature.HasField('bytes_list'):
            #     print(f"  Values: {feature.bytes_list.value}")
            # elif feature.HasField('float_list'):
            #     print(f"  Values: {feature.float_list.value}")
            # else:
            #     print("  Values: Unknown type")

# Main function
def main():
    file_path = '/s/lovelace/h/nobackup/sangmi/hackathon/AAAI-2025/data/Next_Day_Wildfire_Spread/2/next_day_wildfire_spread_eval_00.tfrecord'  # Replace with your TFRecord file path
    inspect_tfrecord(file_path)

if __name__ == "__main__":
    main()