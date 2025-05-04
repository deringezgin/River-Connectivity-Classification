import subprocess
import json
import sys
import time

def main():

    architectures = [
        ("ResNet101", 64, [3,4,23,3], 3, 40),
        ("ResNet50", 64, [3,4,6,3], 3, 40),
        ("ResNet34", 64, [3,4,6,3], 3, 40),
        ("ResNet18", 64, [2,2,2,2], 3, 40)
    ]

    class_weights_list = [
        {1:1.0, 2:1.0},
    ]

    counter = 1

    for arch_name, initial_filters, blocks_per_stage, linear_layer_number, batch_size in architectures:
        for class_weight in class_weights_list:
            print("\n========================================\n")
            print(f"STARTING RUN #{counter} for {arch_name}")
            print("\n========================================\n")

            # Convert class_weight dictionary to JSON string
            class_weight_str = json.dumps(class_weight)
            blocks_per_stage_str = json.dumps(blocks_per_stage)  # Convert list to JSON
            
            cmd = [
                sys.executable, 'resnet_helper.py',
                str(initial_filters),
                blocks_per_stage_str,
                str(linear_layer_number),
                class_weight_str,
                str(batch_size)
            ]

            # Run the command
            subprocess.run(cmd, check=True)

            # Sleep for a bit between runs if desired
            time.sleep(3)
            counter += 1

if __name__ == '__main__':
    main()

