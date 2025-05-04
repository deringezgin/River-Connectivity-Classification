"""
Derin Gezgin

This is the main running script. It allows the user to run tests to test different parameter combinations. The configurable parameters are:
    - Multi-Layer Perceptron (MLP) size
    - Transformer count
    - Linear layer count
    - Class weights
    - Patch size
    - Number of attention heads for each transformer block
The default values already provide a testing opportunity, but you can also change them while running the script.
"""

import argparse, itertools, subprocess, json, sys, time


def parse_args():
    """Function to parse the command line arguments"""
    parser = argparse.ArgumentParser(description='Run training script with various parameters.')
    parser.add_argument('--mlp_size', type=str, default='1024,2048', help='Comma-separated list of mlp_sizes, e.g., 1024,2048 (default: "1024,2048")')
    parser.add_argument('--transformer_count', type=str, default='1,2', help='Comma-separated list of transformer_counts, e.g., 1,2 (default: "1,2")')
    parser.add_argument('--linear_count', type=str, default='3,4,5', help='Comma-separated list of linear_counts, e.g., 3,4,5 (default: "3,4,5")')
    parser.add_argument('--class_weights', type=str, default='[{"1":1.0,"2":1.0},{"1":1.2,"2":1.0}]', help='JSON array of class_weights, e.g., \'[{"1":1.0,"2":1.0},{"1":1.2,"2":1.0}]\' (default: \'[{"1":1.0,"2":1.0},{"1":1.2,"2":1.0}]\')')
    parser.add_argument('--patch_size', type=str, default='10, 20', help='Comma-separated list of patch_size values eg. 10, 20, 30 (default: "10, 15")')
    parser.add_argument("--num_attention_heads", type=str, default='20, 24, 30', help='Comma-separated list of num_attention_heads values eg. 20, 40, 60 (default: "20, 24, 30")')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()  # Take the arguments from the command line

    # Split the arguments to the specific categories and store them in a list
    mlp_sizes = [int(s.strip()) for s in args.mlp_size.split(',')]
    transformer_counts = [int(s.strip()) for s in args.transformer_count.split(',')]
    linear_counts = [int(s.strip()) for s in args.linear_count.split(',')]
    patch_sizes = [int(s.strip()) for s in args.patch_size.split(',')]
    attention_heads = [int(s.strip()) for s in args.num_attention_heads.split(',')]
    class_weights_list = json.loads(args.class_weights)

    # Generate all parameter combinations
    combinations = itertools.product(mlp_sizes, transformer_counts, linear_counts, class_weights_list, patch_sizes, attention_heads)

    counter = 1

    # For each parameter combination
    for mlp_size, transformer_count, linear_count, class_weight, patch_size, attention_head_count in combinations:
        print("\n========================================\n")
        print(f"STARTING RUN #{counter}")
        print("\n========================================\n")

        class_weight_str = json.dumps(class_weight)

        # This is the terminal command for calling the specific combination
        cmd = [
            sys.executable, 'vit_helper.py',
            str(mlp_size),
            str(transformer_count),
            str(linear_count),
            class_weight_str,
            str(patch_size),
            str(attention_head_count)
        ]

        subprocess.run(cmd)  # Run the command using subprocess

        time.sleep(3)  # Sleep 3 seconds to ensure everything is done

        counter += 1


if __name__ == '__main__':
    main()
