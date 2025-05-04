import sys, json
from resnet_training_script import complete_run

if __name__ == "__main__":
    initial_filters = int(sys.argv[1])
    blocks_per_stage = json.loads(sys.argv[2])
    linear_layer_number = int(sys.argv[3])
    
    class_weight_json = json.loads(sys.argv[4])
    class_weight = {int(k): float(v) for k, v in class_weight_json.items()}

    if len(sys.argv) > 5:
        batch_size = int(sys.argv[5])
    else:
        batch_size = 40  # default if not provided

    # Call the complete_run function with the parsed arguments
    complete_run(initial_filters=initial_filters,
                 blocks_per_stage=blocks_per_stage,
                 linear_layer_number=linear_layer_number,
                 batch_size=batch_size,
                 class_weight=class_weight)
