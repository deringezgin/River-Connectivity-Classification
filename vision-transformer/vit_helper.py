"""
Derin Gezgin

This is the helper script which takes the specific parameters from the run-time arguments and passes it to the training script
"""

import sys, json
from vit_training_script import complete_run

if __name__ == "__main__":
    mlp_size = int(sys.argv[1])
    transformer_count = int(sys.argv[2])
    linear_count = int(sys.argv[3])
    class_weight = {int(k): v for k, v in json.loads(sys.argv[4]).items()}
    patch_size = int(sys.argv[5])
    attention_head_count = int(sys.argv[6])

    complete_run(mlp_size=mlp_size,
                 transformer_count=transformer_count,
                 linear_count=linear_count,
                 class_weight=class_weight,
                 patch_size=patch_size,
                 num_attention=attention_head_count)
