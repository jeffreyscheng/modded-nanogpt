#!/bin/bash

# Define the parameter ranges
STARTING_MINIBATCHES=(0 1000 2000 3000 4000 5000)
OPTIMIZERS=("Adam" "Muon")
SEEDS=(0 1 2 3 4 5 6 7)

# Iterate over the Cartesian product of parameters
for start_mb in "${STARTING_MINIBATCHES[@]}"; do
  for opt in "${OPTIMIZERS[@]}"; do
    for seed in "${SEEDS[@]}"; do

      # Format the step number with leading zeros (6 digits)
      formatted_step=$(printf "%06d" $start_mb)

      # Determine the output directory and base checkpoint path based on the optimizer and starting minibatch
      if [ "$opt" == "Muon" ]; then
        output_dir="muon_souping_checkpoints"
        checkpoint_dir="logs/gptm_record"
      else # Optimizer is Adam
        output_dir="adam_souping_checkpoints"
        checkpoint_dir="logs/gptm_adam"
      fi
      BASE_CHECKPOINT_PATH="${checkpoint_dir}/state_step${formatted_step}.pt"

      # Check if the required checkpoint file exists for this specific run
      if [ ! -f "$BASE_CHECKPOINT_PATH" ]; then
          echo "Error: Checkpoint file not found for optimizer $opt, step $start_mb at $BASE_CHECKPOINT_PATH"
          # Decide if you want to skip this run or exit
          continue # Skip this run if checkpoint is missing
          # exit 1 # Exit the script if a required checkpoint is missing
      fi

      # Create the output directory if it doesn't exist
      mkdir -p "$output_dir"

      echo "-----------------------------------------------------------------------"
      echo "Running training:"
      echo "  Starting Minibatch: $start_mb"
      echo "  Optimizer:          $opt"
      echo "  Seed:               $seed"
      echo "  Checkpoint Path:    $BASE_CHECKPOINT_PATH" # Corrected variable name
      echo "  Output Directory:   $output_dir"
      echo "-----------------------------------------------------------------------"

      # Construct and execute the torchrun command
      torchrun --standalone --nnodes=1 --nproc_per_node=8 train_gptm_from_checkpoint.py \
        --starting_checkpoint_path "$BASE_CHECKPOINT_PATH" \
        --starting_minibatch "$start_mb" \
        --minibatch_random_seed "$seed" \
        --optimizer "$opt" \
        --output_path_dir "$output_dir"

      # Check the exit status of the command
      if [ $? -ne 0 ]; then
        echo "Error during training run: start_mb=$start_mb, opt=$opt, seed=$seed"
        # Decide if you want to exit the script on error or continue
        # exit 1 # Uncomment to exit on first error
      fi

      echo "-----------------------------------------------------------------------"
      echo "Finished run: start_mb=$start_mb, opt=$opt, seed=$seed"
      echo "-----------------------------------------------------------------------"

    done
  done
done

echo "All training runs completed." 