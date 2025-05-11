import os
import sys
import time
import copy
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch._inductor.codecache # noqa: E402
import torch._inductor.graph # noqa: E402

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
torch._inductor.config.coordinate_descent_tuning = True # we allow this flag for medium track
torch._dynamo.config.compiled_autograd = True

# Import all the functions and classes from gpt_static.py
from gpt_static import (
    zeropower_via_newtonschulz5, update, norm, init_linear, next_multiple_of_n,
    _load_data_shard, distributed_data_generator, get_lr, get_window_size_blocks_helper,
    get_window_size_blocks, print0, opt_params, 
    Muon, Rotary, CausalSelfAttention, MLP, Block, GPT, Hyperparameters, run_validation,
    load_state_dict_safely
)

def extract_model_parameters(model):
    """Extract model parameters as a dictionary."""
    return {name: param.clone().detach() for name, param in model.named_parameters()}

def interpolate_models(models, weights):
    """Interpolate between multiple models using the given weights."""
    assert len(models) == len(weights), "Number of models must match number of weights"
    
    # Verify weights sum to approximately 1
    weight_sum = sum(weights)
    assert 0.99 <= weight_sum <= 1.01, f"Weights sum to {weight_sum}, not 1.0"
    
    # Create a new model with the same architecture as the first model
    interpolated_model = copy.deepcopy(models[0])
    
    # Extract parameters from each model
    model_params = [extract_model_parameters(model) for model in models]
    
    # Interpolate parameters
    with torch.no_grad():
        for name, param in interpolated_model.named_parameters():
            param.zero_()
            for i, weight in enumerate(weights):
                if weight > 0:
                    param.add_(model_params[i][name] * weight)
    
    return interpolated_model

def train_model(seed=0,
                run_id=0,
                iterations=None,
                checkpoint_path=None,
                output_path=None,
                batch_size_factor=1.0,
):
    # Set PyTorch random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -----------------------------------------------------------------------------
    # int main

    args = Hyperparameters()
    args.train_seq_len = int(args.train_seq_len * batch_size_factor)
    
    # Use custom iterations if provided
    if iterations is None:
        iterations = args.num_iterations

    # torchrun sets these env variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 8 # this code is designed for 8xH100
    assert torch.cuda.is_available()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    # Skip init_process_group as it's now handled by the caller
    # dist.init_process_group(backend="nccl", device_id=device)
    master_process = (rank == 0) # this process will do logging, checkpointing etc.

    ########################################
    #    Construct model and optimizer     #
    ########################################

    model: torch.nn.Module = GPT(vocab_size=args.vocab_size, num_layers=16, num_heads=8, model_dim=1024,
                        max_seq_len=max(args.train_seq_len, args.val_seq_len)).cuda()
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.bfloat16()
    # Sync model parameters across ranks after initial creation
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    # collect the parameters to optimize
    hidden_matrix_params = sorted((p for p in model.blocks.parameters() if p.ndim >= 2), key=lambda x: x.size(), reverse=True)
    embed_params = [*model.embed.parameters(), *model.value_embeds.parameters()]
    scalar_params = [model.scalars]
    head_params: list[torch.nn.Parameter] = [model.lm_head_w]
    # sanity check
    params_collections = [hidden_matrix_params, embed_params, scalar_params, head_params]
    optimized_parameters_set = {p for params in params_collections for p in params}
    assert optimized_parameters_set == {*model.parameters()}
    assert len(optimized_parameters_set) == sum(len(lst) for lst in params_collections)

    # init the optimizer(s)
    adam_param_groups = [dict(params=head_params, lr=1/320), dict(params=embed_params, lr=0.3), dict(params=scalar_params, lr=0.015)]
    # small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
    # discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
    optimizer1 = torch.optim.AdamW(adam_param_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, fused=True)
    optimizer2 = Muon(hidden_matrix_params, lr=0.025, momentum=0.95, rank=rank, world_size=world_size)
    optimizers: list[torch.optim.Optimizer] = [optimizer1, optimizer2]
    opt2params = {opt: opt_params(opt) for opt in optimizers}
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # DON'T compile the model yet - this was the issue!
    # model: torch.nn.Module = torch.compile(model, dynamic=False)

    # Load from checkpoint if provided
    start_step = 0
    loaded_from_checkpoint = False
    if checkpoint_path:
        assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
        loaded_from_checkpoint = True
        if master_process:
            print(f"Loading checkpoint from {checkpoint_path}")
            
        # Only rank 0 loads the checkpoint
        if rank == 0:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            assert 'step' in checkpoint, "Checkpoint must contain 'step' key"
            start_step = checkpoint['step']
            
            # Load state dictionary into model
            load_state_dict_safely(model, checkpoint['model'], strict=False)
            
            # Load optimizer states
            for i, opt in enumerate(optimizers):
                if i < len(checkpoint.get('optimizers', [])):
                    opt.load_state_dict(checkpoint['optimizers'][i])
        
        # Broadcast start_step to all ranks
        start_step_tensor = torch.tensor([start_step], dtype=torch.long, device='cuda')
        dist.broadcast(start_step_tensor, 0)
        start_step = int(start_step_tensor.item())
        
        # Broadcast model parameters to all ranks
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)
            
        # Ensure all parameters are in bfloat16
        for param in model.parameters():
            if param.is_floating_point() and param.dtype != torch.bfloat16:
                param.data = param.data.to(torch.bfloat16)
                
        # Verify no parameter is a zero matrix
        if start_step != 0:
            zero_params = [name for name, param in model.named_parameters() if param.abs().sum().item() == 0]
            assert not zero_params, f"Found {len(zero_params)} zero matrices after loading checkpoint: {zero_params[:5]}"
        
        if master_process:
            sample_norms = {name: param.norm().item() for name, param in list(model.named_parameters())[:5]}
            print(f"Sample parameter norms: {sample_norms}")
            print(f"Resuming from step {start_step}")
    
    end_step = start_step + iterations

    # NOW compile the model AFTER loading the checkpoint
    model: torch.nn.Module = torch.compile(model, dynamic=False)

    ########################################
    #            Warmup kernels            #
    ########################################

    # Skip warmup if loading from checkpoint
    if not loaded_from_checkpoint:
        # Warmup the training kernels, then re-initialize the state so we aren't cheating
        warmup_steps = 10
        initial_state = copy.deepcopy(dict(model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers]))
        # Use the seed for warmup data as well
        torch.manual_seed(seed + 5000)  # Different seed offset for warmup
        for _ in range(warmup_steps):
            inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
            model(inputs.to(torch.int32), targets, get_window_size_blocks(0)).backward()
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
        model.load_state_dict(initial_state["model"])
        for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
            opt.load_state_dict(opt_state)
        del initial_state
        # Ensure model parameters are in sync after warmup
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)
        # Reset the seed after warmup
        torch.manual_seed(seed)
    else:
        if master_process:
            print("Skipping warmup steps since loading from checkpoint")
        # Ensure optimizer states are properly initialized for the loaded model
        model.zero_grad(set_to_none=True)
        for opt in optimizers:
            if isinstance(opt, Muon):
                # Properly initialize Muon optimizer state for each parameter
                for group in opt.param_groups:
                    for p in group['params']:
                        # Make sure parameters are bfloat16
                        if p.dtype != torch.bfloat16:
                            p.data = p.data.to(torch.bfloat16)
                        
                        # Initialize state with proper fields
                        if p not in opt.state or 'mantissa' not in opt.state[p]:
                            opt.state[p] = {
                                "mantissa": torch.zeros_like(p, dtype=torch.uint16),
                                "momentum_buffer": torch.zeros_like(p, dtype=torch.float32)
                            }

    ########################################
    #        Training and validation       #
    ########################################

    torch.cuda.reset_peak_memory_stats()
    if master_process:
        print("Initializing data loader...")
    train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size, seed=seed)

    if master_process:
        print("Starting training loop...")
    
    training_time_ms = 0
    # start the clock
    t0 = time.perf_counter()
    # begin training
    train_steps = end_step
    for step in range(start_step, train_steps + 1):
        last_step = (step == train_steps)
        if last_step:
            # stop the clock
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_batch_size = world_size * args.val_seq_len
            assert args.val_tokens % val_batch_size == 0
            # Also pass the seed to validation data loader
            val_loss = run_validation(model, step, args, rank, world_size, seed)
            if master_process:
                print(f"step:{step}/{train_steps} val_loss:{val_loss:.6f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step - start_step, 1):.2f}ms")
            model.train()
            # start the clock again
            t0 = time.perf_counter()

            if master_process: 
                log = dict(step=step, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
                torch.save(log, output_path)
                print(f"Saved final model to {output_path}")
            # the last step only has the validation loop, so break to avoid training
            break

        # --------------- TRAINING SECTION -----------------
        if step == start_step and master_process:
            print("Fetching first batch...")
        inputs, targets = next(train_loader)
        if master_process: print(f"Rank {rank}: Got batch with shape {inputs.shape}")
        
        # Forward and backward pass
        loss = model(inputs, targets, get_window_size_blocks(step))
        if master_process: print(f"Rank {rank}: Forward pass complete, loss={loss.item()}")
        loss.backward()
        if master_process: print(f"Rank {rank}: Backward pass complete")
        
        # Correct handling of gradients - proper all-reduce on ALL parameters, even if grad is None
        if master_process: print(f"Rank {rank}: Setting up gradient reduction futures")
        
        # CRITICAL FIX: Make sure all parameters participate in all_reduce, even with None gradients
        # The previous code was building futures differently on different ranks
        for param in model.parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param)
                
        # Now collect futures with the guarantee that all parameters have gradients
        opt2futures = {
            opt: [dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future() for p in params]
            for opt, params in opt2params.items()
        }
        if master_process: print(f"Rank {rank}: Got futures: {[len(futures) for opt, futures in opt2futures.items()]}")
        
        # Set optimization hyperparameters
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step)
        for group in optimizer2.param_groups:
            frac = min(step / 300, 1) # momentum warmup for muon
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
        
        # Wait for all processes to reach this point to avoid race conditions
        dist.barrier()
        if master_process: print(f"Rank {rank}: Barrier complete before stepping optimizers")
        
        # Step the optimizers - EXACTLY as in train_gptm.py
        if master_process: print(f"Rank {rank}: Waiting for futures and stepping optimizers")
        for opt in optimizers:
            if master_process: print(f"Rank {rank}: Processing optimizer {opt.__class__.__name__}")
            torch.futures.collect_all(opt2futures[opt]).wait()
            if master_process: print(f"Rank {rank}: All futures complete for {opt.__class__.__name__}")
            opt.step()
            if master_process: print(f"Rank {rank}: Step complete for {opt.__class__.__name__}")
            
        # null the gradients
        model.zero_grad(set_to_none=True)
        if master_process: print(f"Rank {rank}: Zeroed gradients")
        
        # logging
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        if master_process:
            print(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step - start_step + 1):.2f}ms")

    if master_process:
        print(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
            f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")

if __name__ == "__main__":
    tick = time.perf_counter()
    # Initialize the process group once for all runs
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    
    try:
        record_names = ["muon_record"]
        starting_minibatch_steps = [0, 5, 10, 25, 125, 2000, 4000]
        forward_iteration_values = [1, 10, 100]
        batch_size_factors = [1.0, 0.5, 0.125, 0.0625]
        num_seeds = 2
        for record_name in record_names:
            for starting_minibatch_step in starting_minibatch_steps:
                for forward_iterations in forward_iteration_values:
                    for batch_size_factor in batch_size_factors:
                        checkpoint_path = f"logs/{record_name}/state_step{starting_minibatch_step:06d}.pt"
                        # get the name of the folder in the checkpoint path.  should be one level above the file name
                        # i only want the folder name, not the full path
                        folder_name = os.path.basename(os.path.dirname(checkpoint_path))
                        
                        # Create the parent directory for output files if it doesn't exist
                        output_parent_dir = f"souping_logs/{folder_name}_step{starting_minibatch_step:06d}"
                        os.makedirs(output_parent_dir, exist_ok=True)
                        
                        for seed in range(num_seeds):
                            output_path = f"{output_parent_dir}/seed{seed}_forward{forward_iterations}.pt"                       
                            if not os.path.exists(output_path):
                                print(f"Starting seed {seed}")                
                                train_model(seed=seed, run_id=0, iterations=forward_iterations, 
                                            checkpoint_path=checkpoint_path, output_path=output_path)
                            # synchronize
                            dist.barrier()
            
        # Only destroy the process group once, at the very end
        dist.destroy_process_group()
    except Exception as e:
        # Clean up even if there's an error
        if dist.is_initialized():
            dist.destroy_process_group()
        raise e
    finally:
        if rank == 0:
            print(f"Total time taken: {time.perf_counter() - tick:.2f} seconds")