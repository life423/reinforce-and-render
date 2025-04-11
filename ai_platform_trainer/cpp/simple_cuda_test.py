"""
Simple CUDA test program to verify that the GPU can execute CUDA kernels.
This test uses PyTorch's CUDA capabilities to run a basic kernel.
"""
import torch
import time
import numpy as np

def print_gpu_info():
    """Print information about the GPU"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nDevice {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("WARNING: CUDA is not available. The test will run on CPU only.")

def simple_cuda_benchmark():
    """Run a simple matrix multiplication benchmark on both CPU and GPU"""
    print("\n=== Matrix Multiplication Benchmark ===")
    
    # Create large matrices for multiplication
    size = 2000
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    
    # CPU benchmark
    print("\nRunning on CPU...")
    start_time = time.time()
    cpu_result = torch.matmul(a, b)
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.4f} seconds")
    
    # GPU benchmark (if available)
    if torch.cuda.is_available():
        # Move matrices to GPU
        a_gpu = a.cuda()
        b_gpu = b.cuda()
        
        # Warmup
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        # Benchmark
        print("\nRunning on GPU...")
        start_time = time.time()
        gpu_result = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start_time
        print(f"GPU time: {gpu_time:.4f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"\nGPU speedup: {speedup:.2f}x faster than CPU")
        
        # Verify results match
        gpu_result_cpu = gpu_result.cpu()
        error = torch.abs(cpu_result - gpu_result_cpu).max().item()
        print(f"Maximum error between CPU and GPU results: {error:.6e}")
        
        if error < 1e-5:
            print("Results match! GPU computation is correct.")
        else:
            print("WARNING: Results don't match exactly. This could be due to floating-point precision differences.")

def test_cuda_memory_operations():
    """Test CUDA memory operations - similar to what our physics kernels will do"""
    print("\n=== CUDA Memory Operations Test ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return
    
    # Create arrays similar to our position updates
    num_entities = 100000
    
    # Create position and velocity arrays on CPU
    positions_x = torch.rand(num_entities)
    positions_y = torch.rand(num_entities)
    velocities_x = torch.rand(num_entities) * 0.1 - 0.05  # Random velocities between -0.05 and 0.05
    velocities_y = torch.rand(num_entities) * 0.1 - 0.05
    
    # Screen bounds for wrapping
    screen_width = 800.0
    screen_height = 600.0
    
    print(f"Testing with {num_entities} entities")
    
    # CPU implementation
    print("\nRunning position updates on CPU...")
    start_time = time.time()
    
    for _ in range(100):  # Run multiple iterations for benchmarking
        positions_x += velocities_x
        positions_y += velocities_y
        
        # Wrap around screen boundaries
        positions_x = torch.where(positions_x < 0, positions_x + screen_width, positions_x)
        positions_x = torch.where(positions_x >= screen_width, positions_x - screen_width, positions_x)
        positions_y = torch.where(positions_y < 0, positions_y + screen_height, positions_y)
        positions_y = torch.where(positions_y >= screen_height, positions_y - screen_height, positions_y)
    
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.4f} seconds")
    
    # GPU implementation
    print("\nRunning position updates on GPU...")
    
    # Move data to GPU
    gpu_positions_x = positions_x.cuda()
    gpu_positions_y = positions_y.cuda()
    gpu_velocities_x = velocities_x.cuda()
    gpu_velocities_y = velocities_y.cuda()
    
    # Warmup
    gpu_positions_x += gpu_velocities_x
    torch.cuda.synchronize()
    
    # Reset positions for fair comparison
    gpu_positions_x = positions_x.cuda()
    gpu_positions_y = positions_y.cuda()
    
    start_time = time.time()
    
    for _ in range(100):  # Same number of iterations as CPU
        gpu_positions_x += gpu_velocities_x
        gpu_positions_y += gpu_velocities_y
        
        # Wrap around screen boundaries
        gpu_positions_x = torch.where(gpu_positions_x < 0, gpu_positions_x + screen_width, gpu_positions_x)
        gpu_positions_x = torch.where(gpu_positions_x >= screen_width, gpu_positions_x - screen_width, gpu_positions_x)
        gpu_positions_y = torch.where(gpu_positions_y < 0, gpu_positions_y + screen_height, gpu_positions_y)
        gpu_positions_y = torch.where(gpu_positions_y >= screen_height, gpu_positions_y - screen_height, gpu_positions_y)
    
    torch.cuda.synchronize()  # Wait for GPU to finish
    gpu_time = time.time() - start_time
    print(f"GPU time: {gpu_time:.4f} seconds")
    
    # Calculate speedup
    speedup = cpu_time / gpu_time
    print(f"\nGPU speedup: {speedup:.2f}x faster than CPU")
    
    # Report GPU memory usage
    print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # Print results summary
    print("\n=== Results Summary ===")
    print(f"Operations per second on CPU: {100 * num_entities / cpu_time:.2e}")
    print(f"Operations per second on GPU: {100 * num_entities / gpu_time:.2e}")
    print(f"Performance improvement with GPU: {speedup:.2f}x")

if __name__ == "__main__":
    print("=== GPU Acceleration Test for NVIDIA 5070 ===\n")
    print_gpu_info()
    simple_cuda_benchmark()
    test_cuda_memory_operations()
    print("\nTest completed successfully!")
