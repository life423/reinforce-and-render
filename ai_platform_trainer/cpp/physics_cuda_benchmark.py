"""
Physics CUDA Benchmark

This script benchmarks the kinds of physics operations we've implemented in
our CUDA kernels, comparing CPU vs GPU performance for the computational
bottlenecks in our simulation.
"""
import time
from typing import Tuple

import matplotlib.pyplot as plt
import torch


def benchmark_position_updates(num_entities: int, iterations: int = 100) -> Tuple[float, float]:
    """
    Benchmark position update operations
    
    Args:
        num_entities: Number of entities to simulate
        iterations: Number of update iterations
        
    Returns:
        Tuple of (cpu_time, gpu_time) in seconds
    """
    print(f"\n=== Position Updates Benchmark ({num_entities:,} entities) ===")
    
    # Create position and velocity arrays on CPU
    positions_x = torch.rand(num_entities)
    positions_y = torch.rand(num_entities)
    velocities_x = torch.rand(num_entities) * 0.1 - 0.05  # Random velocities between -0.05 and 0.05
    velocities_y = torch.rand(num_entities) * 0.1 - 0.05
    
    # Screen bounds for wrapping
    screen_width = 800.0
    screen_height = 600.0
    
    # CPU implementation
    print("Running position updates on CPU...")
    start_time = time.time()
    
    for _ in range(iterations):
        positions_x += velocities_x
        positions_y += velocities_y
        
        # Wrap around screen boundaries
        positions_x = torch.where(positions_x < 0, positions_x + screen_width, positions_x)
        positions_x = torch.where(
            positions_x >= screen_width, positions_x - screen_width, positions_x)
        positions_y = torch.where(positions_y < 0, positions_y + screen_height, positions_y)
        positions_y = torch.where(
            positions_y >= screen_height, positions_y - screen_height, positions_y)
    
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.4f} seconds")
    
    # GPU implementation (if available)
    gpu_time = float('inf')
    
    if torch.cuda.is_available():
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
        
        print("Running position updates on GPU...")
        start_time = time.time()
        
        for _ in range(iterations):
            gpu_positions_x += gpu_velocities_x
            gpu_positions_y += gpu_velocities_y
            
            # Wrap around screen boundaries
            gpu_positions_x = torch.where(
                gpu_positions_x < 0, gpu_positions_x + screen_width, gpu_positions_x)
            gpu_positions_x = torch.where(
                gpu_positions_x >= screen_width, gpu_positions_x - screen_width, gpu_positions_x)
            gpu_positions_y = torch.where(
                gpu_positions_y < 0, gpu_positions_y + screen_height, gpu_positions_y)
            gpu_positions_y = torch.where(
                gpu_positions_y >= screen_height, gpu_positions_y - screen_height, gpu_positions_y)
        
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start_time
        print(f"GPU time: {gpu_time:.4f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"GPU speedup: {speedup:.2f}x faster than CPU")
    
    return cpu_time, gpu_time


def benchmark_collision_detection(num_entities_a: int, num_entities_b: int) -> Tuple[float, float]:
    """
    Benchmark collision detection operations
    
    Args:
        num_entities_a: Number of entities in first set
        num_entities_b: Number of entities in second set
        
    Returns:
        Tuple of (cpu_time, gpu_time) in seconds
    """
    benchmark_title = f"Collision Detection Benchmark ({num_entities_a:,} x {num_entities_b:,} entities)"
    print(f"\n=== {benchmark_title} ===")
    
    # Create entity data on CPU
    entities_a_x = torch.rand(num_entities_a) * 800.0
    entities_a_y = torch.rand(num_entities_a) * 600.0
    entities_a_sizes = torch.rand(num_entities_a) * 30.0 + 20.0  # Sizes between 20 and 50
    
    entities_b_x = torch.rand(num_entities_b) * 800.0
    entities_b_y = torch.rand(num_entities_b) * 600.0
    entities_b_sizes = torch.rand(num_entities_b) * 10.0 + 5.0  # Sizes between 5 and 15
    
    # CPU implementation
    print("Running collision detection on CPU...")
    start_time = time.time()
    
    # Initialize collision matrix
    cpu_collision_matrix = torch.zeros((num_entities_a, num_entities_b), dtype=torch.bool)
    
    # Detect collisions (O(n²) operation)
    for a in range(num_entities_a):
        a_x = entities_a_x[a]
        a_y = entities_a_y[a]
        a_size = entities_a_sizes[a]
        
        for b in range(num_entities_b):
            b_x = entities_b_x[b]
            b_y = entities_b_y[b]
            b_size = entities_b_sizes[b]
            
            # Calculate distance between entities
            dx = a_x - b_x
            dy = a_y - b_y
            distance_squared = dx * dx + dy * dy
            
            # Calculate minimum distance for collision
            min_distance = (a_size + b_size) * 0.5
            min_distance_squared = min_distance * min_distance
            
            # Check for collision
            cpu_collision_matrix[a, b] = (distance_squared <= min_distance_squared)
    
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.4f} seconds")
    
    # GPU implementation (if available)
    gpu_time = float('inf')
    
    if torch.cuda.is_available():
        # Move data to GPU
        gpu_entities_a_x = entities_a_x.cuda()
        gpu_entities_a_y = entities_a_y.cuda()
        gpu_entities_a_sizes = entities_a_sizes.cuda()
        
        gpu_entities_b_x = entities_b_x.cuda()
        gpu_entities_b_y = entities_b_y.cuda()
        gpu_entities_b_sizes = entities_b_sizes.cuda()
        
        # Initialize GPU collision matrix
        gpu_collision_matrix = torch.zeros(
            (num_entities_a, num_entities_b), dtype=torch.bool, device='cuda')
        
        # Warmup
        _ = torch.cdist(
            torch.stack([gpu_entities_a_x, gpu_entities_a_y], dim=1),
            torch.stack([gpu_entities_b_x, gpu_entities_b_y], dim=1)
        )
        torch.cuda.synchronize()
        
        print("Running collision detection on GPU...")
        start_time = time.time()
        
        # Calculate all pairwise distances in one go
        distances = torch.cdist(
            torch.stack([gpu_entities_a_x, gpu_entities_a_y], dim=1),
            torch.stack([gpu_entities_b_x, gpu_entities_b_y], dim=1)
        )
        
        # Calculate minimum distances for collision
        min_distances = (
            gpu_entities_a_sizes.unsqueeze(1) + gpu_entities_b_sizes.unsqueeze(0)) * 0.5
        
        # Check for collisions
        gpu_collision_matrix = distances <= min_distances
        
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start_time
        print(f"GPU time: {gpu_time:.4f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"GPU speedup: {speedup:.2f}x faster than CPU")
        
        # Verify results match
        cpu_collision_count = cpu_collision_matrix.sum().item()
        gpu_collision_count = gpu_collision_matrix.cpu().sum().item()
        print(f"CPU collisions detected: {cpu_collision_count}")
        print(f"GPU collisions detected: {gpu_collision_count}")
        
        if cpu_collision_count == gpu_collision_count:
            print("Results match! (OK)")
        else:
            print("WARNING: Results don't match exactly.")
    
    return cpu_time, gpu_time


def benchmark_evasion_vector(num_missiles: int, prediction_steps: int = 30) -> Tuple[float, float]:
    """
    Benchmark evasion vector calculation
    
    This is the most computationally intensive operation in our physics engine.
    
    Args:
        num_missiles: Number of missiles to process
        prediction_steps: Number of steps to predict missile trajectory
        
    Returns:
        Tuple of (cpu_time, gpu_time) in seconds
    """
    benchmark_title = f"Evasion Vector Benchmark ({num_missiles:,} missiles, {prediction_steps} steps)"
    print(f"\n=== {benchmark_title} ===")
    
    # Create missile data on CPU
    enemy_x = 400.0
    enemy_y = 300.0
    missiles_x = torch.rand(num_missiles) * 800.0
    missiles_y = torch.rand(num_missiles) * 600.0
    
    # Random velocities pointing roughly toward the enemy
    dx = enemy_x - missiles_x
    dy = enemy_y - missiles_y
    distances = torch.sqrt(dx * dx + dy * dy)
    
    # Normalize and add some randomness
    dx = dx / distances + torch.rand(num_missiles) * 0.5 - 0.25
    dy = dy / distances + torch.rand(num_missiles) * 0.5 - 0.25
    
    # Scale to missile speed
    missile_speed = 5.0
    missiles_vx = dx * missile_speed
    missiles_vy = dy * missile_speed
    
    # Set danger radius
    missile_danger_radius = 150.0
    
    # CPU implementation
    print("Calculating evasion vector on CPU...")
    start_time = time.time()
    
    # Track closest approach for each missile
    min_distances_squared = torch.full((num_missiles,), float('inf'))
    closest_x = missiles_x.clone()
    closest_y = missiles_y.clone()
    will_hit = torch.zeros(num_missiles, dtype=torch.bool)
    
    # Predict trajectories
    for step in range(prediction_steps):
        # Move missiles forward
        missiles_x = missiles_x + missiles_vx
        missiles_y = missiles_y + missiles_vy
        
        # Calculate distances to enemy
        dx = missiles_x - enemy_x
        dy = missiles_y - enemy_y
        distances_squared = dx * dx + dy * dy
        
        # Update closest positions
        closer_indices = distances_squared < min_distances_squared
        min_distances_squared[closer_indices] = distances_squared[closer_indices]
        closest_x[closer_indices] = missiles_x[closer_indices]
        closest_y[closer_indices] = missiles_y[closer_indices]
        
        # Check if missiles will hit enemy
        hit_threshold = (missile_danger_radius * 0.5) ** 2
        new_hits = (distances_squared < hit_threshold) & ~will_hit
        will_hit = will_hit | new_hits
    
    # Calculate threat levels
    min_distances = torch.sqrt(min_distances_squared)
    threat_levels = torch.zeros_like(min_distances)
    
    # Threats only from missiles within danger radius
    dangerous_indices = min_distances < missile_danger_radius
    threat_levels[dangerous_indices] = 1.0 - (
        min_distances[dangerous_indices] / missile_danger_radius)
    
    # Apply higher weight for missiles that will hit
    threat_levels[will_hit] *= 2.0
    
    # Calculate evasion directions
    dx = enemy_x - closest_x
    dy = enemy_y - closest_y
    
    # Normalize directions
    distances = torch.sqrt(dx * dx + dy * dy)
    valid_distances = distances > 0.0001
    
    normalized_dx = torch.zeros_like(dx)
    normalized_dy = torch.zeros_like(dy)
    
    normalized_dx[valid_distances] = dx[valid_distances] / distances[valid_distances]
    normalized_dy[valid_distances] = dy[valid_distances] / distances[valid_distances]
    
    # Apply threat levels as weights
    weighted_dx = normalized_dx * threat_levels
    weighted_dy = normalized_dy * threat_levels
    
    # Sum up to get final evasion vector
    total_threat = threat_levels.sum()
    
    if total_threat > 0.0001:
        evasion_x = weighted_dx.sum() / total_threat
        evasion_y = weighted_dy.sum() / total_threat
    else:
        evasion_x = 0.0
        evasion_y = 0.0
    
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"CPU evasion vector: ({evasion_x:.4f}, {evasion_y:.4f})")
    
    # GPU implementation (if available)
    gpu_time = float('inf')
    
    if torch.cuda.is_available():
        # Reset missiles to original positions
        gpu_enemy_x = torch.tensor(enemy_x, device='cuda')
        gpu_enemy_y = torch.tensor(enemy_y, device='cuda')
        gpu_missiles_x = missiles_x.cuda()
        gpu_missiles_y = missiles_y.cuda()
        gpu_missiles_vx = missiles_vx.cuda()
        gpu_missiles_vy = missiles_vy.cuda()
        
        # Warmup
        _ = torch.sqrt(torch.rand(100, device='cuda'))
        torch.cuda.synchronize()
        
        print("Calculating evasion vector on GPU...")
        start_time = time.time()
        
        # Track closest approach for each missile
        gpu_min_distances_squared = torch.full((num_missiles,), float('inf'), device='cuda')
        gpu_closest_x = gpu_missiles_x.clone()
        gpu_closest_y = gpu_missiles_y.clone()
        gpu_will_hit = torch.zeros(num_missiles, dtype=torch.bool, device='cuda')
        
        # Predict trajectories
        for step in range(prediction_steps):
            # Move missiles forward
            gpu_missiles_x = gpu_missiles_x + gpu_missiles_vx
            gpu_missiles_y = gpu_missiles_y + gpu_missiles_vy
            
            # Calculate distances to enemy
            dx = gpu_missiles_x - gpu_enemy_x
            dy = gpu_missiles_y - gpu_enemy_y
            distances_squared = dx * dx + dy * dy
            
            # Update closest positions
            closer_indices = distances_squared < gpu_min_distances_squared
            gpu_min_distances_squared[closer_indices] = distances_squared[closer_indices]
            gpu_closest_x[closer_indices] = gpu_missiles_x[closer_indices]
            gpu_closest_y[closer_indices] = gpu_missiles_y[closer_indices]
            
            # Check if missiles will hit enemy
            hit_threshold = (missile_danger_radius * 0.5) ** 2
            new_hits = (distances_squared < hit_threshold) & ~gpu_will_hit
            gpu_will_hit = gpu_will_hit | new_hits
        
        # Calculate threat levels
        gpu_min_distances = torch.sqrt(gpu_min_distances_squared)
        gpu_threat_levels = torch.zeros_like(gpu_min_distances)
        
        # Threats only from missiles within danger radius
        dangerous_indices = gpu_min_distances < missile_danger_radius
        gpu_threat_levels[dangerous_indices] = 1.0 - (
            gpu_min_distances[dangerous_indices] / missile_danger_radius)
        
        # Apply higher weight for missiles that will hit
        gpu_threat_levels[gpu_will_hit] *= 2.0
        
        # Calculate evasion directions
        dx = gpu_enemy_x - gpu_closest_x
        dy = gpu_enemy_y - gpu_closest_y
        
        # Normalize directions
        distances = torch.sqrt(dx * dx + dy * dy)
        valid_distances = distances > 0.0001
        
        gpu_normalized_dx = torch.zeros_like(dx)
        gpu_normalized_dy = torch.zeros_like(dy)
        
        gpu_normalized_dx[valid_distances] = dx[valid_distances] / distances[valid_distances]
        gpu_normalized_dy[valid_distances] = dy[valid_distances] / distances[valid_distances]
        
        # Apply threat levels as weights
        gpu_weighted_dx = gpu_normalized_dx * gpu_threat_levels
        gpu_weighted_dy = gpu_normalized_dy * gpu_threat_levels
        
        # Sum up to get final evasion vector
        gpu_total_threat = gpu_threat_levels.sum()
        
        if gpu_total_threat > 0.0001:
            gpu_evasion_x = gpu_weighted_dx.sum() / gpu_total_threat
            gpu_evasion_y = gpu_weighted_dy.sum() / gpu_total_threat
        else:
            gpu_evasion_x = 0.0
            gpu_evasion_y = 0.0
        
        torch.cuda.synchronize()  # Wait for GPU to finish
        gpu_time = time.time() - start_time
        print(f"GPU time: {gpu_time:.4f} seconds")
        print(f"GPU evasion vector: ({gpu_evasion_x.item():.4f}, {gpu_evasion_y.item():.4f})")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time
        print(f"GPU speedup: {speedup:.2f}x faster than CPU")
    
    return cpu_time, gpu_time


def run_scaling_benchmark():
    """
    Run benchmarks with increasing entity counts to measure scaling
    """
    print("\n=== Scaling Benchmark ===")
    
    # Entity counts to test
    entity_counts = [100, 1000, 10000, 100000, 1000000]
    
    # Storage for results
    position_cpu_times = []
    position_gpu_times = []
    
    collision_cpu_times = []
    collision_gpu_times = []
    
    evasion_cpu_times = []
    evasion_gpu_times = []
    
    # Run position update benchmarks
    for count in entity_counts:
        cpu_time, gpu_time = benchmark_position_updates(count)
        position_cpu_times.append(cpu_time)
        position_gpu_times.append(gpu_time)
    
    # Run collision detection benchmarks (use square root of counts to keep matrix size manageable)
    collision_counts = [10, 32, 100, 316, 1000]
    for count in collision_counts:
        cpu_time, gpu_time = benchmark_collision_detection(count, count)
        collision_cpu_times.append(cpu_time)
        collision_gpu_times.append(gpu_time)
    
    # Run evasion vector benchmarks
    for count in entity_counts[:4]:  # Skip the largest count for evasion
        cpu_time, gpu_time = benchmark_evasion_vector(count)
        evasion_cpu_times.append(cpu_time)
        evasion_gpu_times.append(gpu_time)
    
    # Plot results
    plt.figure(figsize=(15, 15))
    
    # Position updates plot
    plt.subplot(3, 1, 1)
    plt.loglog(entity_counts, position_cpu_times, 'b-o', label='CPU')
    plt.loglog(entity_counts, position_gpu_times, 'r-o', label='GPU')
    plt.title('Position Updates Performance')
    plt.xlabel('Number of Entities')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.legend()
    
    # Collision detection plot
    plt.subplot(3, 1, 2)
    plt.loglog([c*c for c in collision_counts], collision_cpu_times, 'b-o', label='CPU')
    plt.loglog([c*c for c in collision_counts], collision_gpu_times, 'r-o', label='GPU')
    plt.title('Collision Detection Performance')
    plt.xlabel('Number of Entity Pairs')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.legend()
    
    # Evasion vector plot
    plt.subplot(3, 1, 3)
    plt.loglog(entity_counts[:4], evasion_cpu_times, 'b-o', label='CPU')
    plt.loglog(entity_counts[:4], evasion_gpu_times, 'r-o', label='GPU')
    plt.title('Evasion Vector Calculation Performance')
    plt.xlabel('Number of Missiles')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('physics_benchmark_results.png')
    print("\nBenchmark results saved to 'physics_benchmark_results.png'")


if __name__ == "__main__":
    print("=== Physics Operations CUDA Benchmark ===\n")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Run a quick benchmark with moderate entity counts
    benchmark_position_updates(10000)
    benchmark_collision_detection(100, 100)
    benchmark_evasion_vector(1000)
    
    # Run scaling benchmark
    run_scaling_benchmark()
