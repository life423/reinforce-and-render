#include "../include/physics.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace gpu_env {

// CUDA kernel for updating entity positions
__global__ void update_positions_kernel(
    float* entities_x,
    float* entities_y,
    const float* velocities_x,
    const float* velocities_y,
    int count,
    float screen_width,
    float screen_height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    // Update position
    entities_x[idx] += velocities_x[idx];
    entities_y[idx] += velocities_y[idx];
    
    // Wrap around screen boundaries
    if (entities_x[idx] < 0) {
        entities_x[idx] += screen_width;
    } else if (entities_x[idx] >= screen_width) {
        entities_x[idx] -= screen_width;
    }
    
    if (entities_y[idx] < 0) {
        entities_y[idx] += screen_height;
    } else if (entities_y[idx] >= screen_height) {
        entities_y[idx] -= screen_height;
    }
}

// Launcher function that calls the CUDA kernel for position updates
void cuda_update_positions(
    float* entities_x,
    float* entities_y,
    const float* velocities_x,
    const float* velocities_y,
    int count,
    float screen_width,
    float screen_height
) {
    // Calculate grid dimensions for CUDA
    const int blockSize = 256; // Good default for newer NVIDIA GPUs
    const int gridSize = (count + blockSize - 1) / blockSize;
    
    // Launch the CUDA kernel
    update_positions_kernel<<<gridSize, blockSize>>>(
        entities_x, entities_y,
        velocities_x, velocities_y,
        count, screen_width, screen_height
    );
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}

// Placeholder declarations for the remaining CUDA functions
// These will be implemented in future steps
// Currently, the CPU implementations in physics_cpu.cpp will be used as fallbacks

// CUDA kernel for collision detection between two entity sets
__global__ void detect_collisions_kernel(
    const float* entities_a_x,
    const float* entities_a_y,
    const float* entities_a_sizes,
    int entities_a_count,
    const float* entities_b_x,
    const float* entities_b_y,
    const float* entities_b_sizes,
    int entities_b_count,
    bool* collision_matrix
) {
    // Get the global thread index
    int a_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread should process an entity from set A
    if (a_idx >= entities_a_count) return;
    
    // Get the properties of entity A
    float a_x = entities_a_x[a_idx];
    float a_y = entities_a_y[a_idx];
    float a_size = entities_a_sizes[a_idx];
    
    // Check collision with all entities in set B
    for (int b_idx = 0; b_idx < entities_b_count; ++b_idx) {
        // Get properties of entity B
        float b_x = entities_b_x[b_idx];
        float b_y = entities_b_y[b_idx];
        float b_size = entities_b_sizes[b_idx];
        
        // Calculate distance between entities
        float dx = a_x - b_x;
        float dy = a_y - b_y;
        float distance_squared = dx * dx + dy * dy;
        
        // Calculate minimum distance for collision
        float min_distance = (a_size + b_size) * 0.5f;
        float min_distance_squared = min_distance * min_distance;
        
        // Check for collision and store result
        collision_matrix[a_idx * entities_b_count + b_idx] = 
            (distance_squared <= min_distance_squared);
    }
}

void cuda_detect_collisions(
    const float* entities_a_x,
    const float* entities_a_y,
    const float* entities_a_sizes,
    int entities_a_count,
    const float* entities_b_x,
    const float* entities_b_y,
    const float* entities_b_sizes,
    int entities_b_count,
    bool* collision_matrix
) {
    // Calculate grid dimensions for CUDA
    const int blockSize = 256;  // Threads per block
    const int gridSize = (entities_a_count + blockSize - 1) / blockSize;
    
    // Launch the CUDA kernel
    detect_collisions_kernel<<<gridSize, blockSize>>>(
        entities_a_x, entities_a_y, entities_a_sizes, entities_a_count,
        entities_b_x, entities_b_y, entities_b_sizes, entities_b_count,
        collision_matrix
    );
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}

// CUDA kernel for calculating distances between entity sets
__global__ void calculate_distances_kernel(
    const float* entities_a_x,
    const float* entities_a_y,
    int entities_a_count,
    const float* entities_b_x,
    const float* entities_b_y,
    int entities_b_count,
    float* distance_matrix
) {
    // Get the global thread index
    int a_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread should process an entity from set A
    if (a_idx >= entities_a_count) return;
    
    // Get the position of entity A
    float a_x = entities_a_x[a_idx];
    float a_y = entities_a_y[a_idx];
    
    // Calculate distance to all entities in set B
    for (int b_idx = 0; b_idx < entities_b_count; ++b_idx) {
        // Get position of entity B
        float b_x = entities_b_x[b_idx];
        float b_y = entities_b_y[b_idx];
        
        // Calculate Euclidean distance
        float dx = a_x - b_x;
        float dy = a_y - b_y;
        float distance = sqrtf(dx * dx + dy * dy);
        
        // Store in distance matrix
        distance_matrix[a_idx * entities_b_count + b_idx] = distance;
    }
}

void cuda_calculate_distances(
    const float* entities_a_x,
    const float* entities_a_y,
    int entities_a_count,
    const float* entities_b_x,
    const float* entities_b_y,
    int entities_b_count,
    float* distance_matrix
) {
    // Calculate grid dimensions for CUDA
    const int blockSize = 256;  // Threads per block
    const int gridSize = (entities_a_count + blockSize - 1) / blockSize;
    
    // Launch the CUDA kernel
    calculate_distances_kernel<<<gridSize, blockSize>>>(
        entities_a_x, entities_a_y, entities_a_count,
        entities_b_x, entities_b_y, entities_b_count,
        distance_matrix
    );
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}

// CUDA kernel for calculating danger map based on missile trajectories
__global__ void calculate_danger_map_kernel(
    const float* missiles_x,
    const float* missiles_y,
    const float* missiles_vx,
    const float* missiles_vy,
    int missile_count,
    const float* grid_positions_x,
    const float* grid_positions_y,
    int grid_width,
    int grid_height,
    float* danger_values,
    int prediction_steps,
    float danger_radius
) {
    // Calculate global thread indices
    int grid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if this thread should process a grid cell
    if (grid_x >= grid_width || grid_y >= grid_height) return;
    
    // Calculate grid index
    int grid_idx = grid_y * grid_width + grid_x;
    
    // Get grid cell position
    float grid_cell_x = grid_positions_x[grid_x];
    float grid_cell_y = grid_positions_y[grid_y];
    
    // Initialize danger value
    float total_danger = 0.0f;
    
    // Process each missile
    for (int m = 0; m < missile_count; ++m) {
        // Start with missile's current position
        float missile_x = missiles_x[m];
        float missile_y = missiles_y[m];
        float missile_vx = missiles_vx[m];
        float missile_vy = missiles_vy[m];
        
        // Track minimum distance to grid cell
        float min_distance_squared = FLT_MAX;
        
        // Predict missile trajectory
        for (int step = 0; step < prediction_steps; ++step) {
            // Move missile forward
            missile_x += missile_vx;
            missile_y += missile_vy;
            
            // Calculate distance to grid cell
            float dx = grid_cell_x - missile_x;
            float dy = grid_cell_y - missile_y;
            float distance_squared = dx * dx + dy * dy;
            
            // Update minimum distance
            min_distance_squared = min(min_distance_squared, distance_squared);
        }
        
        // Convert distance to danger value
        float min_distance = sqrtf(min_distance_squared);
        if (min_distance < danger_radius) {
            // Danger increases as distance decreases
            float danger = 1.0f - (min_distance / danger_radius);
            total_danger += danger;
        }
    }
    
    // Store the calculated danger value
    danger_values[grid_idx] = total_danger;
}

void cuda_calculate_danger_map(
    const float* missiles_x,
    const float* missiles_y,
    const float* missiles_vx,
    const float* missiles_vy,
    int missile_count,
    const float* grid_positions_x,
    const float* grid_positions_y,
    int grid_width,
    int grid_height,
    float* danger_values
) {
    // No missiles means no danger
    if (missile_count == 0) {
        cudaMemset(danger_values, 0, grid_width * grid_height * sizeof(float));
        return;
    }
    
    // Define prediction steps and danger radius
    const int prediction_steps = 20;
    const float danger_radius = 150.0f;
    
    // Define thread block size (16x16 is a common choice for 2D grids)
    dim3 blockSize(16, 16);
    
    // Calculate grid size to cover all cells
    dim3 gridSize(
        (grid_width + blockSize.x - 1) / blockSize.x,
        (grid_height + blockSize.y - 1) / blockSize.y
    );
    
    // Launch the CUDA kernel
    calculate_danger_map_kernel<<<gridSize, blockSize>>>(
        missiles_x, missiles_y, missiles_vx, missiles_vy,
        missile_count,
        grid_positions_x, grid_positions_y,
        grid_width, grid_height,
        danger_values,
        prediction_steps, danger_radius
    );
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}

// CUDA kernel for calculating missile threat levels
__global__ void calculate_missile_threats_kernel(
    float enemy_x,
    float enemy_y,
    const float* missiles_x,
    const float* missiles_y,
    const float* missiles_vx,
    const float* missiles_vy,
    int missile_count,
    int prediction_steps,
    float* threat_levels,
    float* threat_vectors_x,
    float* threat_vectors_y,
    float missile_danger_radius
) {
    // Get the global thread index (one thread per missile)
    int missile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread should process a missile
    if (missile_idx >= missile_count) return;
    
    // Get missile properties
    float missile_x = missiles_x[missile_idx];
    float missile_y = missiles_y[missile_idx];
    float missile_vx = missiles_vx[missile_idx];
    float missile_vy = missiles_vy[missile_idx];
    
    // Initialize variables for tracking closest approach
    float min_distance_squared = FLT_MAX;
    float closest_x = missile_x;
    float closest_y = missile_y;
    bool will_hit = false;
    
    // Predict missile trajectory and find closest approach
    for (int step = 0; step < prediction_steps; ++step) {
        // Move missile forward
        missile_x += missile_vx;
        missile_y += missile_vy;
        
        // Calculate distance to enemy
        float dx = missile_x - enemy_x;
        float dy = missile_y - enemy_y;
        float distance_squared = dx * dx + dy * dy;
        
        // Check if this is the closest position
        if (distance_squared < min_distance_squared) {
            min_distance_squared = distance_squared;
            closest_x = missile_x;
            closest_y = missile_y;
            
            // Check if missile will hit enemy
            if (distance_squared < (missile_danger_radius * 0.5f) * (missile_danger_radius * 0.5f)) {
                will_hit = true;
            }
        }
    }
    
    // Calculate threat level based on distance
    float min_distance = sqrtf(min_distance_squared);
    float threat_level = 0.0f;
    
    if (min_distance < missile_danger_radius) {
        // Threat increases as distance decreases
        threat_level = 1.0f - (min_distance / missile_danger_radius);
        
        // Apply higher weight if missile will hit
        if (will_hit) {
            threat_level *= 2.0f;
        }
        
        // Calculate evasion direction (away from closest predicted position)
        float dx = enemy_x - closest_x;
        float dy = enemy_y - closest_y;
        
        // Normalize direction
        float length = sqrtf(dx * dx + dy * dy);
        if (length > 0.0001f) {
            dx /= length;
            dy /= length;
        }
        
        // Store results
        threat_levels[missile_idx] = threat_level;
        threat_vectors_x[missile_idx] = dx;
        threat_vectors_y[missile_idx] = dy;
    } else {
        // No threat
        threat_levels[missile_idx] = 0.0f;
        threat_vectors_x[missile_idx] = 0.0f;
        threat_vectors_y[missile_idx] = 0.0f;
    }
}

// CUDA kernel for reducing threat vectors to a single evasion vector
__global__ void reduce_threat_vectors_kernel(
    const float* threat_levels,
    const float* threat_vectors_x,
    const float* threat_vectors_y,
    int missile_count,
    float* final_vector_x,
    float* final_vector_y
) {
    // Use shared memory for efficient reduction
    __shared__ float shared_vector_x[256];
    __shared__ float shared_vector_y[256];
    __shared__ float shared_weights[256];
    
    // Get thread index
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    shared_vector_x[tid] = 0.0f;
    shared_vector_y[tid] = 0.0f;
    shared_weights[tid] = 0.0f;
    
    // Load missile threat data into shared memory
    if (gid < missile_count) {
        float threat = threat_levels[gid];
        shared_vector_x[tid] = threat_vectors_x[gid] * threat;
        shared_vector_y[tid] = threat_vectors_y[gid] * threat;
        shared_weights[tid] = threat;
    }
    
    __syncthreads();
    
    // Perform parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_vector_x[tid] += shared_vector_x[tid + s];
            shared_vector_y[tid] += shared_vector_y[tid + s];
            shared_weights[tid] += shared_weights[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes the final result
    if (tid == 0) {
        if (shared_weights[0] > 0.0001f) {
            // Normalize by total weight
            atomicAdd(final_vector_x, shared_vector_x[0] / shared_weights[0]);
            atomicAdd(final_vector_y, shared_vector_y[0] / shared_weights[0]);
        }
    }
}

void cuda_calculate_evasion_vector(
    float enemy_x,
    float enemy_y,
    const float* missiles_x,
    const float* missiles_y,
    const float* missiles_vx,
    const float* missiles_vy,
    int missile_count,
    int prediction_steps,
    float* evasion_vector_x,
    float* evasion_vector_y
) {
    if (missile_count == 0) {
        *evasion_vector_x = 0.0f;
        *evasion_vector_y = 0.0f;
        return;
    }
    
    // Allocate device memory for threat data
    float* d_threat_levels;
    float* d_threat_vectors_x;
    float* d_threat_vectors_y;
    float* d_final_vector_x;
    float* d_final_vector_y;
    
    cudaMalloc(&d_threat_levels, missile_count * sizeof(float));
    cudaMalloc(&d_threat_vectors_x, missile_count * sizeof(float));
    cudaMalloc(&d_threat_vectors_y, missile_count * sizeof(float));
    cudaMalloc(&d_final_vector_x, sizeof(float));
    cudaMalloc(&d_final_vector_y, sizeof(float));
    
    // Initialize final vectors to zero
    cudaMemset(d_final_vector_x, 0, sizeof(float));
    cudaMemset(d_final_vector_y, 0, sizeof(float));
    
    // Calculate missile threats
    const float missile_danger_radius = 150.0f; // Should match the value in physics_cpu.cpp
    const int blockSize = 256;
    const int gridSize = (missile_count + blockSize - 1) / blockSize;
    
    calculate_missile_threats_kernel<<<gridSize, blockSize>>>(
        enemy_x, enemy_y,
        missiles_x, missiles_y, missiles_vx, missiles_vy,
        missile_count, prediction_steps,
        d_threat_levels, d_threat_vectors_x, d_threat_vectors_y,
        missile_danger_radius
    );
    
    // Reduce threats to a single evasion vector
    reduce_threat_vectors_kernel<<<1, blockSize>>>(
        d_threat_levels, d_threat_vectors_x, d_threat_vectors_y,
        missile_count,
        d_final_vector_x, d_final_vector_y
    );
    
    // Copy results back to host
    cudaMemcpy(evasion_vector_x, d_final_vector_x, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(evasion_vector_y, d_final_vector_y, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_threat_levels);
    cudaFree(d_threat_vectors_x);
    cudaFree(d_threat_vectors_y);
    cudaFree(d_final_vector_x);
    cudaFree(d_final_vector_y);
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}

} // namespace gpu_env
#endif // USE_CUDA
