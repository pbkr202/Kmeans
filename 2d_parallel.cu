#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include<random>
#include<cuda_runtime.h>
#include<sm_60_atomic_functions.h>

using namespace std;

//DataStructure for representing 2D point
struct Point {
    double x;
    double y;
    int cluster;
};


// Function to calculate Euclidean distance between two points
__host__ __device__ double distance(Point p1, Point p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}


// CUDA kernel to assign each point to the nearest cluster
__global__ void assign_clusters_kernel(Point* points, int num_points, Point* centroids, int num_centroids) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_points) {
        double min_dist = 1e9;
        int min_idx = -1;
        for (int j = 0; j < num_centroids; ++j) {
            double d = distance(points[i], centroids[j]);
            if (d < min_dist) {
                min_dist = d;
                min_idx = j;
            }
        }
        points[i].cluster = min_idx;
    }
}

// CUDA kernel to update centroids
__global__ void update_centroids_kernel(Point* points, int num_points, Point* centroids, int num_centroids) {
    extern __shared__ Point sdata[];
    int tid = threadIdx.x;
    int cid = blockIdx.x * blockDim.x + threadIdx.x;

    if (cid < num_centroids) {
        sdata[tid].x = 0;
        sdata[tid].y = 0;
        sdata[tid].cluster = 0;
    }
    __syncthreads();

    for (int i = cid; i < num_points; i += gridDim.x * blockDim.x) {
        int cluster = points[i].cluster;
        atomicAdd(&sdata[cluster].x, points[i].x);
        atomicAdd(&sdata[cluster].y, points[i].y);
        atomicAdd(&sdata[cluster].cluster, 1);
    }

    __syncthreads();

    if (tid < num_centroids) {
        double x_sum = sdata[tid].x;
        double y_sum = sdata[tid].y;
        int count = sdata[tid].cluster;
        if (count > 0) {
            centroids[tid].x = x_sum / count;
            centroids[tid].y = y_sum / count;
        }
    }
}


// Function to perform k-means clustering using CUDA
void kmeans_cuda(vector<Point>& points, int k, int max_iter) {

    // initialize centroids with random points
    vector<Point> centroids(k);
    copy(points.begin(), points.begin() + k, centroids.begin());

    // copy points and centroids to device memory
    Point* d_points;
    Point* d_centroids;
    cudaMalloc(&d_points, points.size() * sizeof(Point));
    cudaMalloc(&d_centroids, k * sizeof(Point));
    cudaMemcpy(d_points, points.data(), points.size() * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids.data(), k * sizeof(Point), cudaMemcpyHostToDevice);

    // set block size and grid size for CUDA kernels
    const int block_size = 256;
    const int grid_size = (points.size() + block_size - 1) / block_size;
    
    // run k-means algorithm
    for (int iter = 0; iter < max_iter; ++iter) {
        assign_clusters_kernel<<<grid_size, block_size>>>(d_points, points.size(), d_centroids, k);
        /* cudaMemcpy(points.data(), d_points, points.size() * sizeof(Point), cudaMemcpyDeviceToHost);
             for (auto i:points)
               cout<<i.x<<" "<<i.y<<" "<<i.cluster<<endl; */
        cudaDeviceSynchronize();
        update_centroids_kernel<<<grid_size, block_size, k * sizeof(Point)>>>(d_points, points.size(), d_centroids, k);
        /* cudaMemcpy(centroids.data(), d_centroids, k * sizeof(Point), cudaMemcpyDeviceToHost);
            for (auto i:centroids)
              cout<<i.x<<" "<<i.y<<endl; */
        cudaDeviceSynchronize();
    }

    // copy updated centroids back to host memory
    cudaMemcpy(centroids.data(), d_centroids, k * sizeof(Point), cudaMemcpyDeviceToHost);

    /* for (auto i:centroids)
    cout<<i.x<<" "<<i.y<<endl;*/
    
    // assign each point to the nearest centroid
    for (int i = 0; i < points.size(); ++i) {
        double min_dist = INFINITY;
        int min_idx = -1;
        for (int j = 0; j < k; ++j) {
            double d = distance(points[i], centroids[j]);
            if (d < min_dist) {
                min_dist = d;
                min_idx = j;
            }
        }
        points[i].cluster = min_idx;
    }

    // free device memory
    cudaFree(d_points);
    cudaFree(d_centroids);
}
int main() {
    //Generate random points
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-10, 10);
    vector<Point> points(10000);
    for (auto& point : points) {
       point.x = dist(gen);
        point.y = dist(gen);
    }
    
    /* vector<Point> points = {
        {2.0, 3.0, -1},
        {4.0, 2.0, -1},
        {2.0, 1.0, -1},
        {8.0, 11.0, -1},
        {10.0, 12.0, -1},
        {10.0, 10.0, -1},
    }; */

    // perform k-means clustering with maximum iterations 10
    kmeans_cuda(points, 2, 10);

    // print the clusters
   /* for (int i = 0; i < points.size(); ++i) {
        cout << "Point " << i << " is in cluster " << points[i].cluster << endl;
    }*/

    return 0;
}
