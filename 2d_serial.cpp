#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace std;

//DataStructure for representing 2D point
struct Point {
    double x, y;
    int cluster;
};

//Function to find euclidean distance between 2 points
double distance(Point a, Point b) {
    return sqrt(pow((a.x - b.x), 2) + pow((a.y - b.y), 2));
}

//Fuction to run the Kmeans Clustering Algorithm logic
void kmeans(vector<Point>& points, int k) {

    // Initialize centroids randomly
    vector<Point> centroids(k);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(0, points.size() - 1);
    for (int i = 0; i < k; i++) {
        centroids[i] = points[dist(gen)];
    }

    // Assign points to clusters
    bool assignment_changed = true;
    while (assignment_changed) {
        assignment_changed = false;
        for (auto& point : points) {
            double min_dist = INFINITY;
            int old_cluster = point.cluster;
            for (int i = 0; i < k; i++) {
                double d = distance(point, centroids[i]);
                if (d < min_dist) {
                    min_dist = d;
                    point.cluster = i;
                }
            }
            if (point.cluster != old_cluster) {
                assignment_changed = true;
            }
        }

        // Update centroids
        vector<int> cluster_sizes(k, 0);
        vector<double> x_sum(k, 0);
        vector<double> y_sum(k, 0);
        for (auto& point : points) {
            int cluster = point.cluster;
            cluster_sizes[cluster]++;
            x_sum[cluster] += point.x;
            y_sum[cluster] += point.y;
        }
        for (int i = 0; i < k; i++) {
            centroids[i].x = x_sum[i] / cluster_sizes[i];
            centroids[i].y = y_sum[i] / cluster_sizes[i];
        }
    }
}

int main() {
    //Generate random points
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-10, 10);
    vector<Point> points(100);
    for (auto& point : points) {
       point.x = dist(gen);
        point.y = dist(gen);
    }
    
  /*vector<Point> points = {
        {2.0, 3.0, 1},
        {4.0, 2.0, 1},
        {2.0, 1.0, 1},
        {10.0, 12.0, 1},
        {8.0, 11.0, 1},
        {10.0, 10.0, 1}
    };*/

    // Perform k-means clustering
    int k = 2;
    kmeans(points, k);

    //Print results
    /*for (int i = 0; i < k; i++) {
        cout << "Cluster " << i << ":" << endl;
        for (auto& point : points) {
            if (point.cluster == i) {
                cout << "(" << point.x << ", " << point.y << ")" << endl;
            }
        }
    }*/

    return 0;
}

