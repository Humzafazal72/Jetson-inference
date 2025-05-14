#include <cnpy.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

using namespace Eigen;
using namespace std;

const double EPSILON = 1e-8;
const double CLIP_MIN = -1.0;
const double CLIP_MAX = 1.0;

// Function to calculate Pearson correlation coefficient between two vectors
double pearsonCorrelation(const VectorXd& x, const VectorXd& y) {
    int n = x.size();

    double mean_x = x.mean();
    double mean_y = y.mean();

    VectorXd x_centered = x.array() - mean_x;
    VectorXd y_centered = y.array() - mean_y;

    double std_x = sqrt(x_centered.squaredNorm() / (n - 1));
    double std_y = sqrt(y_centered.squaredNorm() / (n - 1));

    // Add EPSILON to avoid division by zero
    std_x = std::max(std_x, EPSILON);
    std_y = std::max(std_y, EPSILON);

    double corr = x_centered.dot(y_centered) / ((n - 1) * std_x * std_y);

    // Clip correlation between -1 and 1
    corr = std::min(std::max(corr, CLIP_MIN), CLIP_MAX);

    // Remove NaNs (e.g., if x or y is constant)
    if (std::isnan(corr)) corr = 0.0;

    return corr;
}

int main() {
    // Load NPZ file
    cnpy::npz_t npz = cnpy::npz_load("input.npz");
    cnpy::NpyArray arr = npz["data"];

    // Convert to Eigen matrix
    size_t rows = arr.shape[0];
    size_t cols = arr.shape[1];
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> data(arr.data<double>(), rows, cols);

    // Initialize correlation matrix
    MatrixXd correlation = MatrixXd::Zero(rows, rows);

    // Compute Pearson correlation
    for (int i = 0; i < rows; i++) {
        for (int j = i; j < rows; j++) {
            double corr = pearsonCorrelation(data.row(i), data.row(j));
            correlation(i, j) = corr;
            correlation(j, i) = corr; // symmetric
        }

        if (i % 5 == 0) {
            cout << "Processed " << i << " of " << rows << " rows" << endl;
        }
    }

    // Set diagonal to zero
    for (int i = 0; i < rows; i++) {
        correlation(i, i) = 0.0;
    }

    // Save correlation matrix
    vector<size_t> shape = {(size_t)rows, (size_t)rows};
    cnpy::npz_save("correlation.npz", "correlation", correlation.data(), shape, "w");

    cout << "Correlation matrix saved to correlation.npz with shape [" << rows << " x " << rows << "]" << endl;

    return 0;
}

