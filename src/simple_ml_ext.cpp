#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

inline float mat_index(float *X, size_t i, size_t j, size_t m, size_t n)
{
    assert(i < m && j < n);
    return X[i * n + j];
}

inline void mat_index_write(float *X, size_t val, size_t i, size_t j, size_t m, size_t n)
{
    assert(i < m && j < n);
    X[i * n + j] = val;
}

/**
 * Perform matrix mulipication on lhs and rhs
 * @param lhs: (m, n)
 * @param rhs: (n, k)
 * @param X: (m, k)
 */
void matmul(float *X,
            float *lhs,
            float *rhs,
            size_t m,
            size_t n,
            size_t k)
{

    for (size_t row = 0; row < m; row++)
    {
        for (size_t col = 0; col < k; col++)
        {
            float temp = 0;
            for (size_t itm = 0; itm < n; itm++)
            {
                temp += mat_index(lhs, row, itm, m, n) * mat_index(rhs, itm, col, n, k);
            }
            mat_index_write(X, temp, row, col, m, k);
        }
    }
}

void dump_grad(float *grad, int id, size_t len){
    std::string s =  "grad" + std::to_string(id);
    FILE* handle = fopen(s.c_str(), "wb");
    for(size_t i = 0; i < len; i++){
        fwrite(&grad[i], sizeof(float), 1, handle);
    }
    fclose(handle);
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const u
     * nsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t num_samples = m;
    float *minibatch = new float[batch * n];
    unsigned char *minibatch_label = new unsigned char[batch];
    float *h = new float[batch * k];
    float *h_exp = new float[batch * k];
    float *h_sum = new float[batch * k];
    float *Z = new float[batch * k];
    float *labels = new float[batch * k];
    float *grad = new float[n * k];
    float *minibatch_T = new float[n * batch];
    for (size_t idx = 0; idx < num_samples; idx += batch)
    {

        std::copy(X + idx * n, X + (idx + batch) * n, minibatch);
#ifdef DEBUG
        std::cout << "minibatch: " << std::endl;
        for (size_t i = 0; i < batch; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                std::cout << mat_index(minibatch, i, j, batch, n) << " ";
            }
            std::cout << std::endl;
        }
        if (idx == 10)
        {
            FILE *handle = fopen("minibatch", "wb");
            for (size_t i = 0; i < batch * n; i++)
            {
                fwrite(&minibatch[i], sizeof(float), 1, handle);
            }
            fclose(handle);
        }
#endif
        std::copy(y + idx, y + idx + batch, minibatch_label);
#ifdef DEBUG
        std::cout << "minibatch_label: " << std::endl;
        for (size_t i = 0; i < batch; i++)
        {
            std::cout << (int)minibatch_label[i] << " ";
        }
        std::cout << std::endl;
#endif
        for(int i = 0; i < batch; i++){
            for(int j = 0; j < k; j++){
                float sum = 0;
                for(int k_ = 0; k_ < n; k_++){
                    sum += minibatch[i * n + k_] * theta[k_ * k + j];
                }
                h[i * k + j] = sum;
            }
        }
#ifdef DEBUG
        std::cout << "h: " << std::endl;
        for (size_t i = 0; i < batch; i++)
        {
            for (size_t j = 0; j < k; j++)
            {
                std::cout << mat_index(h, i, j, batch, k) << " ";
            }
            std::cout << std::endl;
        }
#endif
        for (int i = 0; i < batch * k; i++)
        {
            h_exp[i] = std::exp(h[i]);
        }
#ifdef DEBUG
        std::cout << "h_exp: " << std::endl;
        for (size_t i = 0; i < batch; i++)
        {
            for (size_t j = 0; j < k; j++)
            {
                std::cout << mat_index(h_exp, i, j, batch, k) << " ";
            }
            std::cout << std::endl;
        }
#endif
        // 不管normalized了
        for (size_t i = 0; i < batch; i++)
        {
            float sum = 0;
            for (size_t j = 0; j < k; j++)
            {
                sum += h_exp[i * k + j];
            }
            std::fill_n(h_sum + i * k, k, sum);
        }
#ifdef DEBUG
        std::cout << "h_sum: " << std::endl;
        for (size_t i = 0; i < batch; i++)
        {
            for (size_t j = 0; j < k; j++)
            {
                std::cout << mat_index(h_sum, i, j, batch, k) << " ";
            }
            std::cout << std::endl;
        }
#endif
        // Z
        for (size_t i = 0; i < batch * k; i++)
        {
            Z[i] = h_exp[i] / h_sum[i];
        }
#ifdef DEBUG
        std::cout << "Z: " << std::endl;
        for (size_t i = 0; i < batch; i++)
        {
            for (size_t j = 0; j < k; j++)
            {
                std::cout << mat_index(Z, i, j, batch, k) << " ";
            }
            std::cout << std::endl;
        }
#endif
        std::fill_n(labels, batch * k, 0);
        for (size_t i = 0; i < batch; i++)
        {
            labels[i * k + minibatch_label[i]] = 1;
        }
#ifdef DEBUG
        std::cout << "labels: " << std::endl;
        for (size_t i = 0; i < batch; i++)
        {
            for (size_t j = 0; j < k; j++)
            {
                std::cout << mat_index(labels, i, j, batch, k) << " ";
            }
            std::cout << std::endl;
        }
#endif
        for (size_t i = 0; i < batch; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                minibatch_T[j * batch + i] = minibatch[i * n + j];
            }
        }
#ifdef DEBUG
        std::cout << "minibatch_T: " << std::endl;
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < batch; j++)
            {
                std::cout << mat_index(minibatch_T, i, j, n, batch) << " ";
            }
            std::cout << std::endl;
        }
#endif
        for (size_t i = 0; i < batch * k; i++)
        {
            Z[i] = Z[i] - labels[i];
        }
#ifdef DEBUG
        std::cout << "Z: " << std::endl;
        for (size_t i = 0; i < batch; i++)
        {
            for (size_t j = 0; j < k; j++)
            {
                std::cout << mat_index(Z, i, j, batch, k) << " ";
            }
            std::cout << std::endl;
        }
#endif
        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < k; j++){
                float sum = 0;
                for(size_t k_ = 0; k_ < batch; k_++){
                    sum += minibatch_T[i * batch + k_] * Z[k_ * k + j];
                }
                grad[i * k + j] = sum;
            }
        }
#ifdef DEBUG
        std::cout << "grad: " << std::endl;
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < k; j++)
            {
                std::cout << mat_index(grad, i, j, n, k) << " ";
            }
            std::cout << std::endl;
        }
        dump_grad(grad, idx, n * k);
#endif
        for (size_t i = 0; i < n * k; i++)
        {
            theta[i] -= lr * (1.0 / float(batch)) * grad[i];
        }
#ifdef DEBUG
        std::cout << "theta: " << std::endl;
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < k; j++)
            {
                std::cout << mat_index(theta, i, j, n, k) << " ";
            }
            std::cout << std::endl;
        }
#endif
        std::memset(minibatch, 0, batch * n * sizeof(float)); 
        std::memset(minibatch_label, 0, batch * sizeof(unsigned char));
        std::memset(h, 0, batch * k * sizeof(float));
        std::memset(h_exp, 0, batch * k * sizeof(float));
        std::memset(h_sum, 0, batch * k * sizeof(float));
        std::memset(Z, 0, batch * k * sizeof(float));
        std::memset(labels, 0, batch * k * sizeof(float));
        std::memset(grad, 0, n * k * sizeof(float));
        std::memset(minibatch_T, 0, n * batch * sizeof(float));
    }
    /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m)
{
    m.def(
        "softmax_regression_epoch_cpp",
        [](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch)
        {
            softmax_regression_epoch_cpp(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch);
        },
        py::arg("X"), py::arg("y"), py::arg("theta"),
        py::arg("lr"), py::arg("batch"));
}
