#include <iostream>
#include <vector>

#define BLAS_LAPACK_OPENBLAS
#include <cblas.h>

#include "matrix.h"

linalg::Vector cg(linalg::Matrix A, linalg::Vector b,
                  linalg::Vector x, int max_iter,
                  float err) {
  int iteration = 0;
  float alpha, beta, delta, delta_new, delta_old;
  linalg::Vector residual, d, q;

  residual = b - (A * x);
  d = residual;

  delta_new = residual.dot(residual);
  delta = delta_new;

  while ((iteration < max_iter) && 
         (delta_new > err * err * delta)) {
    q = A * d;
    alpha = delta_new / d.dot(q);
    x = x + (d * alpha);

    residual = b - (A * x);

    delta_old = delta_new;
    delta_new = residual.dot(residual);
    beta = delta_new / delta_old;

    d = residual + (d * beta);
    ++iteration;
  }

  return x;
}

int main() {
  linalg::Matrix A(2, 2);
  A(0, 0) = 3;
  A(0, 1) = 2;
  A(1, 0) = 2;
  A(1, 1) = 6;

  std::vector<float> orig_b {2, -8};
  linalg::Vector b(2);
  b(0) = 2;
  b(1) = -8;

  linalg::Vector cg_x(2);
  cg_x(0) = 1;
  cg_x(1) = 1;
  linalg::Vector res = cg(A, b, cg_x, 100, 1e-05);
  for (size_t i = 0; i < res.size(); ++i) {
    std::cout << res(i) << " ";
  }
  std::cout << std::endl;
}