#include <stdexcept>

#include "matrix.h"

#define BLAS_LAPACK_OPENBLAS
#include <cblas.h>

namespace linalg {

Vector::Vector() { size_ = 0; }

Vector::Vector(size_t size) : size_(size), data_(size) {};

Vector::Vector(std::vector<float> v) : data_(v) {
  size_ = v.size();
}

float& Vector::operator()(size_t i) {
  return data_[i];
}

float Vector::operator()(size_t i) const {
  return data_[i];
}

Vector Vector::operator+(const Vector& other) {
  if (other.size() != size_) {
    throw std::length_error("Vector sizes are not equal");
  }

  Vector result(size_);
  for (size_t i = 0; i < size_; ++i) {
    result(i) = data_[i] + other(i);
  }

  return result;
}

Vector Vector::operator-(const Vector& other) {
  if (other.size() != size_) {
    throw std::length_error("Vector sizes are not equal");
  }

  Vector result(size_);
  for (size_t i = 0; i < size_; ++i) {
    result(i) = data_[i] - other(i);
  }

  return result;
}

Vector Vector::operator*(const float& other) {
  Vector result(size_);
  for (size_t i = 0; i < size_; ++i) {
    result(i) = data_[i] * other;
  }
  return result;
}

float Vector::dot(const Vector& other) {
  if (other.size() != size_) {
    throw std::length_error("Vector sizes are not equal");
  }

  float result = 0.0;
  for (size_t i = 0; i < size_; ++i) {
    result += data_[i] * other(i);
  }
  return result;
}

Matrix::Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols),
  data_(rows * cols) {};

float& Matrix::operator()(size_t i, size_t j) {
  return data_[i * cols_ + j];
}

float Matrix::operator()(size_t i, size_t j) const {
  return data_[i * cols_ + j];
}

Vector Matrix::operator*(Vector other) {
  Vector result(other.size());

  cblas_sgemv(CblasRowMajor, CblasNoTrans, cols_, other.size(), 1,
              data_.data(), other.size(), other.data(), 1, 0,
              result.data(), 1);
  return result;
}

}  // namespace linalg