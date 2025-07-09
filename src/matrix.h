#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "stdlib.h"
#include <vector>

namespace linalg {

class Vector {
  public:
    Vector();
    Vector(size_t size);
    Vector(std::vector<float> v);

    float& operator()(size_t i);
    float operator()(size_t i) const;

    Vector operator+(const Vector& other);
    Vector operator-(const Vector& other);
    Vector operator*(const float& other);

    float dot(const Vector& other);

    size_t size() const { return size_; }
    float* data() { return data_.data(); }

  private:
    size_t size_;
    std::vector<float> data_;
};

class Matrix {
  public:
    Matrix(size_t rows, size_t cols);
    
    float& operator()(size_t i, size_t j);
    float operator()(size_t i, size_t j) const;

    Vector operator*(Vector other);

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    float* data() { return data_.data(); }

  private:
    size_t rows_;
    size_t cols_;
    std::vector<float> data_;
};

}  // namespace linalg

#endif  // _MATRIX_H_
