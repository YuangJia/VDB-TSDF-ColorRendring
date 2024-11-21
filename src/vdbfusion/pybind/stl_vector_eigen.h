#pragma once
#include <Eigen/Core>
#include <openvdb/math/Vec3.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// Traits for vector-like classes
template <typename T>
struct VectorTraits;

// Specialization for Eigen::Matrix
template <typename Scalar, int Rows>
struct VectorTraits<Eigen::Matrix<Scalar, Rows, 1>> {
    using ScalarType = Scalar;
    static constexpr int RowsAtCompileTime = Rows;
};

// Specialization for openvdb::math::Vec3
template <typename Scalar>
struct VectorTraits<openvdb::math::Vec3<Scalar>> {
    using ScalarType = Scalar;
    static constexpr int RowsAtCompileTime = 3;
};

namespace py = pybind11;

namespace pybind11 {

// Generic conversion from NumPy array to std::vector
template <typename T>
std::vector<T> py_array_to_vectors(py::array_t<typename VectorTraits<T>::ScalarType> array) {
    constexpr int size_at_compile_time = VectorTraits<T>::RowsAtCompileTime;
    if (array.ndim() != 2 || array.shape(1) != size_at_compile_time) {
        throw py::cast_error("Input array must have shape (n, size_at_compile_time)");
    }
    std::vector<T> vectors(array.shape(0));
    auto array_unchecked = array.template unchecked<2>();
    for (auto i = 0; i < array_unchecked.shape(0); ++i) {
        T vec;
        for (int j = 0; j < size_at_compile_time; ++j) {
            vec[j] = static_cast<typename VectorTraits<T>::ScalarType>(array_unchecked(i, j));
        }
        vectors[i] = vec;
    }
    return vectors;
}

// Bind std::vector of Eigen::Matrix or openvdb::math::Vec3 to Python
template <typename VectorType,
          typename Vector = std::vector<VectorType>,
          typename holder_type = std::unique_ptr<Vector>,
          typename InitFunc>
py::class_<Vector, holder_type> pybind_vector(py::module& m,
                                              const std::string& bind_name,
                                              const std::string& repr_name,
                                              InitFunc init_func) {
    auto vec = py::class_<Vector, holder_type>(m, bind_name.c_str(), py::buffer_protocol());
    vec.def(py::init(init_func));

    // Handle buffer protocol for Eigen-like types
    if constexpr (VectorTraits<VectorType>::RowsAtCompileTime > 0) {
        constexpr size_t rows = VectorTraits<VectorType>::RowsAtCompileTime;
        using ScalarType = typename VectorTraits<VectorType>::ScalarType;
        vec.def_buffer([](Vector& v) -> py::buffer_info {
            return py::buffer_info(
                v.data(), sizeof(ScalarType),
                py::format_descriptor<ScalarType>::format(), 2,
                {v.size(), rows}, {sizeof(VectorType), sizeof(ScalarType)});
        });
    }

    // Add common methods
    vec.def("__repr__", [repr_name](const Vector& v) {
        return repr_name + " with " + std::to_string(v.size()) + " elements.";
    });
    vec.def("__copy__", [](Vector& v) { return Vector(v); });
    vec.def("__deepcopy__", [](Vector& v) { return Vector(v); });

    return vec;
}

}  // namespace pybind11
