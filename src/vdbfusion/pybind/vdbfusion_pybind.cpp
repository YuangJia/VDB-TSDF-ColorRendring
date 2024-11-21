#include <openvdb/openvdb.h>

// pybind11
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>


// std stuff
#include <Eigen/Core>
#include <memory>
#include <vector>

#ifdef PYOPENVDB_SUPPORT
#include "pyopenvdb.h"
#endif

#include "stl_vector_eigen.h"
#include "vdbfusion/VDBVolume.h"

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3i>);
PYBIND11_MAKE_OPAQUE(std::vector<openvdb::math::Vec3<int>>);

using namespace py::literals;
namespace py = pybind11;

namespace vdbfusion {

PYBIND11_MODULE(vdbfusion_pybind, m) {
    // Bind std::vector<Eigen::Vector3d>
    auto vector3dvector = pybind11::pybind_vector<Eigen::Vector3d>(
        m, "_VectorEigen3d", "std::vector<Eigen::Vector3d>",
        pybind11::py_array_to_vectors<Eigen::Vector3d>);

    auto vector3ivector = pybind11::pybind_vector<Eigen::Vector3i>(
        m, "_VectorEigen3i", "std::vector<Eigen::Vector3i>",
        pybind11::py_array_to_vectors<Eigen::Vector3i>);

    auto vector_vec3i = pybind11::pybind_vector<openvdb::math::Vec3<int>>(
        m, "_VectorOpenVDB3i", "std::vector<openvdb::math::Vec3<int>>",
        pybind11::py_array_to_vectors<openvdb::math::Vec3<int>>);

    py::class_<VDBVolume, std::shared_ptr<VDBVolume>> vdb_volume(
        m, "_VDBVolume",
        "This is the low-level C++ bindings. "
        "Methods and constructors starting with a `_` "
        "should not be used directly. Refer to the Python Processor class.");

    vdb_volume
        .def(py::init<float, float, bool>(), "voxel_size"_a, "sdf_trunc"_a, "space_carving"_a = false)
        .def("_integrate",
             py::overload_cast<const std::vector<Eigen::Vector3d>&, const std::vector<openvdb::math::Vec3<int>>&,
                               const Eigen::Vector3d&, const std::function<float(float)>&>(
                 &VDBVolume::Integrate),
             "points"_a, "colors"_a, "origin"_a, "weighting_function"_a)
        .def("_integrate",
             py::overload_cast<const std::vector<Eigen::Vector3d>&, const std::vector<openvdb::math::Vec3<int>>&,
                               const Eigen::Matrix4d&, const std::function<float(float)>&>(
                 &VDBVolume::Integrate),
             "points"_a, "colors"_a, "extrinsic"_a, "weighting_function"_a)
        .def(
            "_integrate",
            [](VDBVolume& self, const std::vector<Eigen::Vector3d>& points,
               const std::vector<openvdb::math::Vec3<int>>& colors, const Eigen::Vector3d& origin) {
                self.Integrate(points, colors, origin, [](float /*sdf*/) { return 1.0f; });
            },
            "points"_a, "colors"_a, "origin"_a)
        .def(
            "_integrate",
            [](VDBVolume& self, const std::vector<Eigen::Vector3d>& points,
               const std::vector<openvdb::math::Vec3<int>>& colors, const Eigen::Matrix4d& extrinsics) {
                self.Integrate(points, colors, extrinsics, [](float /*sdf*/) { return 1.0f; });
            },
            "points"_a, "colors"_a, "extrinsic"_a)
#ifdef PYOPENVDB_SUPPORT
        .def("_integrate",
             py::overload_cast<openvdb::FloatGrid::Ptr, const std::function<float(float)>&>(
                 &VDBVolume::Integrate),
             "grid"_a, "weighting_function"_a)
        .def(
            "_integrate",
            [](VDBVolume& self, openvdb::FloatGrid::Ptr grid) {
                self.Integrate(grid, [](float /*sdf*/) { return 1.0f; });
            },
            "grid"_a)
        .def(
            "_integrate",
            [](VDBVolume& self, openvdb::FloatGrid::Ptr grid, float weight) {
                self.Integrate(grid, [=](float /*sdf*/) { return weight; });
            },
            "grid"_a, "weight"_a)
#endif
        .def("_update_tsdf",
             [](VDBVolume& self, const float& sdf, std::vector<int>& ijk,
                const std::function<float(float)>& weighting_function) {
                 self.UpdateTSDF(sdf, openvdb::Coord(ijk[0], ijk[1], ijk[2]), weighting_function);
             },
             "sdf"_a, "weighting_function"_a, "ijk"_a)
        .def(
            "_update_tsdf",
            [](VDBVolume& self, const float& sdf, std::vector<int>& ijk) {
                self.UpdateTSDF(sdf, openvdb::Coord(ijk[0], ijk[1], ijk[2]),
                                [](float /*sdf*/) { return 1.0f; });
            },
            "sdf"_a, "ijk"_a)
        .def("_extract_triangle_mesh", &VDBVolume::ExtractTriangleMesh, "fill_holes"_a, "min_weight"_a)
        .def("_extract_vdb_grids",
             [](const VDBVolume& self, const std::string& filename) {
                 openvdb::io::File(filename).write({self.tsdf_, self.weights_});
             },
             "filename"_a)
#ifndef PYOPENVDB_SUPPORT
        .def_property_readonly_static("PYOPENVDB_SUPPORT_ENABLED", [](py::object) { return false; })
#else
        .def_property_readonly_static("PYOPENVDB_SUPPORT_ENABLED", [](py::object) { return true; })
        .def("_prune", &VDBVolume::Prune, "min_weight"_a)
        .def_readwrite("_colors",&VDBVolume::colors_)
        .def_readwrite("_tsdf", &VDBVolume::tsdf_)
        .def_readwrite("_weights", &VDBVolume::weights_)
#endif
        .def_readwrite("_voxel_size", &VDBVolume::voxel_size_)
        .def_readwrite("_sdf_trunc", &VDBVolume::sdf_trunc_)
        .def_readwrite("_space_carving", &VDBVolume::space_carving_);
}

}  // namespace vdbfusion
