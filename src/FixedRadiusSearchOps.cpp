// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <vector>

#include "Dtype.h"
#include "NeighborSearchCommon.h"
#include "TorchHelper.h"
#include "Helper.h"
#include "torch/script.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic conversion of std::string

namespace py = pybind11;

using namespace open3d::core::nns;

#ifdef BUILD_CUDA_MODULE
template <class T, class TIndex>
void FixedRadiusSearchCUDA(const torch::Tensor &points,
                           const torch::Tensor &queries,
                           double radius,
                           const torch::Tensor &points_row_splits,
                           const torch::Tensor &queries_row_splits,
                           const torch::Tensor &hash_table_splits,
                           const torch::Tensor &hash_table_index,
                           const torch::Tensor &hash_table_cell_splits,
                           const Metric metric,
                           const bool ignore_query_point,
                           const bool return_distances,
                           torch::Tensor &neighbors_index,
                           torch::Tensor &neighbors_row_splits,
                           torch::Tensor &neighbors_distance);
#endif

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> FixedRadiusSearch(
    torch::Tensor points,
    torch::Tensor queries,
    double radius,
    torch::Tensor points_row_splits,
    torch::Tensor queries_row_splits,
    torch::Tensor hash_table_splits,
    torch::Tensor hash_table_index,
    torch::Tensor hash_table_cell_splits,
    torch::ScalarType index_dtype,
    const std::string &metric_str,
    const bool ignore_query_point,
    const bool return_distances)
{
    Metric metric = L2;
    if (metric_str == "L1")
    {
        metric = L1;
    }
    else if (metric_str == "L2")
    {
        metric = L2;
    }
    else if (metric_str == "Linf")
    {
        metric = Linf;
    }
    else
    {
        TORCH_CHECK(false, "metric must be one of (L1, L2, Linf) but got " +
                               metric_str);
    }
    CHECK_TYPE(points_row_splits, kInt64);
    CHECK_TYPE(queries_row_splits, kInt64);
    CHECK_TYPE(hash_table_splits, kInt32);
    CHECK_TYPE(hash_table_index, kInt32);
    CHECK_TYPE(hash_table_cell_splits, kInt32);
    CHECK_SAME_DTYPE(points, queries);
    CHECK_SAME_DEVICE_TYPE(points, queries);
    TORCH_CHECK(index_dtype == torch::kInt32 || index_dtype == torch::kInt64,
                "index_dtype must be int32 or int64");
    // ensure that these are on the cpu
    points_row_splits = points_row_splits.to(torch::kCPU);
    queries_row_splits = queries_row_splits.to(torch::kCPU);
    hash_table_splits = hash_table_splits.to(torch::kCPU);
    points = points.contiguous();
    queries = queries.contiguous();
    points_row_splits = points_row_splits.contiguous();
    queries_row_splits = queries_row_splits.contiguous();
    hash_table_splits = hash_table_splits.contiguous();
    hash_table_index = hash_table_index.contiguous();
    hash_table_cell_splits = hash_table_cell_splits.contiguous();

    // check input shapes
    using namespace open3d::ml::op_util;
    Dim num_points("num_points");
    Dim num_queries("num_queries");
    Dim batch_size("batch_size");
    Dim num_cells("num_cells");
    CHECK_SHAPE(points, num_points, 3);
    CHECK_SHAPE(hash_table_index, num_points);
    CHECK_SHAPE(queries, num_queries, 3);
    CHECK_SHAPE(points_row_splits, batch_size + 1);
    CHECK_SHAPE(queries_row_splits, batch_size + 1);
    CHECK_SHAPE(hash_table_splits, batch_size + 1);
    CHECK_SHAPE(hash_table_cell_splits, num_cells + 1);

    const auto &point_type = points.dtype();

    auto device = points.device().type();
    auto device_idx = points.device().index();

    torch::Tensor neighbors_index;
    torch::Tensor neighbors_row_splits = torch::empty(
        {queries.size(0) + 1},
        torch::dtype(ToTorchDtype<int64_t>()).device(device, device_idx));
    torch::Tensor neighbors_distance;

#define FRS_FN_PARAMETERS                                              \
    points, queries, radius, points_row_splits, queries_row_splits,    \
        hash_table_splits, hash_table_index, hash_table_cell_splits,   \
        metric, ignore_query_point, return_distances, neighbors_index, \
        neighbors_row_splits, neighbors_distance

    if (points.is_cuda())
    {
#ifdef BUILD_CUDA_MODULE
        // pass to cuda function
        if (CompareTorchDtype<float>(point_type))
        {
            if (index_dtype == torch::kInt32)
            {
                FixedRadiusSearchCUDA<float, int32_t>(FRS_FN_PARAMETERS);
            }
            else
            {
                FixedRadiusSearchCUDA<float, int64_t>(FRS_FN_PARAMETERS);
            }
            return std::make_tuple(neighbors_index, neighbors_row_splits,
                                   neighbors_distance);
        }
#else
        TORCH_CHECK(false,
                    "FixedRadiusSearch was not compiled with CUDA support")
#endif
    }
    TORCH_CHECK(false, "FixedRadiusSearch does not support " +
                           points.toString() + " as input for points")
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>();
}

#ifdef BUILD_CUDA_MODULE
template <class T>
void BuildSpatialHashTableCUDA(const torch::Tensor &points,
                               double radius,
                               const torch::Tensor &points_row_splits,
                               const std::vector<uint32_t> &hash_table_splits,
                               torch::Tensor &hash_table_index,
                               torch::Tensor &hash_table_cell_splits);
#endif

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BuildSpatialHashTable(
    torch::Tensor points,
    double radius,
    torch::Tensor points_row_splits,
    double hash_table_size_factor,
    int64_t max_hash_table_size)
{
    // ensure that these tensors are on the cpu
    points_row_splits = points_row_splits.to(torch::kCPU);
    points = points.contiguous();
    points_row_splits = points_row_splits.contiguous();
    CHECK_TYPE(points_row_splits, kInt64);

    // check input shapes
    using namespace open3d::ml::op_util;
    Dim num_points("num_points");
    Dim batch_size("batch_size");

    CHECK_SHAPE(points, num_points, 3);
    CHECK_SHAPE(points_row_splits, batch_size + 1);

    const auto &point_type = points.dtype();

    std::vector<uint32_t> hash_table_splits(batch_size.value() + 1, 0);
    for (int i = 0; i < batch_size.value(); ++i)
    {
        int64_t num_points_i = points_row_splits.data_ptr<int64_t>()[i + 1] -
                               points_row_splits.data_ptr<int64_t>()[i];

        int64_t hash_table_size = std::min<int64_t>(
            std::max<int64_t>(hash_table_size_factor * num_points_i, 1),
            max_hash_table_size);
        hash_table_splits[i + 1] = hash_table_splits[i] + hash_table_size;
    }

    auto device = points.device().type();
    auto device_idx = points.device().index();

    torch::Tensor hash_table_index = torch::empty(
        {points.size(0)},
        torch::dtype(ToTorchDtype<int32_t>()).device(device, device_idx));

    torch::Tensor hash_table_cell_splits = torch::empty(
        {hash_table_splits.back() + 1},
        torch::dtype(ToTorchDtype<int32_t>()).device(device, device_idx));

    torch::Tensor out_hash_table_splits = torch::empty(
        {batch_size.value() + 1}, torch::dtype(ToTorchDtype<int32_t>()));
    for (size_t i = 0; i < hash_table_splits.size(); ++i)
    {
        out_hash_table_splits.data_ptr<int32_t>()[i] = hash_table_splits[i];
    }

#define BSHT_FN_PARAMETERS                                                  \
    points, radius, points_row_splits, hash_table_splits, hash_table_index, \
        hash_table_cell_splits

#define CALL(type, fn)                                                   \
    if (CompareTorchDtype<type>(point_type))                             \
    {                                                                    \
        fn<type>(BSHT_FN_PARAMETERS);                                    \
        return std::make_tuple(hash_table_index, hash_table_cell_splits, \
                               out_hash_table_splits);                   \
    }

    if (points.is_cuda())
    {
#ifdef BUILD_CUDA_MODULE
        // pass to cuda function
        CALL(float, BuildSpatialHashTableCUDA)
#else
        TORCH_CHECK(false,
                    "BuildSpatialHashTable was not compiled with CUDA support")
#endif
    }
    TORCH_CHECK(false, "BuildSpatialHashTable does not support " +
                           points.toString() + " as input for points")
    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>();
}

PYBIND11_MODULE(_fixedneighborsearch, m)
{
    m.def("build_spatial_hash_table", &BuildSpatialHashTable,
          "A function that builds a spatial hash table for points",
          py::arg("points"),
          py::arg("radius"),
          py::arg("points_row_splits"),
          py::arg("hash_table_size_factor"),
          py::arg("max_hash_table_size"));

    m.def("fixed_radius_search", &FixedRadiusSearch, "A function that performs fixed radius search",
          py::arg("points"),
          py::arg("queries"),
          py::arg("radius"),
          py::arg("points_row_splits"),
          py::arg("queries_row_splits"),
          py::arg("hash_table_splits"),
          py::arg("hash_table_index"),
          py::arg("hash_table_cell_splits"),
          py::arg("index_dtype"),
          py::arg("metric_str"),
          py::arg("ignore_query_point"),
          py::arg("return_distances"));
}