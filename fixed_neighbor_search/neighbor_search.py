import torch

try:
    from _fixedneighborsearch import fixed_radius_search, build_spatial_hash_table
except ImportError:
    def fixed_radius_search(*args, **kwargs):
        raise NotImplementedError("_fixedneighborsearch CUDA extension not available")
        
    def build_spatial_hash_table(*args, **kwargs):
        raise NotImplementedError("_fixedneighborsearch CUDA extension not available")

class FixedRadiusSearch(torch.nn.Module):
    """Fixed radius search for 3D point clouds.

    This layer computes the neighbors for a fixed radius on a point cloud.


    Arguments:
      metric: Either L1, L2 or Linf. Default is L2.

      ignore_query_point: If True the points that coincide with the center of
        the search window will be ignored. This excludes the query point if
        'queries' and 'points' are the same point cloud.

      return_distances: If True the distances for each neighbor will be returned.
        If False a zero length Tensor will be returned instead.
    """

    def __init__(self,
                 metric='L2',
                 ignore_query_point=False,
                 return_distances=False,
                 max_hash_table_size=32 * 2**20,
                 **kwargs):
        super().__init__()
        self.metric = metric
        self.ignore_query_point = ignore_query_point
        self.return_distances = return_distances
        self.max_hash_table_size = max_hash_table_size

    def forward(self,
                points,
                queries,
                radius,
                points_row_splits=None,
                queries_row_splits=None,
                hash_table_size_factor=1 / 64,
                hash_table=None):
        """This function computes the neighbors within a fixed radius for each query point.

        Arguments:

          points: The 3D positions of the input points.

          queries: The 3D positions of the query points.

          radius: A scalar with the neighborhood radius

          points_row_splits: Optional 1D vector with the row splits information
            if points is batched. This vector is [0, num_points] if there is
            only 1 batch item.

          queries_row_splits: Optional 1D vector with the row splits information
            if queries is batched.  This vector is [0, num_queries] if there is
            only 1 batch item.

          hash_table_size_factor: Scalar. The size of the hash table as fraction
            of points.

          hash_table: A precomputed hash table generated with build_spatial_hash_table().
            This input can be used to explicitly force the reuse of a hash table in special
            cases and is usually not needed.
            Note that the hash table must have been generated with the same 'points' array.

        Returns:
          3 Tensors in the following order

          neighbors_index
            The compact list of indices of the neighbors. The corresponding query point
            can be inferred from the 'neighbor_count_row_splits' vector.

          neighbors_row_splits
            The exclusive prefix sum of the neighbor count for the query points including
            the total neighbor count as the last element. The size of this array is the
            number of queries + 1.

          neighbors_distance
            Stores the distance to each neighbor if 'return_distances' is True.
            Note that the distances are squared if metric is L2.
            This is a zero length Tensor if 'return_distances' is False.
        """
        device = points.device

        if points_row_splits is None:
            points_row_splits = torch.Tensor([0, points.shape[0]]).to(dtype=torch.int64, device=device)
        if queries_row_splits is None:
            queries_row_splits = torch.Tensor([0, queries.shape[0]]).to(dtype=torch.int64, device=device)

        if hash_table is None:
            table = build_spatial_hash_table(
                max_hash_table_size=self.max_hash_table_size,
                points=points,
                radius=radius,
                points_row_splits=points_row_splits,
                hash_table_size_factor=hash_table_size_factor)
        else:
            table = hash_table

        points_row_splits = points_row_splits.to(device)
        queries_row_splits = queries_row_splits.to(device)
        hash_table_index, hash_table_cell_splits, hash_table_splits = table
        hash_table_index = hash_table_index.to(device)
        hash_table_cell_splits = hash_table_cell_splits.to(device)
        hash_table_splits = hash_table_splits.to(device)
        result = fixed_radius_search(
            points=points,
            queries=queries,
            radius=radius,
            points_row_splits=points_row_splits,
            queries_row_splits=queries_row_splits,
            hash_table_splits=hash_table_splits,
            hash_table_index=hash_table_index,
            hash_table_cell_splits=hash_table_cell_splits,
            index_dtype=hash_table_index.dtype,
            metric_str=self.metric,
            ignore_query_point=self.ignore_query_point,
            return_distances=self.return_distances
        )


        neighbors_index, neighbors_row_splits, neighbors_distance = result
        if self.return_distances:
          neighbors_index, neighbors_row_splits, neighbors_distance
        else:
          return neighbors_index, neighbors_row_splits