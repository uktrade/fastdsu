//! Core disjoint-set implementation over dense indices.

use crate::buffer::BufferInput;
use crate::dtype::DTypeKind;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};

/// Mapping between external node identifiers and internal dense indices.
#[derive(Debug)]
enum NodeSpace {
    /// Dense node space `0..num_nodes`.
    Dense {
        /// Number of nodes in the dense space.
        num_nodes: usize,
    },
    /// Sparse node space defined by explicit node IDs.
    Sparse {
        /// External node ID by dense index.
        unique_nodes: Vec<i128>,
        /// Dense index by external node ID.
        index: HashMap<i128, usize>,
    },
}

/// Union-find storage and node-space mapping.
#[derive(Debug)]
pub(crate) struct DsuCore {
    /// Parent pointer per dense node.
    parent: Vec<usize>,
    /// Rank per dense node.
    rank: Vec<u8>,
    /// External-to-dense node mapping strategy.
    space: NodeSpace,
    /// Working integer dtype for all node identifiers.
    dtype: DTypeKind,
}

impl DsuCore {
    /// Construct a dense DSU over nodes `0..num_nodes`.
    pub(crate) fn new_dense(num_nodes: usize, dtype: DTypeKind) -> PyResult<Self> {
        if num_nodes > 0 {
            let max_index = (num_nodes - 1) as i128;
            if !dtype.contains(max_index) {
                return Err(PyValueError::new_err(format!(
                    "num_nodes={num_nodes} exceeds dtype capacity"
                )));
            }
        }

        Ok(Self::with_space(
            NodeSpace::Dense { num_nodes },
            dtype,
            num_nodes,
        ))
    }

    /// Construct a sparse DSU from an explicit node list.
    pub(crate) fn new_sparse(nodes: Vec<i128>, dtype: DTypeKind) -> PyResult<Self> {
        let mut index = HashMap::with_capacity(nodes.len());
        let mut unique_nodes = Vec::with_capacity(nodes.len());

        for node in nodes {
            if let Entry::Occupied(_) = index.entry(node) {
                continue;
            }

            if !dtype.contains(node) {
                return Err(PyValueError::new_err(format!(
                    "node value {node} is out of range for dtype"
                )));
            }

            let dense_index = unique_nodes.len();
            unique_nodes.push(node);
            index.insert(node, dense_index);
        }

        let node_count = unique_nodes.len();
        Ok(Self::with_space(
            NodeSpace::Sparse {
                unique_nodes,
                index,
            },
            dtype,
            node_count,
        ))
    }

    /// Return the dtype used by this DSU.
    pub(crate) fn dtype(&self) -> DTypeKind {
        self.dtype
    }

    /// Return the number of nodes tracked by this DSU.
    pub(crate) fn node_count(&self) -> usize {
        self.parent.len()
    }

    /// Build a DSU with identity parents and zero ranks.
    fn with_space(space: NodeSpace, dtype: DTypeKind, node_count: usize) -> Self {
        Self {
            parent: (0..node_count).collect(),
            rank: vec![0; node_count],
            space,
            dtype,
        }
    }

    /// Resolve an external node ID to a dense index.
    fn index_of_external(&self, node: i128) -> Option<usize> {
        match &self.space {
            NodeSpace::Dense { num_nodes } => {
                if node < 0 {
                    return None;
                }
                let as_usize = usize::try_from(node).ok()?;
                (as_usize < *num_nodes).then_some(as_usize)
            }
            NodeSpace::Sparse { index, .. } => index.get(&node).copied(),
        }
    }

    /// Resolve a dense index to its external node ID.
    fn external_of_index(&self, index: usize) -> i128 {
        match &self.space {
            NodeSpace::Dense { .. } => index as i128,
            NodeSpace::Sparse { unique_nodes, .. } => unique_nodes[index],
        }
    }

    /// Resolve an external node ID or raise the standard unknown-node error.
    fn index_or_err(&self, node: i128) -> PyResult<usize> {
        self.index_of_external(node)
            .ok_or_else(|| PyValueError::new_err(format!("unknown node id: {node}")))
    }

    /// Find the root dense index for `node`, applying path compression.
    fn find_root(&mut self, node: usize) -> usize {
        let mut root = node;
        while self.parent[root] != root {
            root = self.parent[root];
        }

        let mut current = node;
        while self.parent[current] != current {
            let next = self.parent[current];
            self.parent[current] = root;
            current = next;
        }

        root
    }

    /// Union two dense indices via rank heuristic.
    fn union_indices(&mut self, left: usize, right: usize) {
        let mut left_root = self.find_root(left);
        let mut right_root = self.find_root(right);

        if left_root == right_root {
            return;
        }

        if self.rank[left_root] < self.rank[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }

        let equal_rank = self.rank[left_root] == self.rank[right_root];

        self.parent[right_root] = left_root;
        if equal_rank {
            self.rank[left_root] = self.rank[left_root].saturating_add(1);
        }
    }

    /// Union two external node IDs.
    pub(crate) fn union_external(&mut self, left: i128, right: i128) -> PyResult<()> {
        let left_index = self.index_or_err(left)?;
        let right_index = self.index_or_err(right)?;
        self.union_indices(left_index, right_index);
        Ok(())
    }

    /// Return the current representative for an external node ID.
    pub(crate) fn find_external(&mut self, node: i128) -> PyResult<i128> {
        let index = self.index_or_err(node)?;
        let root = self.find_root(index);
        Ok(self.external_of_index(root))
    }

    /// Return whether two external node IDs are connected.
    pub(crate) fn connected_external(&mut self, left: i128, right: i128) -> PyResult<bool> {
        let left_index = self.index_or_err(left)?;
        let right_index = self.index_or_err(right)?;
        Ok(self.find_root(left_index) == self.find_root(right_index))
    }

    /// Materialise one representative per node in dense-order.
    pub(crate) fn labels_values(&mut self) -> Vec<i128> {
        let mut out = Vec::with_capacity(self.node_count());
        for idx in 0..self.node_count() {
            let root = self.find_root(idx);
            out.push(self.external_of_index(root));
        }
        out
    }

    /// Materialise sorted components grouped by representative.
    pub(crate) fn components_values(&mut self) -> Vec<Vec<i128>> {
        let mut groups: BTreeMap<i128, Vec<i128>> = BTreeMap::new();

        for idx in 0..self.node_count() {
            let root = self.find_root(idx);
            let representative = self.external_of_index(root);
            groups
                .entry(representative)
                .or_default()
                .push(self.external_of_index(idx));
        }

        let mut out = Vec::with_capacity(groups.len());
        for mut group in groups.into_values() {
            group.sort_unstable();
            out.push(group);
        }
        out
    }

    /// Reset parent/rank arrays so each node is its own set.
    pub(crate) fn reset(&mut self) {
        for (idx, parent) in self.parent.iter_mut().enumerate() {
            *parent = idx;
        }
        self.rank.fill(0);
    }

    /// Union all edge pairs from two validated buffers.
    pub(crate) fn union_from_buffers(
        &mut self,
        src: &BufferInput,
        dst: &BufferInput,
    ) -> PyResult<()> {
        if src.len() != dst.len() {
            return Err(PyValueError::new_err("src and dst must have equal length"));
        }

        if src.dtype != self.dtype || dst.dtype != self.dtype {
            return Err(PyValueError::new_err(
                "buffer dtype does not match DSU dtype",
            ));
        }

        for idx in 0..src.len() {
            let left = src.read_checked(idx, self.dtype)?;
            let right = dst.read_checked(idx, self.dtype)?;
            self.union_external(left, right)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for DSU core behaviour.

    use super::*;

    /// Confirm dense DSU operations and label extraction.
    #[test]
    fn dense_union_find_and_labels() {
        let mut dsu = DsuCore::new_dense(6, DTypeKind::U32).expect("dense dsu");
        dsu.union_external(0, 1).unwrap();
        dsu.union_external(1, 2).unwrap();
        dsu.union_external(4, 5).unwrap();

        assert_eq!(dsu.find_external(2).unwrap(), 0);
        assert_eq!(dsu.find_external(5).unwrap(), 4);

        let labels = dsu.labels_values();
        assert_eq!(labels, vec![0, 0, 0, 3, 4, 4]);
    }

    /// Confirm sparse deduplication and deterministic component output.
    #[test]
    fn sparse_dedup_and_components() {
        let mut dsu = DsuCore::new_sparse(vec![10, 25, 47, 99, 130, 200, 25], DTypeKind::I64)
            .expect("sparse dsu");

        dsu.union_external(10, 25).unwrap();
        dsu.union_external(25, 47).unwrap();
        dsu.union_external(130, 200).unwrap();

        assert_eq!(dsu.node_count(), 6);
        assert_eq!(dsu.find_external(47).unwrap(), 10);

        let labels = dsu.labels_values();
        assert_eq!(labels, vec![10, 10, 10, 99, 130, 130]);

        let comps = dsu.components_values();
        assert_eq!(comps, vec![vec![10, 25, 47], vec![99], vec![130, 200]],);
    }

    /// Confirm rank-tie behaviour remains deterministic.
    #[test]
    fn equal_rank_tie_break_is_left_argument_root() {
        let mut dsu = DsuCore::new_dense(4, DTypeKind::U32).expect("dense dsu");
        dsu.union_external(1, 2).unwrap();
        dsu.union_external(3, 2).unwrap();

        assert_eq!(dsu.find_external(1).unwrap(), 1);
        assert_eq!(dsu.find_external(2).unwrap(), 1);
        assert_eq!(dsu.find_external(3).unwrap(), 1);
    }
}
