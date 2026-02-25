//! Core disjoint-set implementation over dense indices.

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::fmt;

/// Errors produced by the DSU core.
#[derive(Debug, PartialEq)]
pub enum CoreError {
    /// A node ID was not found in the node space.
    UnknownNode(i64),
    /// Two edge buffers had differing lengths.
    LengthMismatch { src: usize, dst: usize },
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownNode(id) => write!(f, "unknown node id: {id}"),
            Self::LengthMismatch { src, dst } => {
                write!(f, "src length {src} does not match dst length {dst}")
            }
        }
    }
}

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
        unique_nodes: Vec<i64>,
        /// Dense index by external node ID.
        index: HashMap<i64, usize>,
    },
}

/// Union-find storage and node-space mapping.
#[derive(Debug)]
pub struct DsuCore {
    /// Parent pointer per dense node.
    parent: Vec<usize>,
    /// Rank per dense node.
    rank: Vec<u8>,
    /// External-to-dense node mapping strategy.
    space: NodeSpace,
}

impl DsuCore {
    /// Construct a dense DSU over nodes `0..num_nodes`.
    pub fn new_dense(num_nodes: usize) -> Self {
        Self::with_space(NodeSpace::Dense { num_nodes }, num_nodes)
    }

    /// Construct a sparse DSU from an explicit node list, deduplicating in order.
    pub fn new_sparse(nodes: Vec<i64>) -> Self {
        let mut index = HashMap::with_capacity(nodes.len());
        let mut unique_nodes = Vec::with_capacity(nodes.len());

        for node in nodes {
            if let Entry::Occupied(_) = index.entry(node) {
                continue;
            }
            let dense_index = unique_nodes.len();
            unique_nodes.push(node);
            index.insert(node, dense_index);
        }

        let node_count = unique_nodes.len();
        Self::with_space(
            NodeSpace::Sparse {
                unique_nodes,
                index,
            },
            node_count,
        )
    }

    /// Return the number of nodes tracked by this DSU.
    pub fn node_count(&self) -> usize {
        self.parent.len()
    }

    /// Build a DSU with identity parents and zero ranks.
    fn with_space(space: NodeSpace, node_count: usize) -> Self {
        Self {
            parent: (0..node_count).collect(),
            rank: vec![0; node_count],
            space,
        }
    }

    /// Resolve an external node ID to a dense index.
    fn index_of_external(&self, node: i64) -> Option<usize> {
        match &self.space {
            NodeSpace::Dense { num_nodes } => {
                let as_usize = usize::try_from(node).ok()?;
                (as_usize < *num_nodes).then_some(as_usize)
            }
            NodeSpace::Sparse { index, .. } => index.get(&node).copied(),
        }
    }

    /// Resolve a dense index to its external node ID.
    fn external_of_index(&self, index: usize) -> i64 {
        match &self.space {
            NodeSpace::Dense { .. } => index as i64,
            NodeSpace::Sparse { unique_nodes, .. } => unique_nodes[index],
        }
    }

    /// Resolve an external node ID or return `CoreError::UnknownNode`.
    fn index_or_err(&self, node: i64) -> Result<usize, CoreError> {
        self.index_of_external(node)
            .ok_or(CoreError::UnknownNode(node))
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
    pub fn union_external(&mut self, left: i64, right: i64) -> Result<(), CoreError> {
        let left_index = self.index_or_err(left)?;
        let right_index = self.index_or_err(right)?;
        self.union_indices(left_index, right_index);
        Ok(())
    }

    /// Return the current representative for an external node ID.
    pub fn find_external(&mut self, node: i64) -> Result<i64, CoreError> {
        let index = self.index_or_err(node)?;
        let root = self.find_root(index);
        Ok(self.external_of_index(root))
    }

    /// Return whether two external node IDs are in the same component.
    pub fn connected_external(&mut self, left: i64, right: i64) -> Result<bool, CoreError> {
        let left_index = self.index_or_err(left)?;
        let right_index = self.index_or_err(right)?;
        Ok(self.find_root(left_index) == self.find_root(right_index))
    }

    /// Materialise one representative per node in dense-order.
    pub fn labels_values(&mut self) -> Vec<i64> {
        (0..self.node_count())
            .map(|idx| {
                let root = self.find_root(idx);
                self.external_of_index(root)
            })
            .collect()
    }

    /// Materialise sorted components grouped by representative.
    pub fn components_values(&mut self) -> Vec<Vec<i64>> {
        let mut groups: BTreeMap<i64, Vec<i64>> = BTreeMap::new();

        for idx in 0..self.node_count() {
            let root = self.find_root(idx);
            let representative = self.external_of_index(root);
            groups
                .entry(representative)
                .or_default()
                .push(self.external_of_index(idx));
        }

        let mut out: Vec<Vec<i64>> = groups.into_values().collect();
        for group in &mut out {
            group.sort_unstable();
        }
        out
    }

    /// Reset parent/rank arrays so each node is its own set.
    pub fn reset(&mut self) {
        for (idx, parent) in self.parent.iter_mut().enumerate() {
            *parent = idx;
        }
        self.rank.fill(0);
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for DSU core behaviour.

    use super::*;

    /// Confirm dense DSU operations and label extraction.
    #[test]
    fn dense_union_find_and_labels() {
        let mut dsu = DsuCore::new_dense(6);
        dsu.union_external(0, 1).unwrap();
        dsu.union_external(1, 2).unwrap();
        dsu.union_external(4, 5).unwrap();

        assert_eq!(dsu.find_external(2).unwrap(), 0);
        assert_eq!(dsu.find_external(5).unwrap(), 4);
        assert_eq!(dsu.labels_values(), vec![0, 0, 0, 3, 4, 4]);
    }

    /// Confirm sparse deduplication and deterministic component output.
    #[test]
    fn sparse_dedup_and_components() {
        let mut dsu = DsuCore::new_sparse(vec![10, 25, 47, 99, 130, 200, 25]);

        dsu.union_external(10, 25).unwrap();
        dsu.union_external(25, 47).unwrap();
        dsu.union_external(130, 200).unwrap();

        assert_eq!(dsu.node_count(), 6);
        assert_eq!(dsu.find_external(47).unwrap(), 10);
        assert_eq!(dsu.labels_values(), vec![10, 10, 10, 99, 130, 130]);
        assert_eq!(
            dsu.components_values(),
            vec![vec![10, 25, 47], vec![99], vec![130, 200]]
        );
    }

    /// Confirm rank-tie behaviour remains deterministic.
    #[test]
    fn equal_rank_tie_break_is_left_argument_root() {
        let mut dsu = DsuCore::new_dense(4);
        dsu.union_external(1, 2).unwrap();
        dsu.union_external(3, 2).unwrap();

        assert_eq!(dsu.find_external(1).unwrap(), 1);
        assert_eq!(dsu.find_external(2).unwrap(), 1);
        assert_eq!(dsu.find_external(3).unwrap(), 1);
    }

    /// Confirm unknown node IDs produce the correct error.
    #[test]
    fn unknown_node_returns_error() {
        let mut dsu = DsuCore::new_dense(3);
        assert_eq!(dsu.find_external(99), Err(CoreError::UnknownNode(99)));
        assert_eq!(dsu.union_external(0, 99), Err(CoreError::UnknownNode(99)));
    }
}
