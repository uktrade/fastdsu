use crate::buffer::BufferInput;
use crate::dtype::DTypeKind;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::{BTreeMap, HashMap};

#[derive(Debug)]
enum NodeSpace {
    Dense {
        num_nodes: usize,
    },
    Sparse {
        unique_nodes: Vec<i128>,
        index: HashMap<i128, usize>,
    },
}

#[derive(Debug)]
pub(crate) struct DsuCore {
    parent: Vec<usize>,
    rank: Vec<u8>,
    space: NodeSpace,
    dtype: DTypeKind,
}

impl DsuCore {
    pub(crate) fn new_dense(num_nodes: usize, dtype: DTypeKind) -> PyResult<Self> {
        if num_nodes > 0 {
            let max_index = (num_nodes - 1) as i128;
            if !dtype.contains(max_index) {
                return Err(PyValueError::new_err(format!(
                    "num_nodes={num_nodes} exceeds dtype capacity"
                )));
            }
        }

        let mut parent = Vec::with_capacity(num_nodes);
        for idx in 0..num_nodes {
            parent.push(idx);
        }

        Ok(Self {
            parent,
            rank: vec![0; num_nodes],
            space: NodeSpace::Dense { num_nodes },
            dtype,
        })
    }

    pub(crate) fn new_sparse(nodes: Vec<i128>, dtype: DTypeKind) -> PyResult<Self> {
        let mut index = HashMap::with_capacity(nodes.len());
        let mut unique_nodes = Vec::with_capacity(nodes.len());

        for node in nodes {
            if index.contains_key(&node) {
                continue;
            }

            if !dtype.contains(node) {
                return Err(PyValueError::new_err(format!(
                    "node value {node} is out of range for dtype"
                )));
            }

            let dense = unique_nodes.len();
            unique_nodes.push(node);
            index.insert(node, dense);
        }

        let mut parent = Vec::with_capacity(unique_nodes.len());
        for idx in 0..unique_nodes.len() {
            parent.push(idx);
        }

        Ok(Self {
            parent,
            rank: vec![0; unique_nodes.len()],
            space: NodeSpace::Sparse {
                unique_nodes,
                index,
            },
            dtype,
        })
    }

    pub(crate) fn dtype(&self) -> DTypeKind {
        self.dtype
    }

    pub(crate) fn node_count(&self) -> usize {
        self.parent.len()
    }

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

    fn external_of_index(&self, index: usize) -> i128 {
        match &self.space {
            NodeSpace::Dense { .. } => index as i128,
            NodeSpace::Sparse { unique_nodes, .. } => unique_nodes[index],
        }
    }

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

    fn union_indices(&mut self, left: usize, right: usize) {
        let mut ra = self.find_root(left);
        let mut rb = self.find_root(right);

        if ra == rb {
            return;
        }

        if self.rank[ra] < self.rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }

        let equal_rank = self.rank[ra] == self.rank[rb];

        self.parent[rb] = ra;
        if equal_rank {
            self.rank[ra] = self.rank[ra].saturating_add(1);
        }
    }

    pub(crate) fn union_external(&mut self, left: i128, right: i128) -> PyResult<()> {
        let Some(li) = self.index_of_external(left) else {
            return Err(PyValueError::new_err(format!("unknown node id: {left}")));
        };
        let Some(ri) = self.index_of_external(right) else {
            return Err(PyValueError::new_err(format!("unknown node id: {right}")));
        };

        self.union_indices(li, ri);
        Ok(())
    }

    pub(crate) fn find_external(&mut self, node: i128) -> PyResult<i128> {
        let Some(idx) = self.index_of_external(node) else {
            return Err(PyValueError::new_err(format!("unknown node id: {node}")));
        };

        let root = self.find_root(idx);
        Ok(self.external_of_index(root))
    }

    pub(crate) fn connected_external(&mut self, left: i128, right: i128) -> PyResult<bool> {
        let Some(li) = self.index_of_external(left) else {
            return Err(PyValueError::new_err(format!("unknown node id: {left}")));
        };
        let Some(ri) = self.index_of_external(right) else {
            return Err(PyValueError::new_err(format!("unknown node id: {right}")));
        };

        Ok(self.find_root(li) == self.find_root(ri))
    }

    pub(crate) fn labels_values(&mut self) -> Vec<i128> {
        let mut out = Vec::with_capacity(self.node_count());
        for idx in 0..self.node_count() {
            let root = self.find_root(idx);
            out.push(self.external_of_index(root));
        }
        out
    }

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

    pub(crate) fn reset(&mut self) {
        for idx in 0..self.node_count() {
            self.parent[idx] = idx;
            self.rank[idx] = 0;
        }
    }

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
    use super::*;

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
