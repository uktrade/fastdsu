use std::fmt;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the DSU core.
#[derive(Debug, PartialEq)]
pub enum CoreError {
    /// The src and dst edge buffers had differing lengths.
    LengthMismatch { src: usize, dst: usize },
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { src, dst } => {
                write!(f, "src length {src} does not match dst length {dst}")
            }
        }
    }
}

impl std::error::Error for CoreError {}

// ---------------------------------------------------------------------------
// DsuCore
// ---------------------------------------------------------------------------

/// Disjoint set union over a dense integer node space `0..len`.
///
/// Uses recursive path compression and union by rank.
pub struct DsuCore {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl DsuCore {
    /// Allocate an empty DSU.
    pub fn new() -> Self {
        Self {
            parent: Vec::new(),
            rank: Vec::new(),
        }
    }

    /// The number of allocated nodes.
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Grow the node space to `new_len`, adding singleton nodes as needed.
    pub fn grow(&mut self, new_len: usize) {
        let old_len = self.parent.len();
        self.parent.extend((old_len..new_len).map(|i| i as u32));
        self.rank.resize(new_len, 0);
    }

    /// Union all edges from the `src` and `dst` slices.
    ///
    /// Caller must ensure all IDs are within `0..len` before calling.
    pub fn union_edges(&mut self, src: &[u32], dst: &[u32]) -> Result<(), CoreError> {
        if src.len() != dst.len() {
            return Err(CoreError::LengthMismatch {
                src: src.len(),
                dst: dst.len(),
            });
        }
        for (&s, &d) in src.iter().zip(dst.iter()) {
            self.union_roots(s as usize, d as usize);
        }
        Ok(())
    }

    /// Return the root label for each node.
    ///
    /// Applies path compression across all nodes as a side effect.
    pub fn labels(&mut self) -> Vec<u32> {
        (0..self.parent.len())
            .map(|i| self.find_root(i) as u32)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    fn find_root(&mut self, x: usize) -> usize {
        if self.parent[x] as usize != x {
            let root = self.find_root(self.parent[x] as usize);
            self.parent[x] = root as u32;
        }
        self.parent[x] as usize
    }

    fn union_roots(&mut self, a: usize, b: usize) {
        let ra = self.find_root(a);
        let rb = self.find_root(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb as u32,
            std::cmp::Ordering::Greater => self.parent[rb] = ra as u32,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra as u32;
                self.rank[ra] += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Each node starts as its own root after grow.
    fn new_nodes_are_own_roots() {
        let mut dsu = DsuCore::new();
        dsu.grow(5);
        let labels = dsu.labels();
        for (i, &label) in labels.iter().enumerate() {
            assert_eq!(label as usize, i);
        }
    }

    #[test]
    /// grow extends the node space with singleton nodes.
    fn grow_adds_singleton_nodes() {
        let mut dsu = DsuCore::new();
        dsu.grow(2);
        dsu.grow(5);
        assert_eq!(dsu.len(), 5);
        let labels = dsu.labels();
        for (i, &label) in labels.iter().enumerate() {
            assert_eq!(label as usize, i);
        }
    }

    #[test]
    /// Two disjoint chains are each merged into one component.
    fn union_edges_merges_components() {
        let mut dsu = DsuCore::new();
        dsu.grow(6);
        dsu.union_edges(&[0, 1, 3, 4], &[1, 2, 4, 5]).unwrap();
        let labels = dsu.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    /// Unconnected nodes are not merged into existing components.
    fn labels_groups_components() {
        let mut dsu = DsuCore::new();
        dsu.grow(6);
        dsu.union_edges(&[0, 1, 3], &[1, 2, 4]).unwrap();
        let labels = dsu.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[5]);
    }

    #[test]
    /// Unequal src/dst slices return a LengthMismatch error.
    fn length_mismatch_error() {
        let mut dsu = DsuCore::new();
        dsu.grow(4);
        let result = dsu.union_edges(&[0, 1], &[1]);
        assert_eq!(result, Err(CoreError::LengthMismatch { src: 2, dst: 1 }));
    }
}
