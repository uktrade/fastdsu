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
// NodeIndex trait
// ---------------------------------------------------------------------------

/// A primitive integer type that can serve as a node index.
pub trait NodeIndex: Copy + Eq {
    fn from_usize(n: usize) -> Self;
    fn as_usize(self) -> usize;
}

macro_rules! impl_node_index {
    ($($t:ty),*) => {
        $(impl NodeIndex for $t {
            #[inline] fn from_usize(n: usize) -> Self { n as Self }
            #[inline] fn as_usize(self) -> usize { self as usize }
        })*
    };
}

impl_node_index!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

// ---------------------------------------------------------------------------
// DsuCore
// ---------------------------------------------------------------------------

/// Disjoint set union over a dense integer node space `0..len`.
///
/// Uses recursive path compression and union by rank.
pub struct DsuCore<T> {
    parent: Vec<T>,
    rank: Vec<u8>,
}

impl<T: NodeIndex> DsuCore<T> {
    /// Allocate a DSU for `num_nodes` nodes, each initially its own component.
    pub fn new(num_nodes: usize) -> Self {
        Self {
            parent: (0..num_nodes).map(T::from_usize).collect(),
            rank: vec![0; num_nodes],
        }
    }

    /// The number of allocated nodes.
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// Grow the node space to `new_len`, adding singleton nodes as needed.
    pub fn grow(&mut self, new_len: usize) {
        let old_len = self.parent.len();
        self.parent.extend((old_len..new_len).map(T::from_usize));
        self.rank.resize(new_len, 0);
    }

    /// Union all edges from the `src` and `dst` slices.
    ///
    /// Caller must ensure all IDs are within `0..len` before calling.
    pub fn union_edges(&mut self, src: &[T], dst: &[T]) -> Result<(), CoreError> {
        if src.len() != dst.len() {
            return Err(CoreError::LengthMismatch {
                src: src.len(),
                dst: dst.len(),
            });
        }
        for (&s, &d) in src.iter().zip(dst.iter()) {
            self.union_roots(s.as_usize(), d.as_usize());
        }
        Ok(())
    }

    /// Return the root label for each node.
    ///
    /// Applies path compression across all nodes as a side effect.
    pub fn labels(&mut self) -> Vec<T> {
        (0..self.parent.len())
            .map(|i| T::from_usize(self.find_root(i)))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    fn find_root(&mut self, x: usize) -> usize {
        if self.parent[x].as_usize() != x {
            let root = self.find_root(self.parent[x].as_usize());
            self.parent[x] = T::from_usize(root);
        }
        self.parent[x].as_usize()
    }

    fn union_roots(&mut self, a: usize, b: usize) {
        let ra = self.find_root(a);
        let rb = self.find_root(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = T::from_usize(rb),
            std::cmp::Ordering::Greater => self.parent[rb] = T::from_usize(ra),
            std::cmp::Ordering::Equal => {
                self.parent[rb] = T::from_usize(ra);
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
    fn new_nodes_are_own_roots() {
        let mut dsu: DsuCore<u32> = DsuCore::new(5);
        let labels = dsu.labels();
        for (i, &label) in labels.iter().enumerate().take(5) {
            assert_eq!(label as usize, i);
        }
    }

    #[test]
    fn grow_adds_singleton_nodes() {
        let mut dsu: DsuCore<u32> = DsuCore::new(2);
        dsu.grow(5);
        assert_eq!(dsu.len(), 5);
        let labels = dsu.labels();
        for (i, &label) in labels.iter().enumerate() {
            assert_eq!(label as usize, i);
        }
    }

    #[test]
    fn union_edges_merges_components() {
        let mut dsu: DsuCore<u32> = DsuCore::new(6);
        let src = [0u32, 1, 3, 4];
        let dst = [1u32, 2, 4, 5];
        dsu.union_edges(&src, &dst).unwrap();
        let labels = dsu.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn labels_groups_components() {
        let mut dsu: DsuCore<u32> = DsuCore::new(6);
        let src = [0u32, 1, 3];
        let dst = [1u32, 2, 4];
        dsu.union_edges(&src, &dst).unwrap();
        let labels = dsu.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
        assert_ne!(labels[0], labels[5]);
    }

    #[test]
    fn length_mismatch_error() {
        let mut dsu: DsuCore<u32> = DsuCore::new(4);
        let result = dsu.union_edges(&[0u32, 1], &[1u32]);
        assert_eq!(result, Err(CoreError::LengthMismatch { src: 2, dst: 1 }));
    }

    #[test]
    fn works_with_u8_nodes() {
        let mut dsu: DsuCore<u8> = DsuCore::new(4);
        dsu.union_edges(&[0u8, 2], &[1u8, 3]).unwrap();
        let labels = dsu.labels();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }
}
