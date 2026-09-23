use arrow_array::ArrayRef;
use arrow_schema::DataType;
use hashbrown::HashMap;
use std::fmt;

use crate::interner::Interner;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by the DSU core.
#[derive(Debug, PartialEq)]
pub enum CoreError {
    LengthMismatch { src: usize, dst: usize },
    KeyTypeMismatch { expected: DataType, got: DataType },
    UnsupportedType(DataType),
    NullsNotAllowed { count: usize },
    TooManyKeys,
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { src, dst } => {
                write!(f, "src length {src} does not match dst length {dst}")
            }
            Self::KeyTypeMismatch { expected, got } => {
                write!(f, "expected {expected:?} array, got {got:?}")
            }
            Self::UnsupportedType(data_type) => {
                write!(f, "unsupported key type {data_type:?}")
            }
            Self::NullsNotAllowed { count } => {
                write!(f, "expected non-nullable array, got {count} null(s)")
            }
            Self::TooManyKeys => {
                write!(f, "cannot intern more than u32::MAX distinct keys")
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
    ///
    /// # Panics
    ///
    /// Panics if `new_len` is less than the current [`Self::len`].
    pub fn grow(&mut self, new_len: usize) {
        assert!(
            new_len >= self.parent.len(),
            "grow cannot shrink the node space (current: {}, requested: {})",
            self.parent.len(),
            new_len,
        );
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

    fn root(&self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            x = self.parent[x as usize];
        }
        x
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
// Dsu
// ---------------------------------------------------------------------------

/// Disjoint set union over arbitrary fixed-width integer keys.
///
/// Keys are interned to a dense id space internally.
pub struct Dsu {
    interner: Interner,
    core: DsuCore,
}

impl Default for Dsu {
    fn default() -> Self {
        Self::new()
    }
}

impl Dsu {
    /// Allocate an empty DSU.
    pub fn new() -> Self {
        Self {
            interner: Interner::new(),
            core: DsuCore::new(),
        }
    }

    /// Union all edges from the `src` and `dst` key arrays.
    ///
    /// Both arrays must be non-nullable Arrow arrays of equal length and the
    /// same `DataType`. Keys are interned as needed. The `DataType` of the
    /// first array seen fixes the key type for this `Dsu`'s lifetime.
    /// Previously unseen keys become new singleton nodes.
    pub fn union_edges(&mut self, src: &ArrayRef, dst: &ArrayRef) -> Result<(), CoreError> {
        for array in [src, dst] {
            if array.null_count() != 0 {
                return Err(CoreError::NullsNotAllowed {
                    count: array.null_count(),
                });
            }
        }

        let src_ids = self.interner.intern_array(src)?;
        let dst_ids = self.interner.intern_array(dst)?;

        if self.interner.len() > self.core.len() {
            self.core.grow(self.interner.len());
        }

        self.core.union_edges(&src_ids, &dst_ids)
    }

    /// Return only components touched by these hypothetical edges, without
    /// interning keys or changing parents and ranks.
    pub fn preview_union(
        &self,
        src: &ArrayRef,
        dst: &ArrayRef,
    ) -> Result<(ArrayRef, ArrayRef), CoreError> {
        for array in [src, dst] {
            if array.null_count() != 0 {
                return Err(CoreError::NullsNotAllowed {
                    count: array.null_count(),
                });
            }
        }
        if src.len() != dst.len() {
            return Err(CoreError::LengthMismatch {
                src: src.len(),
                dst: dst.len(),
            });
        }

        let (src_ids, dst_ids, pending) = self.interner.preview_arrays(src, dst)?;
        let existing_len = self.core.len();
        let root = |id: u32| {
            if (id as usize) < existing_len {
                self.core.root(id)
            } else {
                id
            }
        };
        let mut compact_ids = HashMap::<u32, u32>::new();
        for id in src_ids.iter().chain(&dst_ids) {
            let compact_id = compact_ids.len() as u32;
            compact_ids.entry(root(*id)).or_insert(compact_id);
        }

        let mut preview = DsuCore::new();
        preview.grow(compact_ids.len());
        for (&s, &d) in src_ids.iter().zip(&dst_ids) {
            let a = compact_ids[&root(s)] as usize;
            let b = compact_ids[&root(d)] as usize;
            preview.union_roots(a, b);
        }

        let mut key_ids = Vec::new();
        let mut group_ids = Vec::new();
        for id in 0..existing_len + pending.len() {
            let id = id as u32;
            if let Some(&compact_id) = compact_ids.get(&root(id)) {
                key_ids.push(id);
                group_ids.push(preview.root(compact_id));
            }
        }
        let label_ids =
            self.interner
                .preview_minimum_ids(&pending, &key_ids, &group_ids, compact_ids.len());

        Ok((
            self.interner.decode_preview_ids(&pending, &key_ids),
            self.interner.decode_preview_ids(&pending, &label_ids),
        ))
    }

    /// Add all keys from `keys` as singleton nodes when they are unseen.
    ///
    /// The array must be non-nullable. Its `DataType` must match the key type
    /// already established for this `Dsu`, if any.
    pub fn add(&mut self, keys: &ArrayRef) -> Result<(), CoreError> {
        if keys.null_count() != 0 {
            return Err(CoreError::NullsNotAllowed {
                count: keys.null_count(),
            });
        }

        self.interner.intern_array(keys)?;
        self.core.grow(self.interner.len());
        Ok(())
    }

    /// Return every interned key alongside its component label.
    ///
    /// Keys are returned in first-seen order. Each component's label is its
    /// smallest original key, independent of edge insertion order.
    pub fn components(&mut self) -> (ArrayRef, ArrayRef) {
        let dense_labels = self.core.labels();
        let keys = self.interner.keys_array();
        let labels = self.interner.decode_ids_to_array(&dense_labels);
        (keys, labels)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::UInt32Array;
    use arrow_array::cast::AsArray;
    use arrow_array::types::UInt32Type;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// DsuCore

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
    #[should_panic(expected = "grow cannot shrink")]
    fn grow_shrink_panics() {
        let mut dsu = DsuCore::new();
        dsu.grow(5);
        dsu.grow(3);
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

    // Dsu

    fn u32_array(values: Vec<u32>) -> ArrayRef {
        Arc::new(UInt32Array::from(values))
    }

    fn u32_values(array: &ArrayRef) -> Vec<u32> {
        array.as_primitive::<UInt32Type>().values().to_vec()
    }

    #[test]
    /// Sparse, non-contiguous keys merge into one component.
    fn dsu_unions_sparse_keys() {
        let mut dsu = Dsu::new();
        dsu.union_edges(
            &u32_array(vec![5, 1_000_000]),
            &u32_array(vec![1_000_000, 42]),
        )
        .unwrap();
        let (keys, labels) = dsu.components();
        let keys = u32_values(&keys);
        let labels = u32_values(&labels);
        let label_of = |key: u32| labels[keys.iter().position(|&k| k == key).unwrap()];
        assert_eq!(label_of(5), label_of(1_000_000));
        assert_eq!(label_of(1_000_000), label_of(42));
    }

    #[test]
    /// Keys that are never unioned still appear as singleton components.
    fn dsu_keeps_disjoint_keys_separate() {
        let mut dsu = Dsu::new();
        dsu.union_edges(&u32_array(vec![1, 2]), &u32_array(vec![2, 3]))
            .unwrap();
        dsu.union_edges(&u32_array(vec![100]), &u32_array(vec![100]))
            .unwrap();
        let (keys, labels) = dsu.components();
        let keys = u32_values(&keys);
        let labels = u32_values(&labels);
        let label_of = |key: u32| labels[keys.iter().position(|&k| k == key).unwrap()];
        assert_eq!(label_of(1), label_of(2));
        assert_eq!(label_of(2), label_of(3));
        assert_ne!(label_of(1), label_of(100));
    }

    #[test]
    /// add creates singleton nodes and ignores keys already interned.
    fn dsu_adds_singleton_keys() {
        let mut dsu = Dsu::new();
        dsu.add(&u32_array(vec![9, 3, 9])).unwrap();
        dsu.add(&u32_array(vec![3, 9])).unwrap();

        let (keys, labels) = dsu.components();
        let keys = u32_values(&keys);
        let labels = u32_values(&labels);

        assert_eq!(keys, vec![9, 3]);
        assert_ne!(labels[0], labels[1]);
    }

    #[test]
    /// add rejects arrays containing nulls.
    fn dsu_add_rejects_nulls() {
        let mut dsu = Dsu::new();
        let keys: ArrayRef = Arc::new(UInt32Array::from(vec![Some(1), None]));

        let result = dsu.add(&keys);

        assert_eq!(result, Err(CoreError::NullsNotAllowed { count: 1 }));
    }

    #[test]
    /// Interning is idempotent: the same key maps to the same node across separate `union_edges` calls.
    fn dsu_is_consistent_across_calls() {
        let mut dsu = Dsu::new();
        dsu.union_edges(&u32_array(vec![7]), &u32_array(vec![8]))
            .unwrap();
        dsu.union_edges(&u32_array(vec![8]), &u32_array(vec![9]))
            .unwrap();
        let (keys, labels) = dsu.components();
        let keys = u32_values(&keys);
        let labels = u32_values(&labels);
        let label_of = |key: u32| labels[keys.iter().position(|&k| k == key).unwrap()];
        assert_eq!(label_of(7), label_of(8));
        assert_eq!(label_of(8), label_of(9));
    }

    #[test]
    /// components() returns keys in first-seen order.
    fn dsu_components_first_seen_order() {
        let mut dsu = Dsu::new();
        dsu.union_edges(&u32_array(vec![9, 3]), &u32_array(vec![3, 1]))
            .unwrap();
        let (keys, _) = dsu.components();
        assert_eq!(u32_values(&keys), vec![9, 3, 1]);
    }

    #[test]
    /// components() returns insertion order-invariant labels
    fn dsu_labels_canonical_across_edge_orders() {
        let mut forward = Dsu::new();
        forward
            .union_edges(&u32_array(vec![1, 2]), &u32_array(vec![2, 3]))
            .unwrap();

        let mut reversed = Dsu::new();
        reversed
            .union_edges(&u32_array(vec![2, 1]), &u32_array(vec![3, 2]))
            .unwrap();

        fn component_map(dsu: &mut Dsu) -> HashMap<u32, u32> {
            let (keys, labels) = dsu.components();
            u32_values(&keys)
                .into_iter()
                .zip(u32_values(&labels))
                .collect()
        }

        let expected = HashMap::from([(1, 1), (2, 1), (3, 1)]);

        assert_eq!(component_map(&mut forward), expected);
        assert_eq!(component_map(&mut reversed), expected);
    }

    #[test]
    /// Unequal src/dst slices return a LengthMismatch error.
    fn dsu_length_mismatch_error() {
        let mut dsu = Dsu::new();
        let result = dsu.union_edges(&u32_array(vec![1, 2]), &u32_array(vec![1]));
        assert_eq!(result, Err(CoreError::LengthMismatch { src: 2, dst: 1 }));
    }

    #[test]
    /// Nulls are rejected regardless of the array's DataType.
    fn dsu_nulls_not_allowed() {
        let mut dsu = Dsu::new();
        let src: ArrayRef = Arc::new(UInt32Array::from(vec![Some(1), None]));
        let result = dsu.union_edges(&src, &u32_array(vec![1, 2]));
        assert_eq!(result, Err(CoreError::NullsNotAllowed { count: 1 }));
    }

    #[test]
    fn preview_matches_union() {
        let mut dsu = Dsu::new();
        dsu.union_edges(&u32_array(vec![9, 3, 7, 20]), &u32_array(vec![3, 1, 8, 21]))
            .unwrap();
        let parents = dsu.core.parent.clone();
        let ranks = dsu.core.rank.clone();
        let len = dsu.interner.len();
        let src = u32_array(vec![3, 50, 90]);
        let dst = u32_array(vec![50, 90, 60]);
        let (keys, labels) = dsu.preview_union(&src, &dst).unwrap();
        assert_eq!(u32_values(&keys), vec![9, 3, 1, 50, 90, 60]);
        assert_eq!(dsu.core.parent, parents);
        assert_eq!(dsu.core.rank, ranks);
        assert_eq!(dsu.interner.len(), len);

        dsu.union_edges(&src, &dst).unwrap();
        let (actual_keys, actual_labels) = dsu.components();
        let actual_keys = u32_values(&actual_keys);
        let actual_labels = u32_values(&actual_labels);
        for (&key, &label) in u32_values(&keys).iter().zip(u32_values(&labels).iter()) {
            let index = actual_keys.iter().position(|&k| k == key).unwrap();
            assert_eq!(label, actual_labels[index]);
        }
    }

    #[test]
    fn preview_empty() {
        let dsu = Dsu::new();
        let empty = u32_array(vec![]);
        let (keys, labels) = dsu.preview_union(&empty, &empty).unwrap();
        assert!(u32_values(&keys).is_empty());
        assert!(u32_values(&labels).is_empty());
        assert_eq!(dsu.interner.len(), 0);
        assert_eq!(dsu.core.len(), 0);
    }
}
