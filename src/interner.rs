use ahash::AHashMap;

/// Maps arbitrary `u32` keys to a dense, sequential `u32` id space.
///
/// Ids are assigned in first-seen order starting at `0`. The same key always
/// resolves to the same id for the lifetime of the interner.
#[derive(Default)]
pub struct Interner {
    ids: AHashMap<u32, u32>,
    keys: Vec<u32>,
}

impl Interner {
    /// Create an empty interner.
    pub fn new() -> Self {
        Self {
            ids: AHashMap::new(),
            keys: Vec::new(),
        }
    }

    /// The number of distinct keys interned so far.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Resolve `key` to its dense id, assigning a new one if not yet seen.
    pub fn intern(&mut self, key: u32) -> u32 {
        if let Some(&id) = self.ids.get(&key) {
            return id;
        }
        let id = self.keys.len() as u32;
        self.keys.push(key);
        self.ids.insert(key, id);
        id
    }

    /// Resolve a dense id back to its original key.
    ///
    /// # Panics
    ///
    /// Panics if `id` was never assigned by [`Self::intern`].
    pub fn decode(&self, id: u32) -> u32 {
        self.keys[id as usize]
    }

    /// All interned keys, in first-seen (ascending id) order.
    pub fn keys(&self) -> &[u32] {
        &self.keys
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// The same key always resolves to the same id.
    fn repeated_key_reuses_id() {
        let mut interner = Interner::new();
        let a = interner.intern(42);
        let b = interner.intern(42);
        assert_eq!(a, b);
        assert_eq!(interner.len(), 1);
    }

    #[test]
    /// New keys are assigned sequential ids in first-seen order.
    fn new_keys_get_sequential_ids() {
        let mut interner = Interner::new();
        assert_eq!(interner.intern(100), 0);
        assert_eq!(interner.intern(5), 1);
        assert_eq!(interner.intern(100), 0);
        assert_eq!(interner.intern(7), 2);
        assert_eq!(interner.len(), 3);
    }

    #[test]
    /// decode is the inverse of intern.
    fn decode_round_trips() {
        let mut interner = Interner::new();
        let id = interner.intern(1_000_000);
        assert_eq!(interner.decode(id), 1_000_000);
    }

    #[test]
    /// Widely spaced keys are handled like any other key.
    fn sparse_keys() {
        let mut interner = Interner::new();
        let a = interner.intern(0);
        let b = interner.intern(u32::MAX);
        assert_ne!(a, b);
        assert_eq!(interner.decode(a), 0);
        assert_eq!(interner.decode(b), u32::MAX);
    }

    #[test]
    /// keys() reflects first-seen order.
    fn keys_in_first_seen_order() {
        let mut interner = Interner::new();
        interner.intern(9);
        interner.intern(3);
        interner.intern(9);
        interner.intern(1);
        assert_eq!(interner.keys(), &[9, 3, 1]);
    }
}
