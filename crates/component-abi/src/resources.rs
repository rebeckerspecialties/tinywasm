//! Bounded host resources for custom WIT-shaped imports.
//!
//! This table owns host values and validates guest-supplied `i32` handles.
//! It is deliberately separate from the component model's canonical handle
//! table: a future component linker will validate `own`/`borrow` transfers and
//! map them to these host values. Rust borrows protect synchronous lookups
//! today; cross-call and async borrow scopes are not implemented here.

use alloc::string::ToString;
use alloc::vec::Vec;
use core::fmt;
use core::num::NonZeroU16;

/// Identifies a WIT resource type within one host table.
///
/// The host assigns distinct values for distinct resource types. This value
/// is never taken from an untrusted guest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceType(u32);

impl ResourceType {
    /// Creates a host-assigned resource type identifier.
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
}

/// An opaque handle that can cross a core-Wasm `i32` import boundary.
///
/// Raw values received from a guest must always be looked up through the
/// table associated with that guest. Values from distinct tables may collide;
/// the integer alone is not a cross-instance capability or proof of origin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceHandle(u32);

impl ResourceHandle {
    /// Wraps an untrusted raw guest value for validation by the table.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Returns the raw value for a core-Wasm `i32` result.
    pub const fn into_raw(self) -> u32 {
        self.0
    }
}

/// Failure of a host resource operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceError {
    /// The configured live-handle or slot bound has been reached.
    Full,
    /// Table backing storage could not be reserved.
    Allocation,
    /// The handle is zero, stale, or absent from this table.
    InvalidHandle,
    /// The handle is live but belongs to another WIT resource type.
    WrongType,
}

impl fmt::Display for ResourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => f.write_str("host resource table is full"),
            Self::Allocation => f.write_str("host resource table allocation failed"),
            Self::InvalidHandle => f.write_str("invalid host resource handle"),
            Self::WrongType => f.write_str("host resource handle has the wrong type"),
        }
    }
}

impl From<ResourceError> for tinywasm::Error {
    fn from(error: ResourceError) -> Self {
        Self::Other(error.to_string())
    }
}

struct Entry<T> {
    kind: ResourceType,
    value: T,
}

struct Slot<T> {
    generation: u32,
    entry: Option<Entry<T>>,
}

/// A per-instance, bounded table of host-owned resource values.
///
/// Values are stored inline in a vector, with no allocation on lookup or
/// removal. A removed slot's generation changes before reuse, so old handles
/// cannot name a new value in the same table. Exhausted generations retire
/// their slots instead of wrapping. Dropping the table drops every live value.
///
/// This is a host-side table, not the canonical component handle table. A
/// future component adapter must still enforce `own`/`borrow` transfer and
/// borrow-scope rules. Raw core-Wasm handles may collide across tables;
/// bind callbacks to the calling guest/worker's own table.
pub struct ResourceTable<T> {
    slots: Vec<Slot<T>>,
    free: Vec<u16>,
    max_live: u16,
    live: u16,
    slot_bits: u32,
    slot_mask: u32,
}

impl<T> ResourceTable<T> {
    /// Creates a table with an explicit live-resource and backing-slot cap.
    ///
    /// The cap is at most 65,535. Use a smaller cap on constrained devices.
    pub fn new(max_live: NonZeroU16) -> Self {
        let max_live = max_live.get();
        let slot_bits = u32::BITS - u32::from(max_live).leading_zeros();
        let slot_mask = (1_u32 << slot_bits) - 1;
        Self { slots: Vec::new(), free: Vec::new(), max_live, live: 0, slot_bits, slot_mask }
    }

    /// Returns the number of live resources.
    pub const fn len(&self) -> usize {
        self.live as usize
    }

    /// Returns whether no resources are live.
    pub const fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Inserts one value and returns a raw-ABI-compatible handle.
    ///
    /// On failure the uninserted value is dropped, so native resources owned
    /// by its `Drop` implementation are not leaked.
    pub fn insert(&mut self, kind: ResourceType, value: T) -> Result<ResourceHandle, ResourceError> {
        if self.live == self.max_live {
            return Err(ResourceError::Full);
        }
        let index = if let Some(index) = self.free.pop() {
            index
        } else {
            if self.slots.len() >= self.max_live as usize {
                return Err(ResourceError::Full);
            }
            self.slots.try_reserve(1).map_err(|_| ResourceError::Allocation)?;
            // Ensure every future remove can push its free slot without allocating.
            self.free.try_reserve(self.slots.len() + 1 - self.free.len()).map_err(|_| ResourceError::Allocation)?;
            let index = self.slots.len() as u16;
            self.slots.push(Slot { generation: 0, entry: None });
            index
        };
        let slot = &mut self.slots[index as usize];
        debug_assert!(slot.entry.is_none());
        slot.entry = Some(Entry { kind, value });
        self.live += 1;
        Ok(ResourceHandle((slot.generation << self.slot_bits) | (u32::from(index) + 1)))
    }

    /// Borrows a live value after checking its handle and WIT resource type.
    pub fn get(&self, handle: ResourceHandle, kind: ResourceType) -> Result<&T, ResourceError> {
        let index = self.validate(handle, kind)?;
        Ok(&self.slots[index].entry.as_ref().expect("validated live slot").value)
    }

    /// Mutably borrows a live value after checking its handle and type.
    pub fn get_mut(&mut self, handle: ResourceHandle, kind: ResourceType) -> Result<&mut T, ResourceError> {
        let index = self.validate(handle, kind)?;
        Ok(&mut self.slots[index].entry.as_mut().expect("validated live slot").value)
    }

    /// Transfers ownership of a live value out of the table.
    pub fn remove(&mut self, handle: ResourceHandle, kind: ResourceType) -> Result<T, ResourceError> {
        let index = self.validate(handle, kind)?;
        let slot = &mut self.slots[index];
        let entry = slot.entry.take().expect("validated live slot");
        self.live -= 1;
        let max_generation = u32::MAX >> self.slot_bits;
        if slot.generation < max_generation {
            slot.generation += 1;
            // Capacity was reserved when this slot was created.
            self.free.push(index as u16);
        }
        Ok(entry.value)
    }

    /// Removes and drops a live value exactly once.
    pub fn drop_handle(&mut self, handle: ResourceHandle, kind: ResourceType) -> Result<(), ResourceError> {
        drop(self.remove(handle, kind)?);
        Ok(())
    }

    fn validate(&self, handle: ResourceHandle, kind: ResourceType) -> Result<usize, ResourceError> {
        let slot_number = handle.0 & self.slot_mask;
        let index = slot_number.checked_sub(1).ok_or(ResourceError::InvalidHandle)? as usize;
        let slot = self.slots.get(index).ok_or(ResourceError::InvalidHandle)?;
        if slot.generation != handle.0 >> self.slot_bits {
            return Err(ResourceError::InvalidHandle);
        }
        let entry = slot.entry.as_ref().ok_or(ResourceError::InvalidHandle)?;
        if entry.kind != kind {
            return Err(ResourceError::WrongType);
        }
        Ok(index)
    }
}
