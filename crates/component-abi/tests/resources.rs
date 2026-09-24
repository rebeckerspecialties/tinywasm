use std::cell::Cell;
use std::collections::BTreeMap;
use std::num::NonZeroU16;
use std::rc::Rc;

use tinywasm::{FuncContext, HostFunction, Imports, ModuleInstance, Store};
use tinywasm_component_abi::resources::{ResourceError, ResourceHandle, ResourceTable, ResourceType};

const NODE: ResourceType = ResourceType::new(1);
const CONTEXT: ResourceType = ResourceType::new(2);

fn table<T>(max: u16) -> ResourceTable<T> {
    ResourceTable::new(NonZeroU16::new(max).unwrap())
}

#[test]
fn rejects_invalid_wrong_type_dropped_and_reused_handles() {
    let mut table = table(2);
    let first = table.insert(NODE, 41_u32).unwrap();
    assert_ne!(first.into_raw(), 0);
    assert_eq!(table.get(first, NODE).unwrap(), &41);
    assert!(matches!(table.get(first, CONTEXT), Err(ResourceError::WrongType)));
    assert!(matches!(table.remove(first, CONTEXT), Err(ResourceError::WrongType)));
    for raw in [0, u32::MAX, first.into_raw() + 2] {
        assert!(matches!(table.get(ResourceHandle::from_raw(raw), NODE), Err(ResourceError::InvalidHandle)));
    }
    *table.get_mut(first, NODE).unwrap() = 42;
    assert_eq!(table.remove(first, NODE).unwrap(), 42);
    assert!(matches!(table.remove(first, NODE), Err(ResourceError::InvalidHandle)));

    let second = table.insert(NODE, 43).unwrap();
    assert_ne!(first, second);
    assert!(matches!(table.get(first, NODE), Err(ResourceError::InvalidHandle)));
    assert_eq!(table.get(second, NODE).unwrap(), &43);
}

struct Probe(Rc<Cell<usize>>);

impl Drop for Probe {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

#[test]
fn full_table_failed_insert_double_drop_and_teardown_release_once() {
    let drops = Rc::new(Cell::new(0));
    let mut table = table(2);
    let first = table.insert(NODE, Probe(drops.clone())).unwrap();
    let _second = table.insert(CONTEXT, Probe(drops.clone())).unwrap();
    assert_eq!(table.len(), 2);
    assert_eq!(table.insert(NODE, Probe(drops.clone())).err(), Some(ResourceError::Full));
    assert_eq!(drops.get(), 1, "rejected native value must be released");

    assert_eq!(table.drop_handle(first, NODE), Ok(()));
    assert_eq!(drops.get(), 2);
    assert_eq!(table.drop_handle(first, NODE), Err(ResourceError::InvalidHandle));
    assert_eq!(drops.get(), 2);
    drop(table);
    assert_eq!(drops.get(), 3, "live value must be released at table teardown");
}

#[test]
fn bounded_reuse_and_adversarial_guest_words() {
    let mut table = table(4);
    let mut old = Vec::new();
    for value in 0..20_000 {
        let handle = table.insert(NODE, value).unwrap();
        assert_eq!(*table.get(handle, NODE).unwrap(), value);
        table.drop_handle(handle, NODE).unwrap();
        old.push(handle);
        if old.len() > 32 {
            old.remove(0);
        }
        for &stale in &old {
            assert!(matches!(table.get(stale, NODE), Err(ResourceError::InvalidHandle)));
        }
    }
    assert!(table.is_empty());

    let mut random = 0x9e37_79b9_u32;
    for _ in 0..20_000 {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        assert!(matches!(table.get(ResourceHandle::from_raw(random), NODE), Err(ResourceError::InvalidHandle)));
    }
}

#[test]
fn exhausted_generations_retire_slots_instead_of_reviving_stale_handles() {
    let mut table = table(u16::MAX);
    let first = table.insert(NODE, 1_u32).unwrap();
    table.drop_handle(first, NODE).unwrap();
    for _ in 0..u16::MAX {
        let handle = table.insert(NODE, 2).unwrap();
        table.drop_handle(handle, NODE).unwrap();
    }
    let next = table.insert(NODE, 3).unwrap();
    assert_ne!(first, next);
    assert_eq!(table.get(first, NODE).err(), Some(ResourceError::InvalidHandle));
    assert_eq!(*table.get(next, NODE).unwrap(), 3);
}

#[test]
fn mixed_resource_types_track_a_small_reference_model() {
    let mut table = table(8);
    let mut model = BTreeMap::new();
    let mut random = 0x1234_5678_u32;
    for _ in 0..10_000 {
        random ^= random << 13;
        random ^= random >> 17;
        random ^= random << 5;
        if random & 3 == 0 && !model.is_empty() {
            let (&raw, &(kind, value)) = model.iter().next().unwrap();
            let handle = ResourceHandle::from_raw(raw);
            assert_eq!(table.get(handle, kind).unwrap(), &value);
            assert_eq!(table.remove(handle, kind).unwrap(), value);
            model.remove(&raw);
            assert!(matches!(table.get(handle, kind), Err(ResourceError::InvalidHandle)));
        } else if model.len() < 8 {
            let kind = if random & 1 == 0 { NODE } else { CONTEXT };
            let handle = table.insert(kind, random).unwrap();
            assert!(model.insert(handle.into_raw(), (kind, random)).is_none());
        }
        assert_eq!(table.len(), model.len());
    }
}

struct HostState {
    resources: ResourceTable<u32>,
}

#[test]
fn wit_shaped_core_imports_use_a_store_local_table() -> tinywasm::Result<()> {
    let wasm = wat::parse_str(
        r#"(module
            (import "snqr:example/nodes@0.1.0" "new" (func $new (result i32)))
            (import "snqr:example/nodes@0.1.0" "value" (func $value (param i32) (result i32)))
            (import "snqr:example/nodes@0.1.0" "drop" (func $drop (param i32)))
            (func (export "read_raw") (param i32) (result i32)
                local.get 0
                call $value)
            (func (export "run") (result i32)
                (local i32)
                call $new
                local.tee 0
                call $value
                local.get 0
                call $drop))"#,
    )
    .unwrap();
    let module = tinywasm::parse_bytes(&wasm)?;
    let mut imports = Imports::new();
    imports.define(
        "snqr:example/nodes@0.1.0",
        "new",
        HostFunction::from(|mut ctx: FuncContext<'_>, (): ()| -> tinywasm::Result<i32> {
            let table = &mut ctx.state_mut::<HostState>().unwrap().resources;
            Ok(table.insert(NODE, 7)?.into_raw() as i32)
        }),
    );
    imports.define(
        "snqr:example/nodes@0.1.0",
        "value",
        HostFunction::from(|ctx: FuncContext<'_>, raw: i32| -> tinywasm::Result<i32> {
            let table = &ctx.state::<HostState>().unwrap().resources;
            Ok(*table.get(ResourceHandle::from_raw(raw as u32), NODE)? as i32)
        }),
    );
    imports.define(
        "snqr:example/nodes@0.1.0",
        "drop",
        HostFunction::from(|mut ctx: FuncContext<'_>, raw: i32| -> tinywasm::Result<()> {
            ctx.state_mut::<HostState>().unwrap().resources.drop_handle(ResourceHandle::from_raw(raw as u32), NODE)?;
            Ok(())
        }),
    );
    let mut store = Store::default().with_state(HostState { resources: table(8) });
    let instance = ModuleInstance::instantiate(&mut store, &module, Some(&imports))?;
    let read_raw = instance.func::<i32, i32>(&store, "read_raw")?;
    assert!(read_raw.call(&mut store, -1).is_err());
    let wrong_type = store.state_mut::<HostState>().unwrap().resources.insert(CONTEXT, 11)?.into_raw();
    assert!(read_raw.call(&mut store, wrong_type as i32).is_err());
    store.state_mut::<HostState>().unwrap().resources.drop_handle(ResourceHandle::from_raw(wrong_type), CONTEXT)?;
    assert_eq!(instance.func::<(), i32>(&store, "run")?.call(&mut store, ())?, 7);
    assert!(store.state::<HostState>().unwrap().resources.is_empty());
    Ok(())
}

#[test]
fn table_can_move_to_a_separate_worker_thread() {
    let mut table = table(2);
    let handle = table.insert(NODE, 7_u32).unwrap();
    let count = std::thread::spawn(move || {
        assert_eq!(*table.get(handle, NODE).unwrap(), 7);
        table.len()
    })
    .join()
    .unwrap();
    assert_eq!(count, 1);
}
