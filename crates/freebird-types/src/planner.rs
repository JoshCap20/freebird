//! Dependency-aware change planning for multi-file edits.
//!
//! Provides Kahn's topological sort with level-based secondary sorting
//! to produce a deterministic execution order when editing multiple files.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Maximum number of planned changes allowed.
pub const MAX_CHANGES: usize = 256;

/// Maximum topological depth (longest dependency chain).
pub const MAX_DEPTH: usize = 64;

// ── Types ───────────────────────────────────────────────────────────

/// Classification of a planned change for secondary sort ordering.
///
/// **Do not reorder variants.** Derived `Ord` determines secondary sort
/// priority within each topological level. Append new variants at the end only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeKind {
    TraitDefinition,
    TypeDefinition,
    Implementation,
    Consumer,
    Test,
}

/// Classification of the crate containing the change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrateKind {
    Library,
    Binary,
}

/// A single planned file change with resolved dependencies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedChange {
    pub id: usize,
    pub file_path: PathBuf,
    pub description: String,
    pub depends_on: Vec<usize>,
    pub change_kind: ChangeKind,
    pub crate_kind: CrateKind,
}

impl PlannedChange {
    /// Sort key used for secondary ordering within a topological level.
    ///
    /// Single source of truth — used by both the algorithm and proptests.
    #[must_use]
    pub const fn secondary_sort_key(&self) -> (CrateKind, ChangeKind, &PathBuf) {
        (self.crate_kind, self.change_kind, &self.file_path)
    }
}

/// The result of planning: an ordered sequence of changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChangePlan {
    pub ordered_changes: Vec<PlannedChange>,
    pub depth: usize,
}

/// Errors from change planning.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PlanError {
    #[error("dependency cycle detected: {}", format_cycle(cycle))]
    CycleDetected { cycle: Vec<usize> },

    #[error("change {from} depends on nonexistent change {to}")]
    InvalidDependency { from: usize, to: usize },

    #[error("duplicate change id: {0}")]
    DuplicateId(usize),

    #[error("plan exceeds maximum of {max} changes (got {actual})")]
    TooManyChanges { max: usize, actual: usize },

    #[error("dependency chain depth {actual} exceeds maximum of {max}")]
    TooDeep { max: usize, actual: usize },
}

fn format_cycle(cycle: &[usize]) -> String {
    cycle
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" -> ")
}

// ── Algorithm ───────────────────────────────────────────────────────

/// Produce a dependency-aware execution order for planned changes.
///
/// Uses Kahn's algorithm (BFS topological sort) with level-based
/// secondary sorting. Within each topological level, changes are
/// sorted by `(CrateKind, ChangeKind, file_path)` for determinism.
///
/// # Errors
///
/// Returns `PlanError` if:
/// - Input contains duplicate IDs (`DuplicateId`)
/// - Input exceeds `MAX_CHANGES` (`TooManyChanges`)
/// - A dependency references a nonexistent ID (`InvalidDependency`)
/// - Dependencies form a cycle (`CycleDetected`)
/// - Dependency chain depth exceeds `MAX_DEPTH` (`TooDeep`)
pub fn plan_changes(changes: Vec<PlannedChange>) -> Result<ChangePlan, PlanError> {
    if changes.is_empty() {
        return Ok(ChangePlan {
            ordered_changes: Vec::new(),
            depth: 0,
        });
    }

    if changes.len() > MAX_CHANGES {
        return Err(PlanError::TooManyChanges {
            max: MAX_CHANGES,
            actual: changes.len(),
        });
    }

    let id_to_index = build_id_index(&changes)?;
    let n = changes.len();
    let (adjacency, mut in_degree) = build_graph(&changes, &id_to_index)?;

    // Seed Kahn's BFS with zero-in-degree nodes
    let mut queue: VecDeque<usize> = in_degree
        .iter()
        .enumerate()
        .filter(|(_, deg)| **deg == 0)
        .map(|(idx, _)| idx)
        .collect();

    let mut slots: Vec<Option<PlannedChange>> = changes.into_iter().map(Some).collect();
    let mut ordered: Vec<PlannedChange> = Vec::with_capacity(n);
    let mut depth = 0usize;

    while !queue.is_empty() {
        depth += 1;
        if depth > MAX_DEPTH {
            return Err(PlanError::TooDeep {
                max: MAX_DEPTH,
                actual: depth,
            });
        }

        let mut level: Vec<usize> = queue.into_iter().collect();
        queue = VecDeque::new();

        sort_level_by_secondary_key(&mut level, &slots);
        enqueue_next_level(&level, &adjacency, &mut in_degree, &mut queue);
        drain_level_into_ordered(level, &mut slots, &mut ordered);
    }

    if ordered.len() < n {
        let cycle = extract_cycle(&slots, &in_degree, &adjacency);
        return Err(PlanError::CycleDetected { cycle });
    }

    Ok(ChangePlan {
        ordered_changes: ordered,
        depth,
    })
}

fn build_id_index(changes: &[PlannedChange]) -> Result<HashMap<usize, usize>, PlanError> {
    let mut id_to_index: HashMap<usize, usize> = HashMap::with_capacity(changes.len());
    for (idx, change) in changes.iter().enumerate() {
        if id_to_index.insert(change.id, idx).is_some() {
            return Err(PlanError::DuplicateId(change.id));
        }
    }
    Ok(id_to_index)
}

fn build_graph(
    changes: &[PlannedChange],
    id_to_index: &HashMap<usize, usize>,
) -> Result<(Vec<Vec<usize>>, Vec<usize>), PlanError> {
    let n = changes.len();
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_degree: Vec<usize> = vec![0; n];

    for (idx, change) in changes.iter().enumerate() {
        // Deduplicate dependencies to prevent in-degree corruption.
        // Duplicate entries would increment in-degree multiple times for the
        // same logical edge, potentially preventing nodes from ever entering
        // the BFS queue.
        let mut seen_deps = BTreeSet::new();
        for &dep_id in &change.depends_on {
            if !seen_deps.insert(dep_id) {
                continue; // skip duplicate
            }

            let dep_idx =
                id_to_index
                    .get(&dep_id)
                    .copied()
                    .ok_or(PlanError::InvalidDependency {
                        from: change.id,
                        to: dep_id,
                    })?;

            if dep_idx == idx {
                // Self-loop: normalized to [id, id] — same format as
                // multi-node cycles [a, ..., a] (first == last).
                return Err(PlanError::CycleDetected {
                    cycle: vec![change.id, change.id],
                });
            }

            if let Some(adj) = adjacency.get_mut(dep_idx) {
                adj.push(idx);
            }
            if let Some(deg) = in_degree.get_mut(idx) {
                *deg += 1;
            }
        }
    }
    Ok((adjacency, in_degree))
}

fn sort_level_by_secondary_key(level: &mut [usize], slots: &[Option<PlannedChange>]) {
    level.sort_by(|&a, &b| {
        let ca = slots.get(a).and_then(Option::as_ref);
        let cb = slots.get(b).and_then(Option::as_ref);
        match (ca, cb) {
            (Some(ca), Some(cb)) => ca.secondary_sort_key().cmp(&cb.secondary_sort_key()),
            _ => std::cmp::Ordering::Equal,
        }
    });
}

fn enqueue_next_level(
    level: &[usize],
    adjacency: &[Vec<usize>],
    in_degree: &mut [usize],
    queue: &mut VecDeque<usize>,
) {
    for &idx in level {
        if let Some(neighbors) = adjacency.get(idx) {
            for &neighbor in neighbors {
                if let Some(deg) = in_degree.get_mut(neighbor) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
    }
}

fn drain_level_into_ordered(
    level: Vec<usize>,
    slots: &mut [Option<PlannedChange>],
    ordered: &mut Vec<PlannedChange>,
) {
    for idx in level {
        if let Some(change) = slots.get_mut(idx).and_then(Option::take) {
            ordered.push(change);
        }
    }
}

/// Extract one cycle from the residual graph using DFS on the adjacency list.
///
/// After Kahn's BFS completes, any nodes still in the graph (`in_degree` > 0) are
/// part of at least one cycle. This function walks the *adjacency list* (not
/// `PlannedChange.depends_on`) to find a cycle, which is more reliable than the
/// depends_on-based walk because the adjacency list was already deduplicated and
/// validated during graph construction.
fn extract_cycle(
    slots: &[Option<PlannedChange>],
    in_degree: &[usize],
    adjacency: &[Vec<usize>],
) -> Vec<usize> {
    // Build residual-only forward edges (node → its dependencies still in graph).
    // adjacency[dep] contains dependents, so we reverse: dependent → dep.
    let n = in_degree.len();
    let mut forward: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (dep_idx, dependents) in adjacency.iter().enumerate() {
        if in_degree.get(dep_idx).is_some_and(|&d| d > 0) {
            for &dependent in dependents {
                if in_degree.get(dependent).is_some_and(|&d| d > 0) {
                    if let Some(fwd) = forward.get_mut(dependent) {
                        fwd.push(dep_idx);
                    }
                }
            }
        }
    }

    // Try DFS from each residual node
    let mut visited = vec![false; n];

    for (start, &deg) in in_degree.iter().enumerate() {
        if deg == 0 || visited.get(start).is_some_and(|&v| v) {
            continue;
        }
        if let Some(cycle) = dfs_find_cycle(start, &forward, &mut visited, slots) {
            return cycle;
        }
    }

    // Fallback: should not happen if Kahn's BFS correctly identifies residual nodes
    Vec::new()
}

/// Map a graph index to its change ID (falls back to the index itself).
fn node_id(slots: &[Option<PlannedChange>], idx: usize) -> usize {
    slots
        .get(idx)
        .and_then(Option::as_ref)
        .map_or(idx, |c| c.id)
}

/// DFS from `start` through `forward` edges to find a cycle.
///
/// Returns the cycle as a vec of change IDs `[a, b, ..., a]` if found.
fn dfs_find_cycle(
    start: usize,
    forward: &[Vec<usize>],
    visited: &mut [bool],
    slots: &[Option<PlannedChange>],
) -> Option<Vec<usize>> {
    let n = forward.len();
    let mut on_stack = vec![false; n];
    // Stack entries: (node, neighbor_cursor)
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut path: Vec<usize> = Vec::new();

    if let Some(v) = visited.get_mut(start) {
        *v = true;
    }
    if let Some(v) = on_stack.get_mut(start) {
        *v = true;
    }
    path.push(node_id(slots, start));
    stack.push((start, 0));

    while let Some(&mut (node, ref mut cursor)) = stack.last_mut() {
        let neighbors = forward.get(node).map_or(&[] as &[usize], Vec::as_slice);
        if *cursor < neighbors.len() {
            let Some(&next) = neighbors.get(*cursor) else {
                break;
            };
            *cursor += 1;

            if on_stack.get(next).is_some_and(|&v| v) {
                // Found a cycle — extract from path
                let next_id = node_id(slots, next);
                if let Some(pos) = path.iter().position(|&id| id == next_id) {
                    let mut cycle: Vec<usize> =
                        path.get(pos..).map_or_else(Vec::new, <[usize]>::to_vec);
                    cycle.push(next_id);
                    return Some(cycle);
                }
            }

            if !visited.get(next).is_some_and(|&v| v) {
                if let Some(v) = visited.get_mut(next) {
                    *v = true;
                }
                if let Some(v) = on_stack.get_mut(next) {
                    *v = true;
                }
                path.push(node_id(slots, next));
                stack.push((next, 0));
            }
        } else {
            // Backtrack — `node` is a copy (usize), safe to use after pop.
            stack.pop();
            path.pop();
            if let Some(v) = on_stack.get_mut(node) {
                *v = false;
            }
        }
    }

    None
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn change(
        id: usize,
        path: &str,
        deps: Vec<usize>,
        kind: ChangeKind,
        crate_kind: CrateKind,
    ) -> PlannedChange {
        PlannedChange {
            id,
            file_path: PathBuf::from(path),
            description: format!("change {id}"),
            depends_on: deps,
            change_kind: kind,
            crate_kind,
        }
    }

    fn lib_consumer(id: usize, path: &str, deps: Vec<usize>) -> PlannedChange {
        change(id, path, deps, ChangeKind::Consumer, CrateKind::Library)
    }

    // ── Core correctness ────────────────────────────────────────────

    #[test]
    fn test_empty_plan() {
        let result = plan_changes(vec![]).unwrap();
        assert_eq!(result.ordered_changes.len(), 0);
        assert_eq!(result.depth, 0);
    }

    #[test]
    fn test_single_change() {
        let c = lib_consumer(0, "a.rs", vec![]);
        let result = plan_changes(vec![c.clone()]).unwrap();
        assert_eq!(result.ordered_changes.len(), 1);
        assert_eq!(result.ordered_changes[0], c);
        assert_eq!(result.depth, 1);
    }

    #[test]
    fn test_linear_chain() {
        let a = lib_consumer(0, "a.rs", vec![1]);
        let b = lib_consumer(1, "b.rs", vec![2]);
        let c = lib_consumer(2, "c.rs", vec![]);
        let result = plan_changes(vec![a, b, c]).unwrap();
        let ids: Vec<usize> = result.ordered_changes.iter().map(|c| c.id).collect();
        assert_eq!(ids, vec![2, 1, 0]);
        assert_eq!(result.depth, 3);
    }

    #[test]
    fn test_diamond_dependency() {
        let a = lib_consumer(0, "a.rs", vec![1, 2]);
        let b = lib_consumer(1, "b.rs", vec![3]);
        let c = lib_consumer(2, "c.rs", vec![3]);
        let d = lib_consumer(3, "d.rs", vec![]);
        let result = plan_changes(vec![a, b, c, d]).unwrap();
        let ids: Vec<usize> = result.ordered_changes.iter().map(|c| c.id).collect();
        assert_eq!(ids[0], 3);
        assert!(ids[1..3].contains(&1));
        assert!(ids[1..3].contains(&2));
        assert_eq!(ids[3], 0);
        assert_eq!(result.depth, 3);
    }

    #[test]
    fn test_wide_fan_out() {
        let mut changes = vec![lib_consumer(0, "root.rs", vec![])];
        for i in 1..=5 {
            changes.push(lib_consumer(i, &format!("dep_{i}.rs"), vec![0]));
        }
        let result = plan_changes(changes).unwrap();
        assert_eq!(result.ordered_changes[0].id, 0);
        assert_eq!(result.depth, 2);
    }

    #[test]
    fn test_wide_fan_in() {
        let mut changes: Vec<PlannedChange> = (0..5)
            .map(|i| lib_consumer(i, &format!("root_{i}.rs"), vec![]))
            .collect();
        changes.push(lib_consumer(5, "sink.rs", vec![0, 1, 2, 3, 4]));
        let result = plan_changes(changes).unwrap();
        assert_eq!(result.ordered_changes.last().unwrap().id, 5);
        assert_eq!(result.depth, 2);
    }

    #[test]
    fn test_disconnected_components() {
        let a = lib_consumer(0, "a.rs", vec![1]);
        let b = lib_consumer(1, "b.rs", vec![]);
        let c = lib_consumer(2, "c.rs", vec![3]);
        let d = lib_consumer(3, "d.rs", vec![]);
        let result = plan_changes(vec![a, b, c, d]).unwrap();
        let ids: Vec<usize> = result.ordered_changes.iter().map(|c| c.id).collect();
        assert!(
            ids.iter().position(|&x| x == 1).unwrap() < ids.iter().position(|&x| x == 0).unwrap()
        );
        assert!(
            ids.iter().position(|&x| x == 3).unwrap() < ids.iter().position(|&x| x == 2).unwrap()
        );
        assert_eq!(ids.len(), 4);
    }

    // ── Secondary sort ──────────────────────────────────────────────

    #[test]
    fn test_trait_before_impl_same_level() {
        let t = change(
            0,
            "trait.rs",
            vec![],
            ChangeKind::TraitDefinition,
            CrateKind::Library,
        );
        let i = change(
            1,
            "impl.rs",
            vec![],
            ChangeKind::Implementation,
            CrateKind::Library,
        );
        let result = plan_changes(vec![i, t]).unwrap();
        assert_eq!(result.ordered_changes[0].id, 0);
        assert_eq!(result.ordered_changes[1].id, 1);
    }

    #[test]
    fn test_type_before_consumer_same_level() {
        let t = change(
            0,
            "type.rs",
            vec![],
            ChangeKind::TypeDefinition,
            CrateKind::Library,
        );
        let c = change(
            1,
            "consumer.rs",
            vec![],
            ChangeKind::Consumer,
            CrateKind::Library,
        );
        let result = plan_changes(vec![c, t]).unwrap();
        assert_eq!(result.ordered_changes[0].id, 0);
        assert_eq!(result.ordered_changes[1].id, 1);
    }

    #[test]
    fn test_library_before_binary_same_level() {
        let l = change(
            0,
            "lib.rs",
            vec![],
            ChangeKind::Consumer,
            CrateKind::Library,
        );
        let b = change(1, "bin.rs", vec![], ChangeKind::Consumer, CrateKind::Binary);
        let result = plan_changes(vec![b, l]).unwrap();
        assert_eq!(result.ordered_changes[0].id, 0);
        assert_eq!(result.ordered_changes[1].id, 1);
    }

    #[test]
    fn test_alphabetical_tiebreak() {
        let a = change(0, "b.rs", vec![], ChangeKind::Consumer, CrateKind::Library);
        let b = change(1, "a.rs", vec![], ChangeKind::Consumer, CrateKind::Library);
        let result = plan_changes(vec![a, b]).unwrap();
        assert_eq!(result.ordered_changes[0].id, 1);
        assert_eq!(result.ordered_changes[1].id, 0);
    }

    #[test]
    fn test_full_secondary_sort_cascade() {
        let changes = vec![
            change(0, "test.rs", vec![], ChangeKind::Test, CrateKind::Library),
            change(
                1,
                "consumer.rs",
                vec![],
                ChangeKind::Consumer,
                CrateKind::Library,
            ),
            change(
                2,
                "impl.rs",
                vec![],
                ChangeKind::Implementation,
                CrateKind::Library,
            ),
            change(
                3,
                "type.rs",
                vec![],
                ChangeKind::TypeDefinition,
                CrateKind::Library,
            ),
            change(
                4,
                "trait.rs",
                vec![],
                ChangeKind::TraitDefinition,
                CrateKind::Library,
            ),
        ];
        let result = plan_changes(changes).unwrap();
        let ids: Vec<usize> = result.ordered_changes.iter().map(|c| c.id).collect();
        assert_eq!(ids, vec![4, 3, 2, 1, 0]);
    }

    // ── Error cases ─────────────────────────────────────────────────

    #[test]
    fn test_cycle_two_nodes() {
        let a = lib_consumer(0, "a.rs", vec![1]);
        let b = lib_consumer(1, "b.rs", vec![0]);
        let err = plan_changes(vec![a, b]).unwrap_err();
        assert!(matches!(err, PlanError::CycleDetected { .. }));
    }

    #[test]
    fn test_cycle_three_nodes() {
        let a = lib_consumer(0, "a.rs", vec![1]);
        let b = lib_consumer(1, "b.rs", vec![2]);
        let c = lib_consumer(2, "c.rs", vec![0]);
        let err = plan_changes(vec![a, b, c]).unwrap_err();
        assert!(matches!(err, PlanError::CycleDetected { .. }));
    }

    #[test]
    fn test_self_dependency() {
        let a = lib_consumer(0, "a.rs", vec![0]);
        let err = plan_changes(vec![a]).unwrap_err();
        assert!(matches!(err, PlanError::CycleDetected { .. }));
    }

    #[test]
    fn test_invalid_dependency_reference() {
        let a = lib_consumer(0, "a.rs", vec![99]);
        let err = plan_changes(vec![a]).unwrap_err();
        assert!(matches!(
            err,
            PlanError::InvalidDependency { from: 0, to: 99 }
        ));
    }

    #[test]
    fn test_duplicate_ids() {
        let a = lib_consumer(0, "a.rs", vec![]);
        let b = lib_consumer(0, "b.rs", vec![]);
        let err = plan_changes(vec![a, b]).unwrap_err();
        assert!(matches!(err, PlanError::DuplicateId(0)));
    }

    #[test]
    fn test_exceeds_max_changes() {
        let changes: Vec<PlannedChange> = (0..=MAX_CHANGES)
            .map(|i| lib_consumer(i, &format!("{i}.rs"), vec![]))
            .collect();
        let err = plan_changes(changes).unwrap_err();
        assert!(matches!(
            err,
            PlanError::TooManyChanges {
                max: 256,
                actual: 257
            }
        ));
    }

    #[test]
    fn test_exceeds_max_depth() {
        let n = MAX_DEPTH + 1;
        let changes: Vec<PlannedChange> = (0..n)
            .map(|i| {
                let deps = if i == 0 { vec![] } else { vec![i - 1] };
                lib_consumer(i, &format!("{i}.rs"), deps)
            })
            .collect();
        let err = plan_changes(changes).unwrap_err();
        assert!(matches!(
            err,
            PlanError::TooDeep {
                max: 64,
                actual: 65
            }
        ));
    }

    #[test]
    fn test_duplicate_deps_handled_correctly() {
        // depends_on: [1, 1] should behave the same as [1]
        let a = lib_consumer(0, "a.rs", vec![1, 1]);
        let b = lib_consumer(1, "b.rs", vec![]);
        let result = plan_changes(vec![a, b]).unwrap();
        let ids: Vec<usize> = result.ordered_changes.iter().map(|c| c.id).collect();
        assert_eq!(ids, vec![1, 0]);
    }

    // ── Determinism + Scale ─────────────────────────────────────────

    #[test]
    fn test_deterministic_output() {
        let make_changes = || {
            vec![
                change(0, "z.rs", vec![2], ChangeKind::Test, CrateKind::Binary),
                change(1, "a.rs", vec![2], ChangeKind::Consumer, CrateKind::Library),
                change(
                    2,
                    "m.rs",
                    vec![],
                    ChangeKind::TraitDefinition,
                    CrateKind::Library,
                ),
                change(
                    3,
                    "b.rs",
                    vec![2],
                    ChangeKind::Implementation,
                    CrateKind::Library,
                ),
            ]
        };
        let first = plan_changes(make_changes()).unwrap();
        for _ in 0..100 {
            let run = plan_changes(make_changes()).unwrap();
            assert_eq!(first, run);
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_twenty_changes_realistic() {
        let changes = vec![
            change(
                0,
                "traits/provider.rs",
                vec![],
                ChangeKind::TraitDefinition,
                CrateKind::Library,
            ),
            change(
                1,
                "traits/channel.rs",
                vec![],
                ChangeKind::TraitDefinition,
                CrateKind::Library,
            ),
            change(
                2,
                "types/config.rs",
                vec![0, 1],
                ChangeKind::TypeDefinition,
                CrateKind::Library,
            ),
            change(
                3,
                "types/protocol.rs",
                vec![0],
                ChangeKind::TypeDefinition,
                CrateKind::Library,
            ),
            change(
                4,
                "security/taint.rs",
                vec![2],
                ChangeKind::TypeDefinition,
                CrateKind::Library,
            ),
            change(
                5,
                "providers/anthropic.rs",
                vec![0, 2, 3],
                ChangeKind::Implementation,
                CrateKind::Library,
            ),
            change(
                6,
                "channels/tcp.rs",
                vec![1, 2],
                ChangeKind::Implementation,
                CrateKind::Library,
            ),
            change(
                7,
                "runtime/agent.rs",
                vec![0, 1, 2, 4],
                ChangeKind::Consumer,
                CrateKind::Library,
            ),
            change(
                8,
                "runtime/executor.rs",
                vec![0, 4],
                ChangeKind::Consumer,
                CrateKind::Library,
            ),
            change(
                9,
                "core/builder.rs",
                vec![5, 6, 7],
                ChangeKind::Consumer,
                CrateKind::Library,
            ),
            change(
                10,
                "daemon/main.rs",
                vec![9],
                ChangeKind::Consumer,
                CrateKind::Binary,
            ),
            change(
                11,
                "tests/provider_test.rs",
                vec![5],
                ChangeKind::Test,
                CrateKind::Library,
            ),
            change(
                12,
                "tests/channel_test.rs",
                vec![6],
                ChangeKind::Test,
                CrateKind::Library,
            ),
            change(
                13,
                "tests/agent_test.rs",
                vec![7],
                ChangeKind::Test,
                CrateKind::Library,
            ),
            change(
                14,
                "tests/executor_test.rs",
                vec![8],
                ChangeKind::Test,
                CrateKind::Library,
            ),
            change(
                15,
                "tests/builder_test.rs",
                vec![9],
                ChangeKind::Test,
                CrateKind::Library,
            ),
            change(
                16,
                "tests/integration.rs",
                vec![10],
                ChangeKind::Test,
                CrateKind::Binary,
            ),
            change(
                17,
                "docs/api.rs",
                vec![0, 1],
                ChangeKind::Consumer,
                CrateKind::Library,
            ),
            change(
                18,
                "security/capability.rs",
                vec![4],
                ChangeKind::TypeDefinition,
                CrateKind::Library,
            ),
            change(
                19,
                "runtime/session.rs",
                vec![0, 18],
                ChangeKind::Consumer,
                CrateKind::Library,
            ),
        ];
        let result = plan_changes(changes).unwrap();
        assert_eq!(result.ordered_changes.len(), 20);

        let mut position: HashMap<usize, usize> = HashMap::new();
        for (pos, c) in result.ordered_changes.iter().enumerate() {
            position.insert(c.id, pos);
        }
        for c in &result.ordered_changes {
            for dep in &c.depends_on {
                assert!(
                    position[dep] < position[&c.id],
                    "dependency {dep} should come before {}",
                    c.id
                );
            }
        }
    }

    // ── Serde roundtrip ─────────────────────────────────────────────

    #[test]
    fn test_change_kind_serde_roundtrip() {
        for kind in [
            ChangeKind::TraitDefinition,
            ChangeKind::TypeDefinition,
            ChangeKind::Implementation,
            ChangeKind::Consumer,
            ChangeKind::Test,
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let back: ChangeKind = serde_json::from_str(&json).unwrap();
            assert_eq!(kind, back);
        }
    }

    #[test]
    fn test_crate_kind_serde_roundtrip() {
        for kind in [CrateKind::Library, CrateKind::Binary] {
            let json = serde_json::to_string(&kind).unwrap();
            let back: CrateKind = serde_json::from_str(&json).unwrap();
            assert_eq!(kind, back);
        }
    }

    #[test]
    fn test_plan_error_display() {
        let err = PlanError::CycleDetected {
            cycle: vec![1, 2, 3, 1],
        };
        assert_eq!(
            err.to_string(),
            "dependency cycle detected: 1 -> 2 -> 3 -> 1"
        );

        let err = PlanError::InvalidDependency { from: 5, to: 99 };
        assert_eq!(err.to_string(), "change 5 depends on nonexistent change 99");

        let err = PlanError::DuplicateId(7);
        assert_eq!(err.to_string(), "duplicate change id: 7");
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::redundant_clone
)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_change_kind() -> impl Strategy<Value = ChangeKind> {
        prop_oneof![
            Just(ChangeKind::TraitDefinition),
            Just(ChangeKind::TypeDefinition),
            Just(ChangeKind::Implementation),
            Just(ChangeKind::Consumer),
            Just(ChangeKind::Test),
        ]
    }

    fn arb_crate_kind() -> impl Strategy<Value = CrateKind> {
        prop_oneof![Just(CrateKind::Library), Just(CrateKind::Binary),]
    }

    /// Generate a valid DAG of `PlannedChange` values (no cycles, unique IDs).
    fn arb_dag(max_nodes: usize) -> impl Strategy<Value = Vec<PlannedChange>> {
        (1..=max_nodes)
            .prop_flat_map(|n| {
                let kinds = proptest::collection::vec(arb_change_kind(), n);
                let crate_kinds = proptest::collection::vec(arb_crate_kind(), n);
                (Just(n), kinds, crate_kinds)
            })
            .prop_map(|(n, kinds, crate_kinds)| {
                let mut changes = Vec::with_capacity(n);
                for i in 0..n {
                    let max_deps = i.min(3);
                    let mut deps = Vec::new();
                    for j in 0..max_deps {
                        if (i + j) % 3 == 0 {
                            deps.push(j);
                        }
                    }
                    changes.push(PlannedChange {
                        id: i,
                        file_path: PathBuf::from(format!("file_{i}.rs")),
                        description: format!("change {i}"),
                        depends_on: deps,
                        change_kind: kinds[i],
                        crate_kind: crate_kinds[i],
                    });
                }
                changes
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_output_is_permutation(changes in arb_dag(20)) {
            let input_ids: BTreeSet<usize> = changes.iter().map(|c| c.id).collect();
            let result = plan_changes(changes).unwrap();
            let output_ids: BTreeSet<usize> = result.ordered_changes.iter().map(|c| c.id).collect();
            prop_assert_eq!(input_ids, output_ids);
        }

        #[test]
        fn prop_dependencies_satisfied(changes in arb_dag(20)) {
            let result = plan_changes(changes).unwrap();
            let mut position: HashMap<usize, usize> = HashMap::new();
            for (pos, c) in result.ordered_changes.iter().enumerate() {
                position.insert(c.id, pos);
            }
            for c in &result.ordered_changes {
                for dep in &c.depends_on {
                    prop_assert!(
                        position[dep] < position[&c.id],
                        "dep {} at pos {} should be before {} at pos {}",
                        dep, position[dep], c.id, position[&c.id]
                    );
                }
            }
        }

        #[test]
        fn prop_deterministic(changes in arb_dag(20)) {
            let first = plan_changes(changes.clone()).unwrap();
            let second = plan_changes(changes).unwrap();
            prop_assert_eq!(first, second);
        }

        #[test]
        fn prop_secondary_sort_within_level(changes in arb_dag(15)) {
            let result = plan_changes(changes).unwrap();
            let mut position: HashMap<usize, usize> = HashMap::new();
            for (pos, c) in result.ordered_changes.iter().enumerate() {
                position.insert(c.id, pos);
            }

            for window in result.ordered_changes.windows(2) {
                let a = &window[0];
                let b = &window[1];
                let a_max_dep = a.depends_on.iter().filter_map(|d| position.get(d)).max().copied().unwrap_or(0);
                let b_max_dep = b.depends_on.iter().filter_map(|d| position.get(d)).max().copied().unwrap_or(0);
                if a_max_dep == b_max_dep && a.depends_on.len() == b.depends_on.len() {
                    let a_key = a.secondary_sort_key();
                    let b_key = b.secondary_sort_key();
                    prop_assert!(a_key <= b_key, "secondary sort violated: {:?} > {:?}", a_key, b_key);
                }
            }
        }
    }
}
