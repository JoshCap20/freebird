//! `PageRank` algorithm for ranking files by cross-file reference importance.
//!
//! Implements the iterative power method with configurable damping factor,
//! convergence tolerance, and maximum iterations. Produces deterministic,
//! stable rankings from a directed adjacency list.

use std::collections::HashSet;

/// Default damping factor (probability of following an edge vs. random jump).
const DEFAULT_DAMPING: f64 = 0.85;

/// Default convergence tolerance (L1 norm between iterations).
const DEFAULT_TOLERANCE: f64 = 1e-6;

/// Default maximum iterations before stopping.
const DEFAULT_MAX_ITERATIONS: u32 = 100;

/// Run `PageRank` over a directed graph represented as an adjacency list.
///
/// Each entry `adjacency[i]` is the set of node indices that node `i` points to.
/// Returns a score vector indexed by node (same order as adjacency), where
/// higher scores indicate more heavily referenced nodes.
///
/// # Properties
///
/// - Deterministic: same input always produces the same output.
/// - Scores are non-negative and sum to approximately 1.0.
/// - Division-safe: handles dangling nodes (no outgoing edges) and empty graphs.
pub(super) fn pagerank(adjacency: &[HashSet<usize>], num_nodes: usize) -> Vec<f64> {
    pagerank_with_params(
        adjacency,
        num_nodes,
        DEFAULT_DAMPING,
        DEFAULT_MAX_ITERATIONS,
        DEFAULT_TOLERANCE,
    )
}

/// Run `PageRank` with explicit parameters.
pub(super) fn pagerank_with_params(
    adjacency: &[HashSet<usize>],
    num_nodes: usize,
    damping: f64,
    max_iterations: u32,
    tolerance: f64,
) -> Vec<f64> {
    if num_nodes == 0 {
        return Vec::new();
    }

    debug_assert!(
        adjacency.len() == num_nodes,
        "adjacency list length ({}) must equal num_nodes ({num_nodes})",
        adjacency.len(),
    );

    #[expect(
        clippy::cast_precision_loss,
        reason = "graph sizes well within f64 mantissa range"
    )]
    let n = num_nodes as f64;
    let initial = 1.0 / n;
    let mut ranks = vec![initial; num_nodes];
    let mut new_ranks = vec![0.0_f64; num_nodes];

    for _ in 0..max_iterations {
        new_ranks.fill(0.0);

        // Accumulate dangling node mass (nodes with no outgoing edges).
        let dangling_sum: f64 = adjacency
            .iter()
            .enumerate()
            .filter(|(_, neighbors)| neighbors.is_empty())
            .map(|(i, _)| ranks.get(i).copied().unwrap_or(0.0))
            .sum();

        // Distribute rank contributions.
        for (i, neighbors) in adjacency.iter().enumerate() {
            let rank_i = ranks.get(i).copied().unwrap_or(0.0);
            if neighbors.is_empty() {
                continue; // Dangling mass handled separately.
            }
            #[expect(
                clippy::cast_precision_loss,
                reason = "graph sizes well within f64 mantissa range"
            )]
            let contribution = rank_i / neighbors.len() as f64;
            for &j in neighbors {
                if let Some(r) = new_ranks.get_mut(j) {
                    *r += contribution;
                }
            }
        }

        // Apply damping: rank = (1-d)/N + d*(dangling/N + contributions).
        let base = (1.0 - damping) / n;
        let dangling_contrib = damping * dangling_sum / n;
        for r in &mut new_ranks {
            *r = damping.mul_add(*r, base + dangling_contrib);
        }

        // Check convergence (L1 norm).
        let diff: f64 = ranks
            .iter()
            .zip(new_ranks.iter())
            .map(|(old, new)| (old - new).abs())
            .sum();

        std::mem::swap(&mut ranks, &mut new_ranks);

        if diff < tolerance {
            break;
        }
    }

    ranks
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let result = pagerank(&[], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_node_no_edges() {
        let adj = vec![HashSet::new()];
        let result = pagerank(&adj, 1);
        assert_eq!(result.len(), 1);
        assert!(
            (result[0] - 1.0).abs() < 1e-6,
            "single node should have rank 1.0"
        );
    }

    #[test]
    fn test_two_nodes_one_edge_target_ranks_higher() {
        // Node 0 -> Node 1 (0 references 1).
        let adj = vec![
            HashSet::from([1]),
            HashSet::new(), // dangling
        ];
        let result = pagerank(&adj, 2);
        assert_eq!(result.len(), 2);
        assert!(
            result[1] > result[0],
            "target node should rank higher: {result:?}"
        );
    }

    #[test]
    fn test_star_graph_center_highest() {
        // Nodes 1,2,3,4 all point to node 0.
        let adj = vec![
            HashSet::new(), // center (dangling)
            HashSet::from([0]),
            HashSet::from([0]),
            HashSet::from([0]),
            HashSet::from([0]),
        ];
        let result = pagerank(&adj, 5);

        // Center should have highest rank.
        let max_rank = result
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        assert!(
            (result[0] - max_rank).abs() < 1e-10,
            "center should be highest: {result:?}"
        );
    }

    #[test]
    fn test_cycle_graph_equal_ranks() {
        // 0 -> 1 -> 2 -> 0 (symmetric cycle).
        let adj = vec![HashSet::from([1]), HashSet::from([2]), HashSet::from([0])];
        let result = pagerank(&adj, 3);

        // All ranks should be approximately equal (1/3).
        for &r in &result {
            assert!(
                (r - 1.0 / 3.0).abs() < 1e-4,
                "cycle nodes should have equal rank: {result:?}"
            );
        }
    }

    #[test]
    fn test_scores_sum_to_approximately_one() {
        let adj = vec![
            HashSet::from([1, 2]),
            HashSet::from([2]),
            HashSet::from([0]),
            HashSet::new(),
        ];
        let result = pagerank(&adj, 4);
        let sum: f64 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "scores should sum to ~1.0: sum={sum}"
        );
    }

    #[test]
    fn test_all_scores_non_negative() {
        let adj = vec![HashSet::from([1]), HashSet::from([2]), HashSet::new()];
        let result = pagerank(&adj, 3);
        for (i, &r) in result.iter().enumerate() {
            assert!(r >= 0.0, "rank[{i}] should be non-negative: {r}");
        }
    }

    #[test]
    fn test_deterministic() {
        let adj = vec![
            HashSet::from([1, 2]),
            HashSet::from([0]),
            HashSet::from([1]),
        ];
        let r1 = pagerank(&adj, 3);
        let r2 = pagerank(&adj, 3);
        assert_eq!(r1, r2, "same input should produce same output");
    }

    #[test]
    fn test_convergence_within_max_iterations() {
        // A simple graph should converge well before 100 iterations.
        let adj = vec![HashSet::from([1]), HashSet::from([0])];
        // Use a very tight tolerance to ensure we actually iterate.
        let result = pagerank_with_params(&adj, 2, 0.85, 5, 1e-15);
        assert_eq!(result.len(), 2);
        // Should still produce reasonable results even with few iterations.
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_all_dangling_nodes() {
        // No edges at all — all nodes are dangling.
        let adj = vec![HashSet::new(), HashSet::new(), HashSet::new()];
        let result = pagerank(&adj, 3);
        // All ranks should be equal (1/3).
        for &r in &result {
            assert!(
                (r - 1.0 / 3.0).abs() < 1e-6,
                "all-dangling should have equal rank: {result:?}"
            );
        }
    }

    #[test]
    fn test_heavily_referenced_node_ranks_highest() {
        // Node 0 is referenced by everyone; node 3 references everyone.
        let adj = vec![
            HashSet::new(), // dangling
            HashSet::from([0]),
            HashSet::from([0]),
            HashSet::from([0, 1, 2]),
        ];
        let result = pagerank(&adj, 4);
        // Node 0 should be the highest-ranked.
        assert!(
            result[0] > result[1] && result[0] > result[2] && result[0] > result[3],
            "node 0 should rank highest: {result:?}"
        );
    }
}
