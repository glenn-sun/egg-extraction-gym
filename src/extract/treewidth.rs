use csv::Writer;
use std::collections::{BTreeSet, VecDeque};
use std::error::Error;
use std::hash::{BuildHasherDefault, Hash, Hasher};

use super::*;
use arboretum_td::graph::{HashMapGraph, MutableGraph};
use arboretum_td::solver::Solver;
use arboretum_td::tree_decomposition::TreeDecomposition;

use nohash_hasher::{IntMap, IntSet, NoHashHasher};

mod id_converter;
use id_converter::IdConverter;

mod circuit;
use circuit::{Circuit, Gate};

mod td_tools;
use td_tools::{to_nice_decomp, forget_until_root};

mod env;
use env::{Env, SimplifyOptions};

mod summary;
use summary::{Summary, Evaluation, KnownTag};

pub struct TreewidthExtractor;

const VERBOSE: bool = true;
const SAVE_COSMOGRAPH: bool = true;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WhichChild {
    Child1,
    Child2,
}

#[derive(Debug, Default)]
struct Trace {
    cost: Cost,
    neighbor_eval: Evaluation,
    summary: Option<Summary>,
    summary2: Option<Summary>,
}

impl Trace {
    fn for_insert(cost: Cost, neighbor_eval: Evaluation, summary: Summary) -> Trace {
        Trace {
            cost, 
            neighbor_eval,
            summary: Some(summary),
            summary2: None,
        }
    }

    fn for_forget(cost: Cost, neighbor_eval: Evaluation, summary: Summary) -> Trace {
        Trace {
            cost,
            neighbor_eval,
            summary: Some(summary),
            summary2: None,
        }
    }

    fn for_join(cost: Cost, neighbor_eval: Evaluation, summary1: Summary, summary2: Summary) -> Trace {
        Trace {
            cost,
            neighbor_eval,
            summary: Some(summary1),
            summary2: Some(summary2),
        }
    }
}

type DPMapByGroup = FxHashMap<Evaluation, FxHashMap<Summary, Trace>>;

#[derive(Debug)]
enum NiceBag {
    Leaf {
        vertices: IntSet<usize>,
        map: DPMapByGroup,
    },
    Insert {
        vertices: IntSet<usize>,
        child: Box<NiceBag>,
        x: usize,
        map: DPMapByGroup,
    },
    Forget {
        vertices: IntSet<usize>,
        child: Box<NiceBag>,
        x: usize,
        map: DPMapByGroup,
    },
    Join {
        vertices: IntSet<usize>,
        child1: Box<NiceBag>,
        child2: Box<NiceBag>,
        map: DPMapByGroup,
    },
}

impl NiceBag {
    fn get_dpmap_by_group(&self) -> &DPMapByGroup {
        match self {
            NiceBag::Leaf {vertices: _, map} => map,
            NiceBag::Insert {vertices: _, child: _, x: _, map} => map,
            NiceBag::Forget {vertices: _, child: _, x: _, map } => map,
            NiceBag::Join {vertices: _, child1: _, child2: _, map} => map,
        }
    }

    /// See write-up for algorithmic details and proof.
    fn compute_dpmap_by_group(&mut self, env: &Env) {
        fn insert_if_better(map: &mut DPMapByGroup, summary: Summary, trace: Trace) {
            let dpmap = map.entry(summary.bag_eval.clone()).or_default();

            if let Some(old_trace) = dpmap.get(&summary) {
                if trace.cost < old_trace.cost {
                    dpmap.insert(summary, trace);
                }
            } else {
                dpmap.insert(summary, trace);
            }
        }

        match self {
            NiceBag::Leaf {vertices: _, map} => {
                let mut dpmap = FxHashMap::default();
                dpmap.insert(Summary::default(), Trace::default());
                map.insert(Evaluation::default(), dpmap);
            }
            NiceBag::Insert { vertices, child, x, map} => {
                child.compute_dpmap_by_group(env);
                let child_dpmap_by_group = child.get_dpmap_by_group();

                for child_dpmap in child_dpmap_by_group.values() {
                    for (summary, trace) in child_dpmap.iter() {
                        if x != &env.circuit.root_id {
                            let neighbor_eval0 = trace.neighbor_eval.insert_unchecked(*x, false);
                            if neighbor_eval0.is_valid_around(x, env) {
                                let bag_eval0 = summary.bag_eval.insert_unchecked(*x, false);
                                let is_known0 = summary.is_known.clone();
                                let tgraph0 = summary.transitive_graph.clone();
                                let summary0 = Summary::new(bag_eval0, is_known0, tgraph0);
                                insert_if_better(map, summary0, Trace::for_insert(trace.cost, neighbor_eval0, summary.clone()));
                            }
                        }
                        let neighbor_eval1 = trace.neighbor_eval.insert_unchecked(*x, true);
                        let new_cost = trace.cost + env.cost.get(x).unwrap_or(&Cost::from(0));
                        if new_cost <= env.max_cost && neighbor_eval1.is_valid_around(x, env) {
                            let tgraph1 = summary.transitive_graph.insert_vertex(x, env);
                            if !tgraph1.has_cycle() {
                                let bag_eval1 = summary.bag_eval.insert_unchecked(*x, true);
                                let is_known1 = summary.is_known.fix_tags_around(
                                    x,
                                    &vertices,
                                    &neighbor_eval1,
                                    env,
                                );
                                let summary1 = Summary::new(bag_eval1, is_known1, tgraph1);
                                insert_if_better(map, summary1, Trace::for_insert(new_cost, neighbor_eval1, summary.clone()));
                            }
                        }
                    }
                }
            }
            NiceBag::Forget {
                vertices,
                child,
                x,
                map,
            } => {
                child.compute_dpmap_by_group(env);
                let child_dpmap_by_group = child.get_dpmap_by_group();
                for child_dpmap in child_dpmap_by_group.values() {
                    for (summary, trace) in child_dpmap.iter() {
                        let new_bag_eval = summary.bag_eval.remove(x);
                        let new_is_known = summary.is_known.remove(x);
                        let new_tgraph = summary.transitive_graph.remove(x);
                        let new_summary = Summary::new(new_bag_eval, new_is_known, new_tgraph);
                        let mut new_neighbor_eval = Evaluation::default();

                        // Rebuild neighbors from current set of vertices.
                        // This can be optimized by using iterative updates and keeping
                        // track of how many neighbors each vertex has in the current set,
                        // but it's a significant bit of work.

                        let mut neighborhood: FxHashSet<usize> = FxHashSet::default();
                        for u in vertices.iter() {
                            neighborhood.extend(env.circuit.inputs_or_unreachable(u));
                            neighborhood.extend(env.circuit.outputs_or_unreachable(u));
                            neighborhood.insert(*u);
                        }
                        for (u, b) in trace.neighbor_eval.map.iter() {
                            if neighborhood.contains(u) {
                                new_neighbor_eval.map.insert(*u, *b);
                            }
                        }
                        insert_if_better(map, new_summary, Trace::for_forget(trace.cost, new_neighbor_eval, summary.clone()));
                    }
                }
            }
            NiceBag::Join {
                vertices,
                child1,
                child2,
                map,
            } => {
                child1.compute_dpmap_by_group(env);
                child2.compute_dpmap_by_group(env);
                let child1_dpmap_by_group = child1.get_dpmap_by_group();
                let child2_dpmap_by_group = child2.get_dpmap_by_group();
                for (bag_eval, child1_dpmap) in child1_dpmap_by_group.iter() {
                    if let Some(child2_dpmap) = child2_dpmap_by_group.get(bag_eval) {
                        for (summary1, trace1) in child1_dpmap.iter() {
                            for (summary2, trace2) in child2_dpmap.iter() {
                                let new_neighbor_eval = trace1.neighbor_eval.merge(&trace2.neighbor_eval);
                                let new_cost = trace1.cost + trace2.cost - vertices.iter().map(|u| env.cost.get(u).unwrap_or(&Cost::from(0)).clone()).sum::<Cost>();
                                if vertices.iter()
                                    .filter(|u| summary1.is_known.map.get(u) == Some(&false)
                                        && summary2.is_known.map.get(u) == Some(&false))
                                    .all(|u| new_neighbor_eval.is_valid_at(u, env))
                                    && new_cost <= env.max_cost
                                {
                                    let new_tgraph =
                                        summary1.transitive_graph.merge(&summary2.transitive_graph);
                                    if !new_tgraph.has_cycle() {
                                        let new_is_known = KnownTag::compute_all_tags(
                                            vertices,
                                            &new_neighbor_eval,
                                            env,
                                        );
                                        let new_summary = Summary::new(
                                            bag_eval.clone(),
                                            new_is_known,
                                            new_tgraph,
                                        );

                                        insert_if_better(map, new_summary, Trace::for_join(new_cost, new_neighbor_eval, summary1.clone(), summary2.clone()));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn vertices(&self) -> &IntSet<usize> {
        match self {
            NiceBag::Leaf { vertices, map: _} => vertices,
            NiceBag::Insert {
                vertices,
                child: _,
                x: _,
                map: _,
            } => vertices,
            NiceBag::Forget {
                vertices,
                child: _,
                x: _,
                map: _,
            } => vertices,
            NiceBag::Join {
                vertices,
                child1: _,
                child2: _,
                map: _,
            } => vertices,
        }
    }

    fn new_leaf() -> Self {
        NiceBag::Leaf {
            vertices: IntSet::default(),
            map: FxHashMap::default(),
        }
    }

    fn new_insert(vertices: IntSet<usize>, child: Box<NiceBag>, x: usize) -> NiceBag {
        NiceBag::Insert { vertices, child, x, map: FxHashMap::default() }
    }

    fn new_forget(vertices: IntSet<usize>, child: Box<NiceBag>, x: usize) -> NiceBag {
        NiceBag::Forget { vertices, child, x, map: FxHashMap::default() }
    }

    fn new_join(vertices: IntSet<usize>, child1: Box<NiceBag>, child2: Box<NiceBag>) -> NiceBag {
        NiceBag::Join {
            vertices,
            child1,
            child2,
            map: FxHashMap::default(),
        }
    }
}


/// For debug-use only. Given a set of true variables (not vertices),
/// evaluate the circuit and compute as many gates as possible.
fn verify(true_variables: &IntSet<usize>, env: &Env) -> Evaluation {
    let mut eval = Evaluation::default();

    let variables = env
        .circuit
        .get_vertices()
        .iter()
        .filter(|u| env.circuit.gate_type_or_unreachable(u) == &Gate::Variable);
    for u in variables {
        if true_variables.contains(u) {
            eval.map.insert(*u, true);
        } else {
            eval.map.insert(*u, false);
        }
    }

    let mut queue: VecDeque<usize> = VecDeque::default();
    for u in env.circuit.get_vertices() {
        queue.push_back(*u);
    }

    while let Some(u) = queue.pop_front() {
        match env.circuit.gate_type_or_unreachable(&u) {
            Gate::And => {
                let inputs = env.circuit.inputs_or_unreachable(&u);
                let mut all_true = true;
                let mut updated = false;
                for v in inputs {
                    match eval.map.get(&v) {
                        Some(true) => {}
                        Some(false) => {
                            eval.map.insert(u, false);
                            for w in env.circuit.outputs_or_unreachable(&u) {
                                queue.push_back(*w);
                            }
                            updated = true;
                        }
                        None => all_true = false,
                    }
                }
                if all_true && !updated {
                    eval.map.insert(u, true);
                    for w in env.circuit.outputs_or_unreachable(&u) {
                        queue.push_back(*w);
                    }
                }
            }
            Gate::Or => {
                let inputs = env.circuit.inputs_or_unreachable(&u);
                let mut all_false = true;
                let mut updated = false;
                for v in inputs {
                    match eval.map.get(&v) {
                        Some(false) => {}
                        Some(true) => {
                            eval.map.insert(u, true);
                            for w in env.circuit.outputs_or_unreachable(&u) {
                                queue.push_back(*w);
                            }
                            updated = true;
                        }
                        None => all_false = false,
                    }
                }
                if all_false && !updated {
                    eval.map.insert(u, false);
                    for w in env.circuit.outputs_or_unreachable(&u) {
                        queue.push_back(*w);
                    }
                }
            }
            Gate::Variable => {}
        }
    }

    eval
}

impl Extractor for TreewidthExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        let start_time = std::time::Instant::now();

        let baseline_extract = extract::faster_greedy_dag::FasterGreedyDagExtractor
            .boxed()
            .extract(egraph, roots);
        let max_cost = baseline_extract.dag_cost(egraph, roots) * (1.0 + 1e-5);

        // Make ClassIds and NodeIds compatible with arboretum_td
        let mut id_converter = IdConverter::default();
        for (cid, class) in egraph.classes() {
            id_converter.get_oid_or_add_class(cid);
            for nid in class.nodes.iter() {
                id_converter.get_aid_or_add_node(nid);
                id_converter.get_vid_or_add_node(nid);
            }
        }

        if VERBOSE {
            println!("# classes: {}", egraph.classes().len());
            println!("# nodes: {}", egraph.nodes.len());
        }

        // Create circuit by replacing e-classes with Or gates and e-nodes
        // with both an And gate (to its input classes) and a Variable gate
        // under the And gate (selecting the e-node for extraction)
        let root_id = id_converter.reserve_id();

        if VERBOSE {
            println!("root id: {}", root_id);
        }

        let mut circuit = Circuit::with_root(root_id);
        for (cid, class) in egraph.classes() {
            let u = id_converter.get_oid_or_add_class(cid);
            circuit.add_vertex(u, Gate::Or);

            for nid in class.nodes.iter() {
                let v = id_converter.get_aid_or_add_node(nid);
                circuit.add_vertex(v, Gate::And);
                circuit.add_edge(v, u);

                if let Some(node) = egraph.nodes.get(nid) {
                    for child_nid in node.children.iter() {
                        let w = id_converter.get_oid_or_add_class(egraph.nid_to_cid(child_nid));
                        circuit.add_vertex(w, Gate::Or);
                        circuit.add_edge(w, v);
                    }
                }

                let w = id_converter.get_vid_or_add_node(nid);
                circuit.add_vertex(w, Gate::Variable);
                circuit.add_edge(w, v);
            }
        }

        // Require extraction of all roots by joining all the root e-classes
        // to a new And gate at the top.
        circuit.add_vertex(root_id, Gate::And);
        for root in roots {
            let v = id_converter.get_oid_or_add_class(root);
            circuit.add_edge(v, root_id);
        }

        let options = SimplifyOptions {
            remove_unreachable: true,
            contract_indegree_one: true,
            contract_same_gate: true,
            remove_lone_or_loops: true,
            same_gate_is_tree: true,
            distributivity: true,
            collect_variables: true,
            verbose: false,
        };
        let mut env = Env::new(id_converter, circuit, egraph, max_cost);

        env.simplify(options);
        env.compute_costs();

        if SAVE_COSMOGRAPH {
            env.save_cosmograph("cosmo".to_string())
                .expect("cosmo failed");
        }

        let graph = env.circuit.to_graph();

        if VERBOSE {
            println!("preprocessing: {} us", start_time.elapsed().as_micros());
        }

        let start_time = std::time::Instant::now();

        // Run tree decomposition and identify a bag with the root
        // Solver is from arboretum_td and implements the winning algorithm from a 2017 competition:
        // https://pacechallenge.org/2017/treewidth/
        //
        // Solver::default_heuristic() --- Heuristic solver that may not return optimal treewidth
        // Solver::default_exact() ------- Exact solver that is slower
        //
        // Note: the treewidth does not affect the accuracy of the main algorithm.
        // A tree decomposition with larger treewidth will just have larger bags and
        // take longer to compute the optimal extraction.
        let td = Solver::default_heuristic().solve(&graph);

        if VERBOSE {
            println!("td: {} us", start_time.elapsed().as_micros());
        }

        let start_time = std::time::Instant::now();

        let mut root_bag_id: usize = 0;
        for bag in td.bags() {
            if bag.vertex_set.contains(&root_id) {
                root_bag_id = bag.id;
                break;
            }
        }
        let mut nice_td = forget_until_root(to_nice_decomp(&td, &root_bag_id, None), &root_id);

        if VERBOSE {
            println!("treewidth: {}", td.max_bag_size - 1);
            println!("nice td: {} us", start_time.elapsed().as_micros());
        }

        let start_time = std::time::Instant::now();

        // Find the best satisfying assignment
        nice_td.compute_dpmap_by_group(&env);
        
        if VERBOSE {
            println!("extract: {} us", start_time.elapsed().as_micros());
        }
        let start_time = std::time::Instant::now();
        
        // Follow trace
        let dpmap_by_group = nice_td.get_dpmap_by_group();
        let mut root_true_eval = Evaluation::default();
        root_true_eval.map.insert(root_id, true);
        let mut result = ExtractionResult::default();
        if let Some(dpmap) = dpmap_by_group.get(&root_true_eval) {
            if let Some((summary, _)) = dpmap.iter().next() {
                let mut best_eval = Evaluation::default();
                let mut stack = Vec::default();
                stack.push((&nice_td, summary));
                while let Some((bag, summary)) = stack.pop() {
                    match **bag {
                        NiceBag::Leaf { vertices: _, map: _ } => {},
                        NiceBag::Insert { vertices: _, ref child, x, map: _ } => {
                            let dpmap_by_group = bag.get_dpmap_by_group();
                            let dpmap = dpmap_by_group.get(&summary.bag_eval).unwrap();
                            let new_summary = dpmap.get(summary).as_ref().unwrap().summary.as_ref().unwrap();
                            best_eval.map.insert(x, *summary.bag_eval.get_or_unreachable(&x));
                            stack.push((&child, &new_summary));
                        }
                        NiceBag::Forget { vertices: _, ref child, x: _, map: _ } => {
                            let dpmap_by_group = bag.get_dpmap_by_group();
                            let dpmap = dpmap_by_group.get(&summary.bag_eval).unwrap();
                            let new_summary = dpmap.get(summary).as_ref().unwrap().summary.as_ref().unwrap();
                            stack.push((&child, &new_summary));
                        }
                        NiceBag::Join { vertices: _, ref child1, ref child2, map: _ } => {
                            let dpmap_by_group = bag.get_dpmap_by_group();
                            let dpmap = dpmap_by_group.get(&summary.bag_eval).unwrap();
                            let new_summary1 = dpmap.get(summary).as_ref().unwrap().summary.as_ref().unwrap();
                            let new_summary2 = dpmap.get(summary).as_ref().unwrap().summary2.as_ref().unwrap();
                            stack.push((&child1, &new_summary1));
                            stack.push((&child2, &new_summary2));
                        }
                        
                    }
                }
                for (u, value) in best_eval.map.iter() {
                    if let Some(nid_set) = env.id_converter.vid_to_nid_set(u) {
                        for nid in nid_set {
                            if *value {
                                result.choose(egraph.nid_to_cid(nid).clone(), nid.clone());
                            }
                        }
                    }
                }
            }
        }

        if VERBOSE {
            println!("trace: {} us", start_time.elapsed().as_micros());
            println!("baseline (+ epsilon): {}", env.max_cost);
        }

        result
    }
}
