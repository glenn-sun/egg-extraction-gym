use csv::Writer;
use std::collections::{BTreeSet, VecDeque};
use std::error::Error;
use std::hash::{BuildHasherDefault, Hash, Hasher};

use super::*;
use arboretum_td::graph::{HashMapGraph, MutableGraph};
use arboretum_td::solver::Solver;
use arboretum_td::tree_decomposition::TreeDecomposition;

use nohash_hasher::{IntMap, IntSet, NoHashHasher};

pub struct TreewidthExtractor;

const VERBOSE: bool = true;
const SAVE_COSMOGRAPH: bool = true;

/// Converts between usize IDs and ClassIds/NodeIds.
/// IDs for use with arboretum_td are required to be usize.
/// For convenience, this maintains maps in both directions
/// and segregates IDs for Variables, And gates, and Or gates.
/// One variable may correspond to a set of NodeIds to be
/// selected simultaneously.
#[derive(Debug, Default)]
struct IdConverter {
    vid_to_nid_set: IntMap<usize, FxHashSet<NodeId>>,
    cid_to_oid: FxHashMap<ClassId, usize>,
    nid_to_aid: FxHashMap<NodeId, usize>,
    nid_to_vid: FxHashMap<NodeId, usize>,
    counter: usize,
}

impl IdConverter {
    fn get_oid_or_add_class(&mut self, cid: &ClassId) -> usize {
        if let Some(oid) = self.cid_to_oid.get(cid) {
            *oid
        } else {
            self.cid_to_oid.insert(cid.clone(), self.counter);
            self.counter += 1;
            self.counter - 1
        }
    }

    fn get_aid_or_add_node(&mut self, nid: &NodeId) -> usize {
        if let Some(aid) = self.nid_to_aid.get(&nid) {
            *aid
        } else {
            self.nid_to_aid.insert(nid.clone(), self.counter);
            self.counter += 1;
            self.counter - 1
        }
    }

    fn get_vid_or_add_node(&mut self, nid: &NodeId) -> usize {
        if let Some(vid) = self.nid_to_vid.get(&nid) {
            *vid
        } else {
            self.vid_to_nid_set
                .entry(self.counter)
                .or_default()
                .insert(nid.clone());
            self.nid_to_vid.insert(nid.clone(), self.counter);
            self.counter += 1;
            self.counter - 1
        }
    }

    fn merge_vid_keep1(&mut self, vid1: usize, vid2: usize) {
        let nid_set1 = self.vid_to_nid_set.get(&vid1).cloned().unwrap_or_default();
        let nid_set2 = self.vid_to_nid_set.get(&vid2).cloned().unwrap_or_default();
        for nid in &nid_set2 {
            self.nid_to_vid.insert(nid.clone(), vid1);
        }
        let union: FxHashSet<NodeId> = nid_set1.union(&nid_set2).cloned().collect();
        self.vid_to_nid_set.insert(vid1, union);
        self.vid_to_nid_set.remove(&vid2);
    }

    /// Reserve an ID that does not correspond to any e-class or e-node.
    fn reserve_id(&mut self) -> usize {
        self.counter += 1;
        self.counter - 1
    }

    fn vid_to_nid_set(&self, vid: &usize) -> Option<&FxHashSet<NodeId>> {
        self.vid_to_nid_set.get(vid)
    }
}

/// A basic implementation of a monotone circuit. The circuit
/// may have cycles and may not be connected. Used as
/// an intermediate representation between e-graphs and
/// tree decomposition.
#[derive(Debug)]
struct Circuit {
    vertices: IntSet<usize>,
    inputs: IntMap<usize, IntSet<usize>>,
    outputs: IntMap<usize, IntSet<usize>>,
    gate_type: IntMap<usize, Gate>,
    root_id: usize,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Gate {
    Variable,
    Or,
    And,
}

impl Circuit {
    fn with_root(root_id: usize) -> Self {
        Circuit {
            vertices: IntSet::default(),
            inputs: IntMap::default(),
            outputs: IntMap::default(),
            gate_type: IntMap::default(),
            root_id,
        }
    }

    fn add_vertex(&mut self, u: usize, gate_type: Gate) {
        self.vertices.insert(u);
        self.gate_type.insert(u, gate_type);
        self.inputs.entry(u).or_default();
        self.outputs.entry(u).or_default();
    }

    fn add_edge(&mut self, u: usize, v: usize) {
        self.inputs.entry(v).or_default().insert(u);
        self.outputs.entry(u).or_default().insert(v);
    }

    fn remove_vertex(&mut self, u: usize) {
        let inputs = self.inputs(&u).cloned().unwrap_or_default();
        for v in inputs {
            self.remove_edge(v, u);
        }
        let outputs = self.outputs(&u).cloned().unwrap_or_default();
        for v in outputs {
            self.remove_edge(u, v);
        }
        self.vertices.remove(&u);
    }

    fn remove_edge(&mut self, u: usize, v: usize) {
        self.inputs.entry(v).or_default().remove(&u);
        self.outputs.entry(u).or_default().remove(&v);
    }

    fn contract_edge_remove_out(&mut self, u: usize, v: usize) {
        self.remove_edge(u, v);
        let inputs = self.inputs(&v).cloned().unwrap_or_default();
        let outputs = self.outputs(&v).cloned().unwrap_or_default();
        for w in outputs {
            self.remove_edge(v, w);
            if u != w {
                self.add_edge(u, w);
            }
            self.add_edge(u, w);
        }
        for w in inputs {
            self.remove_edge(w, v);
            self.add_edge(w, u);
        }
        self.vertices.remove(&v);
    }

    fn get_vertices(&self) -> &IntSet<usize> {
        &self.vertices
    }

    fn inputs(&self, u: &usize) -> Option<&IntSet<usize>> {
        self.inputs.get(u)
    }
    fn inputs_or_unreachable(&self, u: &usize) -> &IntSet<usize> {
        self.inputs.get(u).unwrap_or_else(|| unreachable!())
    }

    fn outputs(&self, u: &usize) -> Option<&IntSet<usize>> {
        self.outputs.get(u)
    }
    fn outputs_or_unreachable(&self, u: &usize) -> &IntSet<usize> {
        self.outputs.get(u).unwrap_or_else(|| unreachable!())
    }

    fn gate_type(&self, u: &usize) -> Option<&Gate> {
        self.gate_type.get(u)
    }
    fn gate_type_or_unreachable(&self, u: &usize) -> &Gate {
        self.gate_type.get(u).unwrap_or_else(|| unreachable!())
    }

    /// Convert to an undirected graph compatiable with arboretum_td
    fn to_graph(&self) -> HashMapGraph {
        let mut graph = HashMapGraph::new();
        for u in self.get_vertices() {
            for v in self.inputs_or_unreachable(u) {
                graph.add_edge(*u, *v);
            }
        }
        graph
    }
}

/// A "summary" of an evaluation. It contains all the information
/// needed to determine whether or not we need to keep a record
/// of the evaluation moving forward.
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
struct Summary {
    bag_eval: Evaluation,
    is_known: KnownTag,
    transitive_graph: TransitiveGraph,
}

impl Summary {
    fn new(bag_eval: Evaluation, is_known: KnownTag, transitive_graph: TransitiveGraph) -> Self {
        Self {
            bag_eval,
            is_known,
            transitive_graph,
        }
    }
}

/// A map from vertices to T/F. May only map the vertices in a bag of
/// the tree decomposition, or all of the vertices below a bag.
/// Inputs of a vertex in the domain may or may not be in the domain.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
struct Evaluation {
    map: IntMap<usize, bool>,
}

impl Hash for Evaluation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut sorted: Vec<(&usize, &bool)> = self.map.iter().collect();
        sorted.sort_by_key(|(k, _)| *k);
        for elt in sorted {
            elt.hash(state);
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum DFSStatus {
    Unvisited,
    CurrentPath,
    Visited,
}

impl Evaluation {
    fn get_or_unreachable(&self, u: &usize) -> &bool {
        self.map.get(u).unwrap_or_else(|| unreachable!())
    }

    /// Check if there exists a way to pick the inputs to u
    /// outside the evaluation's domain such that together,
    /// u meets the requirements of its gate type.
    ///  
    /// Panics if u is not in the domain.
    fn is_valid_at(&self, u: &usize, env: &Env) -> bool {
        match env.circuit.gate_type_or_unreachable(u) {
            Gate::And => {
                let inputs = env.circuit.inputs_or_unreachable(u);
                if self.get_or_unreachable(u) == &true {
                    for v in inputs {
                        if self.map.get(v) == Some(&false) {
                            return false;
                        }
                    }
                    return true;
                } else {
                    for v in inputs {
                        let b = self.map.get(v);
                        if b == None || b == Some(&false) {
                            return true;
                        }
                    }
                    return false;
                }
            }
            Gate::Or => {
                let inputs = env.circuit.inputs_or_unreachable(u);
                if self.get_or_unreachable(u) == &false {
                    for v in inputs {
                        if self.map.get(v) == Some(&true) {
                            return false;
                        }
                    }
                    return true;
                } else {
                    for v in inputs {
                        let b = self.map.get(v);
                        if b == None || b == Some(&true) {
                            return true;
                        }
                    }
                    return false;
                }
            }
            Gate::Variable => true,
        }
    }

    /// Checks validity for u and the outputs of u
    fn is_valid_around(&self, u: &usize, env: &Env) -> bool {
        self.is_valid_at(u, env)
            && env
                .circuit
                .outputs_or_unreachable(u)
                .iter()
                .all(|v| !self.map.contains_key(v) || self.is_valid_at(v, env))
    }

    fn insert_unchecked(&self, u: usize, b: bool) -> Evaluation {
        let mut new_eval = self.clone();
        new_eval.map.insert(u, b);
        new_eval
    }

    fn remove(&self, u: &usize) -> Evaluation {
        let mut new_eval = self.clone();
        new_eval.map.remove(u);
        new_eval
    }

    fn merge(&self, other: &Evaluation) -> Evaluation {
        let mut new_eval = self.clone();
        new_eval.map.reserve(other.map.len());
        new_eval.map.extend(other.map.clone());
        new_eval
    }

    fn cost(&self, env: &Env) -> Cost {
        self.map
            .iter()
            .filter(|(u, b)| {
                b == &&true && env.circuit.gate_type_or_unreachable(*u) == &Gate::Variable
            })
            .map(|(u, _)| env.cost.get(u).unwrap_or_else(|| unreachable!()))
            .sum()
    }

    /// For debug-use only. Cycle checking is done more
    /// quickly using TransitiveGraph.
    fn has_cycle(&self, env: &Env) -> bool {
        let mut answer = false;
        let mut status: HashMap<usize, DFSStatus, BuildHasherDefault<NoHashHasher<usize>>> =
            HashMap::with_capacity_and_hasher(self.map.len(), BuildHasherDefault::default());
        let mut call_stack: Vec<usize> = Vec::with_capacity(self.map.len() * 4);

        for (u, _) in self.map.iter().filter(|(_, b)| **b) {
            status.insert(*u, DFSStatus::Unvisited);
            call_stack.push(*u);
            call_stack.push(*u);
        }

        while let Some(u) = call_stack.pop() {
            match status.get(&u) {
                Some(&DFSStatus::Visited) => {}
                Some(&DFSStatus::CurrentPath) => {
                    status.insert(u, DFSStatus::Visited);
                }
                Some(&DFSStatus::Unvisited) => {
                    status.insert(u, DFSStatus::CurrentPath);
                    for v in env.circuit.outputs_or_unreachable(&u) {
                        match status.get(v) {
                            Some(&DFSStatus::CurrentPath) => {
                                answer = true;
                                println!("{:#?}", status);
                                break;
                            }

                            Some(&DFSStatus::Unvisited) => {
                                call_stack.push(*v);
                                call_stack.push(*v);
                            }

                            _ => {}
                        }
                    }
                }
                None => unreachable!(),
            }
        }
        answer
    }

    /// For debug-use only. It is more efficient to update
    /// TransitiveGraph incrementally than to compute from an evaluation.
    fn to_tgraph(&self, env: &Env) -> TransitiveGraph {
        let mut tgraph = TransitiveGraph::default();
        for (u, _) in self.map.iter().filter(|(_, b)| **b) {
            tgraph = tgraph.insert_vertex(u, env);
        }
        tgraph
    }
}

/// Tags every true vertex in an evaluation with whether or not
/// for all choices of unassigned inputs, the vertex should evaluate
/// to true.
///
/// For example, an true OR gate with all inputs not yet assigned would
/// be tagged false (because we may later set all inputs to false,
/// invalidating it), whereas a true OR gate with a true input would
/// be tagged true.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
struct KnownTag {
    map: IntMap<usize, bool>,
}

impl Hash for KnownTag {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut sorted: Vec<(&usize, &bool)> = self.map.iter().collect();
        sorted.sort_by_key(|(k, _)| *k);
        for elt in sorted {
            elt.hash(state);
        }
    }
}

impl KnownTag {
    /// Intended for internal use. (Re)compute the tag of a vertex.
    /// This should be called whenever an input changes.
    fn fix_tag_at(&mut self, u: &usize, full_eval: &Evaluation, env: &Env) {
        match env.circuit.gate_type_or_unreachable(u) {
            Gate::And => {
                let inputs = env.circuit.inputs_or_unreachable(u);
                for v in inputs {
                    if full_eval.map.get(v).is_none() {
                        self.map.insert(*u, false);
                        return;
                    }
                }
                self.map.insert(*u, true);
            }
            Gate::Or => {
                let inputs = env.circuit.inputs_or_unreachable(u);
                for v in inputs {
                    if full_eval.map.get(v) == Some(&true) {
                        self.map.insert(*u, true);
                        return;
                    }
                }
                self.map.insert(*u, false);
            }
            Gate::Variable => {
                self.map.insert(*u, true);
            }
        }
    }

    // (Re)compute the tags of a vertex and its outputs.
    fn fix_tags_around(
        &self,
        u: &usize,
        vertices: &IntSet<usize>,
        full_eval: &Evaluation,
        env: &Env,
    ) -> KnownTag {
        let mut new_kt = self.clone();
        new_kt.fix_tag_at(u, full_eval, env);
        for v in env.circuit.outputs_or_unreachable(u) {
            if vertices.contains(v) && full_eval.map.get(v) == Some(&true) {
                new_kt.fix_tag_at(v, full_eval, env);
            }
        }
        new_kt
    }

    // Compute all tags for a set of vertices.
    fn compute_all_tags(vertices: &IntSet<usize>, full_eval: &Evaluation, env: &Env) -> KnownTag {
        let mut new_kt = KnownTag::default();
        for u in vertices {
            if full_eval.map.get(u) == Some(&true) {
                new_kt.fix_tag_at(u, full_eval, env);
            }
        }
        new_kt
    }

    fn remove(&self, u: &usize) -> KnownTag {
        let mut new_kt = self.clone();
        new_kt.map.remove(u);
        new_kt
    }
}

/// A transitive graph is a graph where a path u -> v implies an
/// edge u -> v.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
struct TransitiveGraph {
    inputs: IntMap<usize, BTreeSet<usize>>,
    outputs: IntMap<usize, BTreeSet<usize>>,
    vertices: IntSet<usize>,
}

impl Hash for TransitiveGraph {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut sorted: Vec<(&usize, &BTreeSet<usize>)> = self.inputs.iter().collect();
        sorted.sort_by_key(|(k, _)| *k);
        for elt in sorted {
            elt.hash(state);
        }
    }
}

impl TransitiveGraph {
    fn has_cycle(&self) -> bool {
        for (u, inputs) in &self.inputs {
            if inputs.contains(u) {
                return true;
            }
        }
        false
    }

    /// Insert a vertex into the TransitiveGraph and propagate all
    /// new paths into new edges.
    ///
    /// Warning: does not insert a self-loop at u, even if the
    /// transitive graph should have it. This is an optimization,
    /// since there will always be a self-loop elsewhere, too.
    fn insert_vertex(&self, u: &usize, env: &Env) -> TransitiveGraph {
        let mut new_tg = self.clone();

        let l1_inputs = env
            .circuit
            .inputs_or_unreachable(u)
            .intersection(&self.vertices);
        let mut l2_inputs = IntSet::default();
        for v in l1_inputs {
            l2_inputs.extend(self.inputs.get(v).unwrap());
            l2_inputs.insert(*v);
        }

        let l1_outputs = env
            .circuit
            .outputs_or_unreachable(u)
            .intersection(&self.vertices);
        let mut l2_outputs = IntSet::default();
        for v in l1_outputs {
            l2_outputs.extend(self.outputs.get(v).unwrap());
            l2_outputs.insert(*v);
        }

        new_tg.vertices.insert(*u);
        new_tg.inputs.entry(*u).or_default().extend(&l2_inputs);
        new_tg.outputs.entry(*u).or_default().extend(&l2_outputs);

        for v in &l2_inputs {
            new_tg.outputs.get_mut(v).unwrap().extend(&l2_outputs);
            new_tg.outputs.get_mut(v).unwrap().insert(*u);
        }
        for v in &l2_outputs {
            new_tg.inputs.get_mut(v).unwrap().extend(&l2_inputs);
            new_tg.inputs.get_mut(v).unwrap().insert(*u);
        }

        new_tg
    }

    /// Intended for internal use. Insert an edge into the TransitiveGraph
    /// and propagate all new paths into new edges.
    fn insert_edge(&mut self, u: &usize, v: &usize) {
        let mut inputs: IntSet<usize> = self.inputs.get(u).unwrap().iter().cloned().collect();
        let mut outputs: IntSet<usize> = self.outputs.get(v).unwrap().iter().cloned().collect();
        inputs.insert(*u);
        outputs.insert(*v);
        for w in &inputs {
            self.outputs.get_mut(w).unwrap().extend(&outputs);
        }
        for w in &outputs {
            self.inputs.get_mut(w).unwrap().extend(&inputs);
        }
    }

    fn remove(&self, u: &usize) -> TransitiveGraph {
        let mut new_tg = self.clone();
        new_tg.vertices.remove(u);
        new_tg.inputs.remove(u);
        for v in &new_tg.vertices {
            new_tg.inputs.get_mut(v).unwrap().remove(u);
        }
        new_tg.outputs.remove(u);
        for v in &new_tg.vertices {
            new_tg.outputs.get_mut(v).unwrap().remove(u);
        }
        new_tg
    }

    /// Union the edge sets of two TransitiveGraphs on the same vertex set
    /// and propagate all new paths into new edges.
    /// Panics if not the same vertex set.
    fn merge(&self, other: &TransitiveGraph) -> TransitiveGraph {
        let mut new_tg = self.clone();
        for (v, inputs) in &other.inputs {
            for u in inputs {
                if !new_tg.inputs.get(v).unwrap().contains(u) {
                    new_tg.insert_edge(u, v)
                }
            }
        }
        new_tg
    }

    /// For debug-use only. Restrict a TransitiveGraph to a set of vertices.
    fn restrict(&self, vertices: &IntSet<usize>) -> TransitiveGraph {
        let mut new_tg = self.clone();
        for u in self.vertices.iter() {
            if !vertices.contains(u) {
                new_tg = new_tg.remove(u);
            }
        }
        new_tg
    }
}

/// The main data structure used to remember partial computations
/// throughout the dynamic programming. For every summary, map to the
/// cheapest evaluation on all vertices under the current bag that produces
/// the summary.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
struct DPMap {
    map: FxHashMap<Summary, Evaluation>,
}

impl DPMap {
    fn add_if_better(&mut self, summary: Summary, full_eval: Evaluation, env: &Env) {
        if let Some(self_full_assign) = self.map.get_mut(&summary) {
            let self_cost = self_full_assign.cost(env);
            if self_cost > full_eval.cost(env) {
                self.map.insert(summary, full_eval);
            }
        } else {
            self.map.insert(summary, full_eval);
        }
    }
}

#[derive(Default, Debug, Clone)]
struct DPMapByGroup {
    map: FxHashMap<Evaluation, DPMap>,
}

#[derive(Debug)]
enum NiceBag {
    Leaf {
        vertices: IntSet<usize>,
    },
    Insert {
        vertices: IntSet<usize>,
        child: Box<NiceBag>,
        x: usize,
    },
    Forget {
        vertices: IntSet<usize>,
        child: Box<NiceBag>,
        x: usize,
    },
    Join {
        vertices: IntSet<usize>,
        child1: Box<NiceBag>,
        child2: Box<NiceBag>,
    },
}

impl NiceBag {
    /// See write-up for algorithmic details and proof.
    fn get_dpmap_by_group(&self, env: &Env) -> DPMapByGroup {
        let mut dpmap_by_group = DPMapByGroup::default();
        match self {
            NiceBag::Leaf { vertices: _ } => {
                let mut dpmap = DPMap::default();
                dpmap.add_if_better(Summary::default(), Evaluation::default(), env);
                dpmap_by_group.map.insert(Evaluation::default(), dpmap);
            }
            NiceBag::Insert { vertices, child, x } => {
                let child_dpmap_by_group = child.get_dpmap_by_group(env);
                for child_dpmap in child_dpmap_by_group.map.values() {
                    for (summary, full_eval) in child_dpmap.map.iter() {
                        if x != &env.circuit.root_id {
                            let full_eval0 = full_eval.insert_unchecked(*x, false);
                            if full_eval0.is_valid_around(x, env) {
                                let bag_eval0 = summary.bag_eval.insert_unchecked(*x, false);
                                let is_known0 = summary.is_known.clone();
                                let tgraph0 = summary.transitive_graph.clone();
                                let summary0 = Summary::new(bag_eval0, is_known0, tgraph0);
                                dpmap_by_group
                                    .map
                                    .entry(summary0.bag_eval.clone())
                                    .or_default()
                                    .add_if_better(summary0, full_eval0, env);
                            }
                        }
                        let full_eval1 = full_eval.insert_unchecked(*x, true);
                        if full_eval1.is_valid_around(x, env)
                            && full_eval1.cost(env) <= env.max_cost
                        {
                            let tgraph1 = summary.transitive_graph.insert_vertex(x, env);
                            if !tgraph1.has_cycle() {
                                let bag_eval1 = summary.bag_eval.insert_unchecked(*x, true);
                                let is_known1 = summary.is_known.fix_tags_around(
                                    x,
                                    &vertices,
                                    &full_eval1,
                                    env,
                                );
                                let summary1 = Summary::new(bag_eval1, is_known1, tgraph1);
                                dpmap_by_group
                                    .map
                                    .entry(summary1.bag_eval.clone())
                                    .or_default()
                                    .add_if_better(summary1, full_eval1, env)
                            }
                        }
                    }
                }
            }
            NiceBag::Forget {
                vertices: _,
                child,
                x,
            } => {
                let child_dpmap_by_group = child.get_dpmap_by_group(env);
                for child_dpmap in child_dpmap_by_group.map.values() {
                    for (summary, full_eval) in child_dpmap.map.iter() {
                        let new_bag_eval = summary.bag_eval.remove(x);
                        let new_is_known = summary.is_known.remove(x);
                        let new_tgraph = summary.transitive_graph.remove(x);
                        let new_summary = Summary::new(new_bag_eval, new_is_known, new_tgraph);
                        let new_full_eval = full_eval.clone();
                        dpmap_by_group
                            .map
                            .entry(new_summary.bag_eval.clone())
                            .or_default()
                            .add_if_better(new_summary, new_full_eval, env);
                    }
                }
            }
            NiceBag::Join {
                vertices,
                child1,
                child2,
            } => {
                let child1_dpmap_by_group = child1.get_dpmap_by_group(env);
                let child2_dpmap_by_group = child2.get_dpmap_by_group(env);
                for (bag_eval, child1_dpmap) in child1_dpmap_by_group.map.iter() {
                    if let Some(child2_dpmap) = child2_dpmap_by_group.map.get(bag_eval) {
                        for (summary1, full_eval1) in child1_dpmap.map.iter() {
                            for (summary2, full_eval2) in child2_dpmap.map.iter() {
                                let new_full_eval = full_eval1.merge(full_eval2);
                                if vertices.iter()
                                    .filter(|u| summary1.is_known.map.get(u) == Some(&false)
                                        && summary2.is_known.map.get(u) == Some(&false))
                                    .all(|u| new_full_eval.is_valid_at(u, env))
                                    && new_full_eval.cost(env) <= env.max_cost
                                {
                                    let new_tgraph =
                                        summary1.transitive_graph.merge(&summary2.transitive_graph);
                                    if !new_tgraph.has_cycle() {
                                        let new_is_known = KnownTag::compute_all_tags(
                                            vertices,
                                            &new_full_eval,
                                            env,
                                        );
                                        let new_summary = Summary::new(
                                            bag_eval.clone(),
                                            new_is_known,
                                            new_tgraph,
                                        );
                                        dpmap_by_group
                                            .map
                                            .entry(bag_eval.clone())
                                            .or_default()
                                            .add_if_better(new_summary, new_full_eval, env);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        dpmap_by_group
    }

    fn vertices(&self) -> &IntSet<usize> {
        match self {
            NiceBag::Leaf { vertices } => vertices,
            NiceBag::Insert {
                vertices,
                child: _,
                x: _,
            } => vertices,
            NiceBag::Forget {
                vertices,
                child: _,
                x: _,
            } => vertices,
            NiceBag::Join {
                vertices,
                child1: _,
                child2: _,
            } => vertices,
        }
    }

    fn new_leaf() -> Self {
        NiceBag::Leaf {
            vertices: IntSet::default(),
        }
    }

    fn new_insert(vertices: IntSet<usize>, child: Box<NiceBag>, x: usize) -> NiceBag {
        NiceBag::Insert { vertices, child, x }
    }

    fn new_forget(vertices: IntSet<usize>, child: Box<NiceBag>, x: usize) -> NiceBag {
        NiceBag::Forget { vertices, child, x }
    }

    fn new_join(vertices: IntSet<usize>, child1: Box<NiceBag>, child2: Box<NiceBag>) -> NiceBag {
        NiceBag::Join {
            vertices,
            child1,
            child2,
        }
    }
}

/// Convert a tree decomposition to a nice tree decomposition.
fn to_nice_decomp(
    td: &TreeDecomposition,
    root_bag_id: &usize,
    parent_id: Option<&usize>,
) -> Box<NiceBag> {
    let root_bag = &td.bags[*root_bag_id];
    let mut child_ids = root_bag.neighbors.clone();
    parent_id.map(|u| child_ids.remove(u));
    if child_ids.is_empty() {
        let mut prev: Box<NiceBag> = Box::new(NiceBag::new_leaf());
        let mut vertices: IntSet<usize> = IntSet::default();
        for u in root_bag.vertex_set.iter() {
            vertices.insert(*u);
            let next = NiceBag::new_insert(vertices.clone(), prev, *u);
            prev = Box::new(next);
        }
        return prev;
    }

    let mut same_as_root_bag: Vec<Box<NiceBag>> = Vec::default();

    let child_nice_bags = child_ids
        .iter()
        .map(|u| to_nice_decomp(td, u, Some(root_bag_id)));
    for child_nice_bag in child_nice_bags {
        let mut prev = child_nice_bag;
        let mut vertices = prev.vertices().clone();
        let vertices_clone = vertices.clone();

        let root_vertices = IntSet::from_iter(root_bag.vertex_set.clone().into_iter());
        let root_not_child = root_vertices.difference(&vertices_clone);
        let child_not_root = vertices_clone.difference(&root_vertices);

        for u in child_not_root {
            vertices.remove(u);
            let next = NiceBag::new_forget(vertices.clone(), prev, *u);
            prev = Box::new(next);
        }
        for u in root_not_child {
            vertices.insert(*u);
            let next = NiceBag::new_insert(vertices.clone(), prev, *u);
            prev = Box::new(next);
        }
        same_as_root_bag.push(prev);
    }

    if let Some(mut prev) = same_as_root_bag.pop() {
        let vertices = prev.vertices().clone();
        for bag in same_as_root_bag {
            let join = NiceBag::new_join(vertices.clone(), prev, bag);
            prev = Box::new(join);
        }
        return prev;
    }

    Box::new(NiceBag::new_leaf())
}

/// Process a nice tree decomposition so that the root of the tree
/// is a bag containing a single vertex, namely the root of circuit.
fn forget_until_root(nice_td: Box<NiceBag>, root_id: &usize) -> Box<NiceBag> {
    let mut prev = nice_td;
    let mut vertices = prev.vertices().clone();
    let vertices_clone = vertices.clone();
    for u in vertices_clone.iter() {
        if u != root_id {
            vertices.remove(u);
            let next = NiceBag::new_forget(vertices.clone(), prev, *u);
            prev = Box::new(next);
        }
    }
    prev
}

fn print_stats(circuit: &Circuit) {
    let num_verts = circuit.get_vertices().len();
    let num_edges = circuit
        .get_vertices()
        .iter()
        .map(|u| circuit.inputs_or_unreachable(u).len() as f32)
        .sum::<f32>() as f32;
    println!("|V|, |E| = {}, {}", num_verts, num_edges);
}

/// A variety of circuit simplification rules. See write-up for details.
struct SimplifyOptions {
    pub remove_unreachable: bool,
    pub contract_indegree_one: bool,
    pub contract_same_gate: bool,
    pub remove_lone_or_loops: bool,
    pub same_gate_is_tree: bool,
    pub distributivity: bool,
    pub collect_variables: bool,
    pub verbose: bool,
}

impl Default for SimplifyOptions {
    fn default() -> Self {
        SimplifyOptions {
            remove_unreachable: true,
            contract_indegree_one: true,
            contract_same_gate: true,
            remove_lone_or_loops: true,
            same_gate_is_tree: true,
            distributivity: true,
            collect_variables: true,
            verbose: false,
        }
    }
}

/// An environment to be passed around various functions.
struct Env<'a> {
    id_converter: IdConverter,
    circuit: Circuit,
    egraph: &'a EGraph,
    max_cost: Cost,
    cost: IntMap<usize, Cost>,
}

impl<'a> Env<'a> {
    fn new(
        id_converter: IdConverter,
        circuit: Circuit,
        egraph: &'a EGraph,
        max_cost: Cost,
    ) -> Self {
        Env {
            id_converter,
            circuit,
            egraph,
            max_cost,
            cost: IntMap::default(),
        }
    }

    /// Graph visualization software at https://cosmograph.app.
    /// Saves csv files to disk that allow you to visualize the circuit.
    /// Upload filename.csv and filename_meta.csv to the web app, then
    /// under "Appearance", set node color by metadata color and link
    /// color by data color.
    ///
    /// Red means AND gate, green means OR gate, and blue means variable.
    /// Don't forget to turn on link arrows in the sidebar. Arrows
    /// point in the direction of data flow.
    fn save_cosmograph(&self, filename_noext: String) -> Result<(), Box<dyn Error>> {
        let mut data_writer = Writer::from_path(format!("{}.csv", filename_noext))?;
        #[derive(serde::Serialize)]
        struct CosmoData {
            input: usize,
            output: usize,
            color: String,
        }
        for u in self.circuit.get_vertices() {
            for v in self.circuit.outputs(u).unwrap_or(&IntSet::default()) {
                data_writer.serialize(CosmoData {
                    input: *u,
                    output: *v,
                    color: String::from("white"),
                })?;
            }
        }
        data_writer.flush()?;

        let mut meta_writer = Writer::from_path(format!("{}_meta.csv", filename_noext))?;
        #[derive(serde::Serialize)]
        struct CosmoMeta {
            id: usize,
            color: String,
            cost: Cost,
        }
        for u in self.circuit.get_vertices() {
            let color = match self.circuit.gate_type_or_unreachable(u) {
                Gate::And => "#ff9c9c",
                Gate::Or => "#9fff9c",
                Gate::Variable => "#9cd6ff",
            };
            let cost = self.cost.get(u).cloned().unwrap_or_default();
            meta_writer.serialize(CosmoMeta {
                id: *u,
                color: color.to_string(),
                cost,
            })?;
        }
        meta_writer.flush()?;

        Ok(())
    }

    /// Precompute the costs of each variable vertex, which may correspond to multiple
    /// e-nodes due to circuit simplification. A small adjustment biases the algorithm
    /// against selecting nodes of 0 cost when unnecessary.
    fn compute_costs(&mut self) {
        let epsilon = Cost::new(1e-5).unwrap();
        let mut adjustment = epsilon;
        for u in self.circuit.get_vertices() {
            if self.circuit.gate_type_or_unreachable(u) == &Gate::Variable {
                if let Some(nid_set) = self.id_converter.vid_to_nid_set(u) {
                    let u_cost = nid_set
                        .iter()
                        .map(|nid| self.egraph.nodes.get(nid).unwrap().cost)
                        .sum();
                    if u_cost > epsilon {
                        self.cost.insert(*u, u_cost);
                    } else {
                        self.cost.insert(*u, epsilon);
                        adjustment += epsilon;
                    }
                } else {
                    self.cost.insert(*u, epsilon);
                }
            }
        }
        self.max_cost += adjustment;
    }

    /// Simplify and preprocess the e-graph to remove vertices and
    /// and that can never be extracted. These steps are vaguely inspired
    /// by the simplification steps taken by the ILP solver in the gym,
    /// eventually they should be merged. More simplifications than
    /// what is done here may be helpful.
    fn simplify(&mut self, options: SimplifyOptions) {
        if options.verbose {
            println!("before simplify");
            print_stats(&self.circuit);
        }

        let mut changed = true;
        while changed {
            changed = false;

            // DFS from the extraction target to remove unreachable vertices
            if options.remove_unreachable {
                let mut call_stack: Vec<usize> = Vec::default();
                call_stack.push(self.circuit.root_id);
                let mut visited: IntSet<usize> = IntSet::default();

                while let Some(u) = call_stack.pop() {
                    if visited.contains(&u) {
                        continue;
                    }
                    visited.insert(u);
                    for v in self.circuit.inputs_or_unreachable(&u) {
                        call_stack.push(*v);
                    }
                }

                let vertices = self.circuit.get_vertices().clone();
                for u in vertices {
                    if !visited.contains(&u) {
                        self.circuit.remove_vertex(u);
                        changed = true;
                    }
                }

                if options.verbose {
                    println!("after remove unreachable");
                    print_stats(&self.circuit);
                }
            }
            
            // Suppose u has indeg 1.
            // If v -> u -> others, replace with v -> others.
            // Works because AND and OR gates are identity when there is only
            // one input.
            if options.contract_indegree_one {
                let vertices = self.circuit.get_vertices().clone();
                for u in vertices {
                    if u == self.circuit.root_id {
                        continue;
                    }
                    let inputs = self.circuit.inputs_or_unreachable(&u);
                    if inputs.len() == 1 {
                        let v = inputs.into_iter().next().unwrap();
                        self.circuit.contract_edge_remove_out(*v, u);
                        changed = true;
                    }
                }
                if options.verbose {
                    println!("after contract indegree one");
                    print_stats(&self.circuit);
                }
            }

            // Suppose u and v are both AND or both OR gates, and v has outdeg 1.
            // If others -> v -> u -> others, replace with others -> v -> others
            //              ^                                        ^
            //            others                                   others
            if options.contract_same_gate {
                let vertices = self.circuit.get_vertices().clone();
                for u in vertices {
                    if u == self.circuit.root_id {
                        continue;
                    }
                    let inputs = self.circuit.inputs(&u).cloned().unwrap_or_default();
                    for v in inputs {
                        let outputs_v = self.circuit.outputs(&v).cloned().unwrap_or_default();
                        if outputs_v.len() == 1 {
                            match (self.circuit.gate_type(&u), self.circuit.gate_type(&v)) {
                                (Some(Gate::And), Some(Gate::And)) => {
                                    self.circuit.contract_edge_remove_out(v, u);
                                    changed = true;
                                }
                                (Some(Gate::Or), Some(Gate::Or)) => {
                                    self.circuit.contract_edge_remove_out(v, u);
                                    changed = true;
                                }
                                _ => {}
                            }
                        }
                    }
                }

                if options.verbose {
                    println!("after contract same gate");
                    print_stats(&self.circuit);
                }
            }

            // Any directed cycle with a single OR can be broken.
            // To see this, consider a cycle OR -> AND1 -> AND2 -> AND3 -> OR.
            // If AND3 were true, this would require AND2 to be true, and AND1, and OR.
            // This is a directed cycle of true vertices, so it is safe to delete AND3.
            if options.remove_lone_or_loops {
                let vertices = self.circuit.get_vertices().clone();
                for u in vertices {
                    if self.circuit.gate_type_or_unreachable(&u) != &Gate::Or {
                        continue;
                    }

                    let mut call_stack: Vec<usize> = Vec::default();
                    for v in self.circuit.outputs_or_unreachable(&u).clone() {
                        if self.circuit.gate_type_or_unreachable(&v) == &Gate::And {
                            call_stack.push(v);
                        }
                    }
                    while let Some(v) = call_stack.pop() {
                        for w in self.circuit.outputs_or_unreachable(&v).clone() {
                            if self.circuit.gate_type_or_unreachable(&w) == &Gate::And {
                                call_stack.push(w);
                            }
                            if w == u {
                                self.circuit.remove_vertex(v);
                            }
                        }
                    }
                }
                if options.verbose {
                    println!("after remove loops with one or gate");
                    print_stats(&self.circuit);
                }
            }

            // If u and v are the same gate type, then
            // w --> u
            //   \   ^
            //    -> v 
            // is unnecessary, and can be simplified to w -> v -> u. 
            if options.same_gate_is_tree {
                let vertices = self.circuit.get_vertices().clone();
                for u in &vertices {
                    let inputs = self.circuit.inputs_or_unreachable(u).clone();
                    let mut call_stack: Vec<usize> = Vec::default();
                    for v in inputs.iter() {
                        if self.circuit.gate_type_or_unreachable(v)
                            == self.circuit.gate_type_or_unreachable(u)
                        {
                            call_stack.push(*v);
                        }
                    }
                    while let Some(v) = call_stack.pop() {
                        for w in self.circuit.inputs_or_unreachable(&v).clone() {
                            if self.circuit.gate_type_or_unreachable(&w)
                                == self.circuit.gate_type_or_unreachable(u)
                            {
                                call_stack.push(w);
                            }
                            if inputs.contains(&w) {
                                self.circuit.remove_edge(w, *u);
                                changed = true;
                            }
                        }
                    }
                }
                if options.verbose {
                    println!("after same gate subgraphs are trees");
                    print_stats(&self.circuit);
                }
            }

            // Suppose u is an AND gate, and v1, v2, v3 are OR gates.
            //        /-> v1 -\                        v1 -\
            // Then w --> v2 -> u can be simplified to v2 -> AND -> OR -> u
            //        \-> v3 -/                        v3 -/   w -/
            //
            // Also works if ANDs and ORs are swapped.
            // Note: this option may occasionally increase the graph size, though
            // it is usually beneficial.
            if options.distributivity {
                let vertices = self.circuit.get_vertices().clone();
                for u in &vertices {
                    let target = match self.circuit.gate_type_or_unreachable(u) {
                        Gate::And => Gate::Or,
                        Gate::Or => Gate::And,
                        Gate::Variable => continue,
                    };
                    let mut targets_by_grandinputs: IntMap<usize, IntSet<usize>> =
                        IntMap::default();
                    for v in self.circuit.inputs_or_unreachable(u).clone() {
                        if self.circuit.gate_type_or_unreachable(&v) == &target
                            && self.circuit.outputs_or_unreachable(&v).len() == 1
                        {
                            for w in self.circuit.inputs_or_unreachable(&v).clone() {
                                targets_by_grandinputs.entry(w).or_default().insert(v);
                            }
                        }
                    }
                    let factor_out = targets_by_grandinputs.iter().max_by_key(|(_, vs)| vs.len());
                    if let Some((w, vs)) = factor_out {
                        if vs.len() > 1 {
                            for v in vs {
                                self.circuit.remove_edge(*v, *u);
                                self.circuit.remove_edge(*w, *v);
                            }
                            let target_id = self.id_converter.reserve_id();
                            let bunch_id = self.id_converter.reserve_id();
                            self.circuit.add_vertex(target_id, target);
                            self.circuit
                                .add_vertex(bunch_id, *self.circuit.gate_type_or_unreachable(u));
                            
                            // I'm pretty sure that the following code is not needed.
                            // It attaches a variable node to the new AND gate.
                            // This is necessary for AND nodes that come from e-nodes,
                            // to disable them when their inputs are true (because they are
                            // useful elsewhere) but you don't want to extract it.
                            // However, there is no e-node corresponding to the new AND
                            // gate here, so we are free to avoid the variable.
                            // Leaving it here in case I was wrong.

                            // let variable_id = self.id_converter.reserve_id();
                            // self.circuit.add_vertex(variable_id, Gate::Variable);
                            // match target {
                            //     Gate::And => self.circuit.add_edge(variable_id, target_id),
                            //     Gate::Or => self.circuit.add_edge(variable_id, bunch_id),
                            //     Gate::Variable => unreachable!(),
                            // }
                            self.circuit.add_edge(target_id, *u);
                            self.circuit.add_edge(*w, target_id);
                            self.circuit.add_edge(bunch_id, target_id);
                            for v in vs {
                                self.circuit.add_edge(*v, bunch_id);
                            }
                            changed = true;
                        }
                    }
                }
                if options.verbose {
                    println!("after distributivity");
                    print_stats(&self.circuit);
                }
            }

            // If multiple variable nodes are inputs to the same set of AND gates,
            // they can be merged into a mega variable node (with larger cost)
            // and processed simulataneously. 
            if options.collect_variables {
                let variables: Vec<usize> = self
                    .circuit
                    .get_vertices()
                    .clone()
                    .into_iter()
                    .filter(|u| self.circuit.gate_type_or_unreachable(u) == &Gate::Variable)
                    .collect();
                let mut removed: FxHashSet<usize> = FxHashSet::default();
                for u in &variables {
                    if removed.contains(u) {
                        continue;
                    }
                    let outputs = self.circuit.outputs_or_unreachable(u).clone();
                    if !outputs
                        .iter()
                        .all(|u| self.circuit.gate_type_or_unreachable(u) == &Gate::And)
                    {
                        continue;
                    }
                    for v in &variables {
                        if removed.contains(v) || u == v {
                            continue;
                        }
                        if self.circuit.outputs_or_unreachable(v) == &outputs {
                            self.circuit.remove_vertex(*v);
                            self.id_converter.merge_vid_keep1(*u, *v);
                            removed.insert(*v);
                            changed = true;
                        }
                    }
                }

                if options.verbose {
                    println!("after collect variables");
                    print_stats(&self.circuit);
                }
            }
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
        let nice_td = forget_until_root(to_nice_decomp(&td, &root_bag_id, None), &root_id);

        if VERBOSE {
            println!("treewidth: {}", td.max_bag_size - 1);
            println!("nice td: {} us", start_time.elapsed().as_micros());
        }

        let start_time = std::time::Instant::now();

        // Find the best satisfying assignment
        let dpmap_by_group = nice_td.get_dpmap_by_group(&env);
        let mut root_true_eval = Evaluation::default();
        root_true_eval.map.insert(root_id, true);

        let mut result = ExtractionResult::default();
        if let Some(dpmap) = dpmap_by_group.map.get(&root_true_eval) {
            if let Some((_, best_eval)) = dpmap.map.iter().next() {
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
            println!("extract: {} us", start_time.elapsed().as_micros());
            println!("baseline (+ epsilon): {}", env.max_cost);
        }

        result
    }
}
