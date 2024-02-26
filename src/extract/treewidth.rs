use std::env::var;
use std::error::Error;
use std::hash::{Hash, Hasher, BuildHasherDefault};
use std::collections::HashSet;
use csv::Writer;

use super::*;
use arboretum_td::graph::{HashMapGraph, MutableGraph};
use arboretum_td::solver::Solver;
use arboretum_td::tree_decomposition::TreeDecomposition;

use nohash_hasher::{IntMap, IntSet, NoHashHasher};

pub struct TreewidthExtractor;

/// Converts between usize IDs and ClassIds/NodeIds.
/// IDs for use with arboretum_td are required to be usize.
/// For convenience, this maintains maps in both directions
/// and segregates IDs for Variables, And gates, and Or gates.
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
            self.vid_to_nid_set.entry(self.counter).or_default().insert(nid.clone());
            self.nid_to_vid.insert(nid.clone(), self.counter);
            self.counter += 1;
            self.counter - 1
        }
    }

    fn merge_vid_keep1(&mut self, vid1: usize, vid2: usize) {
        let nid_set1 = self.vid_to_nid_set.get(&vid1).cloned().unwrap();
        let nid_set2 = self.vid_to_nid_set.get(&vid2).cloned().unwrap();
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

    fn vid_to_nid_set_or_unreachable(&self, vid: &usize) -> &FxHashSet<NodeId> {
        self.vid_to_nid_set.get(vid).unwrap_or_else(|| unreachable!())
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Gate {
    Variable,
    Or,
    And,
}

/// A basic implementation of monotone circuits. The circuit
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

    fn to_graph(&self) -> HashMapGraph {
        let mut graph = HashMapGraph::new();
        for u in self.get_vertices() {
            for v in self.inputs_or_unreachable(u) {
                graph.add_edge(*u, *v);
            }
        }
        graph
    }

    fn save_cosmograph(&self, filename_noext: String) -> Result<(), Box<dyn Error>> {
        let mut data_writer = Writer::from_path(format!("{}.csv", filename_noext))?;
        #[derive(serde::Serialize)]
        struct CosmoData {
            input: usize,
            output: usize,
            color: String,
        }
        for u in self.get_vertices() {
            for v in self.outputs(u).unwrap_or(&IntSet::default()) {
                data_writer.serialize(CosmoData {input: *u, output: *v, color: String::from("white")})?;
            }
        }
        data_writer.flush()?;

        let mut meta_writer = Writer::from_path(format!("{}_meta.csv", filename_noext))?;
        #[derive(serde::Serialize)]
        struct CosmoMeta {
            id: usize,
            color: String,
        }
        for u in self.get_vertices() {
            let color = match self.gate_type_or_unreachable(u) {
                Gate::And => "#ff9c9c",
                Gate::Or => "#9fff9c",
                Gate::Variable => "#9cd6ff",
            };
            meta_writer.serialize(CosmoMeta {id: *u, color: color.to_string()})?;
        }
        meta_writer.flush()?;

        Ok(())
    }
}

// Assignments take values from AssignValue. We distinguish vertices known
// to be true based on other assignments made so far and vertices that we
// want to be true. It would be sound to also distinguish vertices known to
// be false and vertices that we want to be false, but the distinction
// turns out to be unnecessary (see other comments for details).
#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
enum AssignValue {
    KnownTrue,
    SupposedTrue,
    False,
}

impl AssignValue {
    fn to_bool(&self) -> bool {
        match self {
            AssignValue::False => false,
            AssignValue::KnownTrue => true,
            AssignValue::SupposedTrue => true,
        }
    }
}

/// A map that represents a valid partial evaluation of a circuit.
/// Uses BTreeMap so that it can be hashed.
#[derive(PartialEq, Eq, Default, Clone, Debug)]
struct Assignment {
    map: IntMap<usize, AssignValue>,
    true_vertices: IntSet<usize>,
    exists_cycle: Option<bool>,
    is_deterministic: Option<bool>,
    cost: Option<Cost>,
}

impl Hash for Assignment {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for elt in &self.map {
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

impl Assignment {
    fn from_map(map: IntMap<usize, AssignValue>) -> Self {
        let true_vertices = map.iter().filter_map(|(u, value)| 
            match value.to_bool() {
                true => Some(*u),
                false => None,
            }
        ).collect();
        Assignment { 
            map, 
            true_vertices,
            exists_cycle: None,
            is_deterministic: None,
            cost: None 
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        let map: HashMap<usize, AssignValue, BuildHasherDefault<NoHashHasher<usize>>> =
                HashMap::with_capacity_and_hasher(capacity, BuildHasherDefault::default());
        Assignment {
            map,
            true_vertices: IntSet::default(),
            exists_cycle: None,
            is_deterministic: None,
            cost: None
        }
    }

    fn insert(&mut self, k: usize, v: AssignValue) {
        if v.to_bool() {
            self.true_vertices.insert(k);
            self.exists_cycle = None; //TODO: optimize these
            self.is_deterministic = None;
            self.cost = None;
        }
        self.map.insert(k, v);
    }

    fn remove(&mut self, k: &usize) {
        if self.true_vertices.contains(k) {
            self.true_vertices.remove(k);
            self.exists_cycle = None; //TODO: optimize these
            self.is_deterministic = None;
            self.cost = None;
        }
        self.map.remove(k);
    }

    fn exists_cycle(&self, env: &Env) -> bool {
        if let Some(answer) = self.exists_cycle {
            return answer;
        } else {
            let mut answer = false;
            let mut status: HashMap<usize, DFSStatus, BuildHasherDefault<NoHashHasher<usize>>> =
                HashMap::with_capacity_and_hasher(self.map.len(), BuildHasherDefault::default());
            let mut call_stack: Vec<usize> = Vec::with_capacity(self.map.len() * 4);
    
            for u in self.true_vertices.iter() {
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
        
    }

    fn is_deterministic(&self, env: &Env) -> bool {
        if let Some(answer) = self.is_deterministic {
            return answer;
        } else {
            for u in &self.true_vertices {
                if env.circuit.gate_type_or_unreachable(u) == &Gate::Or {
                    if env
                        .circuit
                        .inputs_or_unreachable(u)
                        .intersection(&self.true_vertices)
                        .count()
                        >= 2
                    {
                        return false;
                    }
                }
            }
            true
        }
    }

    fn cost(&mut self, env: &Env) -> Cost {
        let cost_u = |u: &usize| -> Cost {
            env.id_converter.vid_to_nid_set_or_unreachable(u).iter().map(|nid|
                env.egraph.nodes.get(nid).unwrap().cost
            ).sum()
        };

        if let Some(answer) = self.cost {
            return answer;
        } else {
            let sum = self.true_vertices.iter()
                .filter(|u| env.circuit.gate_type_or_unreachable(u) == &Gate::Variable)
                .map(|u| cost_u(u))
                .sum();
            self.cost = Some(sum);
            sum
        }
    }

    fn valid(&mut self, env: &Env) -> bool {
        self.is_deterministic(env) && !self.exists_cycle(env) && self.cost(env) <= env.max_cost + 1e-5
    }
}

/// Recall that a tree decomposition places gates into bags, and these
/// bags form a tree. At each bag X, consider the set of all possible
/// assignments on all gates in the subtree rooted at X. We restrict each
/// such assignment to X, then remember each restriction with the full
/// assignment of lowest cost that produced it. This struct maintains this
/// map from restrictions to full assignments.
#[derive(Default, Debug)]
struct ExtendAssigns(FxHashMap<Assignment, Assignment>);

impl ExtendAssigns {
    fn add_if_better(&mut self, env: &Env, bag_assign: Assignment, mut full_assign: Assignment) {
        if full_assign.map.get(&env.circuit.root_id) == Some(&AssignValue::False) {
            return;
        }

        if let Some(self_full_assign) = self.0.get_mut(&bag_assign) {
            let self_cost = self_full_assign.cost(env);
            if self_cost > full_assign.cost(env) {
                self.0.insert(bag_assign, full_assign);
            }
        } else {
            self.0.insert(bag_assign, full_assign);
        }
    }
}

// Interpret as False, SupposedTrue, KnownTrue
struct PossibleValues([bool; 3]);

impl PossibleValues {
    fn from_vertex(u: usize, env: &Env) -> Self {
        match env.circuit.gate_type_or_unreachable(&u) {
            Gate::And => PossibleValues([true, true, true]),
            Gate::Or => PossibleValues([true, true, true]),
            Gate::Variable => PossibleValues([true, false, true]),
        }
    }

    // When inserting x, if some outputs/inputs of x are already assigned,
    // this restricts the values we are allowed to assign x by gate rules.
    fn restrict_by_outputs(&mut self, bag: &NiceBag, bag_assign: &Assignment, full_assign: &Assignment, env: &Env) {
        match bag {
            NiceBag::Insert { vertices, child: _, x } => {
                let outputs = env.circuit.outputs_or_unreachable(x);
                for out in outputs {
                    // Outputs to x which have not yet been considered do not
                    // restrict the possible assignments to x. No outputs to x
                    // can be already forgotten, because that would violate
                    // the definition of tree decomposition. Hence, only outputs
                    // contained in the bag can restrict possible values of x.
                    if !vertices.contains(&out) {
                        continue;
                    }
                    match env.circuit.gate_type_or_unreachable(&out) {
                        Gate::Or => match bag_assign.map.get(&out) {
                            // To be symmetric with And gates, this False case is really both
                            // SupposedFalse and KnownFalse together. But KnownFalse is
                            // unreachable, so we can interpret this False as SupposedFalse.
                            Some(AssignValue::False) => {
                                self.0[1] = false; // SupposedTrue
                                self.0[2] = false; // KnownTrue
                            }
                            Some(AssignValue::KnownTrue) => {}
                            Some(AssignValue::SupposedTrue) => {
                                let siblings = env.circuit.inputs_or_unreachable(&out);
                                if siblings.iter().all(|u| full_assign.map.get(u) == Some(&AssignValue::False) || u == x) {
                                    self.0[0] = false; // False
                                }
                            }
                            None => {
                                unreachable!()
                            }
                        },
                        Gate::And => match bag_assign.map.get(&out) {
                            // To be symmetric with Or gates, this False case is really both
                            // SupposedFalse and KnownFalse together. Although KnownFalse
                            // places no restrictions, we know that a sibling must be false,
                            // so the following check for SupposedFalse will always pass.
                            Some(AssignValue::False) => {
                                let siblings = env.circuit.inputs_or_unreachable(&out);
                                if siblings.iter().all(|u| full_assign.true_vertices.contains(u) || u == x) {
                                    self.0[1] = false; // SupposedTrue
                                    self.0[2] = false; // KnownTrue
                                }
                            }
                            Some(AssignValue::KnownTrue) => {
                                unreachable!()
                            }
                            Some(AssignValue::SupposedTrue) => {
                                self.0[0] = false; // False
                            }
                            None => {}
                        },
                        Gate::Variable => {}
                    }
                }
            }
            _ => unreachable!(),
        }
    }

    fn restrict_by_inputs(&mut self, bag: &NiceBag, bag_assign: &Assignment, env: &Env) {
        match bag {
            NiceBag::Insert { vertices: _, child: _, x, } => {
                match env.circuit.gate_type_or_unreachable(x) {
                    Gate::Or => {
                        let input_colors = FxHashSet::from_iter(
                            env.circuit
                                .inputs_or_unreachable(x)
                                .iter()
                                .map(|input| bag_assign.map.get(input)),
                        );
                        if input_colors.contains(&Some(&AssignValue::KnownTrue))
                            || input_colors.contains(&Some(&AssignValue::SupposedTrue))
                        {
                            self.0[0] = false; // False
                            self.0[1] = false; // SupposedTrue
                        } else if input_colors.contains(&None) {
                            self.0[2] = false; // KnownTrue
                        } else { // All inputs are False
                            self.0[1] = false; // SupposedTrue
                            self.0[2] = false; // KnownTrue
                        }
                    }
                    Gate::And => {
                        let input_colors = FxHashSet::from_iter(
                            env.circuit
                                .inputs_or_unreachable(x)
                                .iter()
                                .map(|input| bag_assign.map.get(input)),
                        );
                        if input_colors.contains(&Some(&AssignValue::False)) {
                            self.0[1] = false; // SupposedTrue
                            self.0[2] = false; // KnownTrue
                        } else if input_colors.contains(&None) {
                            self.0[2] = false; // KnownTrue
                        } else { // All inputs are SupposedTrue or KnownTrue
                            self.0[0] = false; // False
                            self.0[1] = false; // SupposedTrue
                        }
                    }
                    Gate::Variable => {}
                }
            }
            _ => unreachable!(),
        }
    }

    fn restrict(&mut self, bag: &NiceBag, bag_assign: &Assignment, full_assign: &Assignment, env: &Env) {
        self.restrict_by_outputs(bag, bag_assign, full_assign, env);
        self.restrict_by_inputs(bag, bag_assign, env);
    }

    fn to_vec(&self) -> Vec<AssignValue> {
        let mut v: Vec<AssignValue> = Vec::with_capacity(3);
        if self.0[0] {
            v.push(AssignValue::False);
        }
        if self.0[1] {
            v.push(AssignValue::SupposedTrue);
        }
        if self.0[2] {
            v.push(AssignValue::KnownTrue)
        }
        v
    }
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
    fn get_assignments(&self, env: &Env) -> ExtendAssigns {
        let mut ea = ExtendAssigns::default();
        match self {
            NiceBag::Leaf { vertices: _ } => {
                ea.add_if_better(env, Assignment::default(), Assignment::default());
            }

            NiceBag::Insert { vertices: _, child, x } => {
                let child_assignments = child.get_assignments(env).0;
                for (bag_assign, full_assign) in &child_assignments {
                    let mut pv = PossibleValues::from_vertex(*x, env);
                    pv.restrict(self, bag_assign, full_assign, env);
        
                    for value in pv.to_vec() {
                        let mut new_full_assign = full_assign.clone();
                        new_full_assign.insert(*x, value);
                        if !new_full_assign.valid(env) {
                            continue;
                        }

                        let mut new_bag_assign = bag_assign.clone();
                        new_bag_assign.insert(*x, value);

                        // Upgrade SupposedTrue to KnownTrue if needed
                        for u in env.circuit.outputs_or_unreachable(x) {
                            if bag_assign.map.get(u) == Some(&AssignValue::SupposedTrue) {
                                match env.circuit.gate_type(u) {
                                    Some(&Gate::Or) => {
                                        if value.to_bool() {
                                            new_bag_assign.insert(*u, AssignValue::KnownTrue);
                                            new_full_assign.insert(*u, AssignValue::KnownTrue);
                                        }
                                    }
                                    Some(&Gate::And) => {
                                        if env.circuit.inputs_or_unreachable(u).iter().all(|v| {
                                            new_full_assign.map.get(v).map(|value| value.to_bool())
                                                == Some(true)
                                        }) {
                                            new_bag_assign.insert(*u, AssignValue::KnownTrue);
                                            new_full_assign.insert(*u, AssignValue::KnownTrue);
                                        }
                                    }
                                    Some(&Gate::Variable) => {} // case not possible
                                    None => {}
                                }
                            }
                        }
                        ea.add_if_better(env, new_bag_assign, new_full_assign);
                    }
                }
            }
            NiceBag::Forget { vertices: _, child, x } => {
                let child_assignments = child.get_assignments(env).0;

                for (bag_assign, full_assign) in &child_assignments {
                    let mut new_bag_assign = bag_assign.clone();
                    new_bag_assign.remove(x);
                    ea.add_if_better(env, new_bag_assign, full_assign.clone());
                }
            }
            NiceBag::Join { vertices, child1, child2 } => {
                let child1_assignments = child1.get_assignments(env).0;
                let child2_assignments = child2.get_assignments(env).0;
                for (bag_assign1, full_assign1) in &child1_assignments {
                    for (bag_assign2, full_assign2) in &child2_assignments {
                        if bag_assign1.true_vertices != bag_assign2.true_vertices {
                            continue;
                        }

                        let mut new_bag_assign = Assignment::with_capacity(bag_assign1.map.len());
                        let mut valid = true;
                        for u in vertices {
                            match (bag_assign1.map.get(u), bag_assign2.map.get(u)) {
                                (Some(AssignValue::False), Some(AssignValue::False)) => {
                                    new_bag_assign.insert(*u, AssignValue::False);
                                }
                                (Some(AssignValue::KnownTrue), Some(AssignValue::KnownTrue)) => {
                                    new_bag_assign.insert(*u, AssignValue::KnownTrue);
                                }
                                (Some(AssignValue::SupposedTrue), Some(AssignValue::KnownTrue)) => {
                                    new_bag_assign.insert(*u, AssignValue::KnownTrue);
                                }
                                (Some(AssignValue::KnownTrue), Some(AssignValue::SupposedTrue)) => {
                                    new_bag_assign.insert(*u, AssignValue::KnownTrue);
                                }
                                (Some(AssignValue::SupposedTrue), Some(AssignValue::SupposedTrue)) => {
                                    match env.circuit.gate_type(u) {
                                        Some(Gate::Variable) => unreachable!(),
                                        Some(Gate::And) => {
                                            if env.circuit.inputs_or_unreachable(u).iter().all(|v| {
                                                full_assign1.map.contains_key(v)
                                                    || full_assign2.map.contains_key(v)
                                            }) {
                                                new_bag_assign.insert(*u, AssignValue::KnownTrue);
                                            } else {
                                                new_bag_assign.insert(*u, AssignValue::SupposedTrue);
                                            }
                                        }
                                        Some(Gate::Or) => {
                                            if env.circuit.inputs_or_unreachable(u).iter().all(|v| {
                                                full_assign1.map.contains_key(v)
                                                    || full_assign2.map.contains_key(v)
                                            }) {
                                                valid = false;
                                                break;
                                            } else {
                                                new_bag_assign.insert(*u, AssignValue::SupposedTrue);
                                            }
                                        }
                                        None => unreachable!()
                                    }
                                }
                                _ => {
                                    unreachable!();
                                }
                            }
                        }
                        if valid {
                            let mut new_full_assign = new_bag_assign.clone();
                            for (u, value) in full_assign1.map.iter() {
                                if !vertices.contains(u) {
                                    new_full_assign.insert(*u, *value);
                                }
                            }
                            for (u, value) in full_assign2.map.iter() {
                                if !vertices.contains(u) {
                                    new_full_assign.insert(*u, *value);
                                }
                            }

                            if !new_full_assign.valid(env) {
                                continue;
                            }

                            ea.add_if_better(
                                env,
                                new_bag_assign,
                                new_full_assign,
                            );
                        }
                    }
                }
            }    
        }
        ea
    }

    fn vertices(&self) -> &IntSet<usize> {
        match self {
            NiceBag::Leaf { vertices } => vertices,
            NiceBag::Insert { vertices, child: _, x: _ } => vertices,
            NiceBag::Forget { vertices, child: _, x: _ } => vertices,
            NiceBag::Join { vertices, child1: _, child2: _ } => vertices,
        }
    }

    fn new_leaf() -> Self {
        NiceBag::Leaf { vertices: IntSet::default() }
    }

    fn new_insert(vertices: IntSet<usize>, child: Box<NiceBag>, x: usize) -> NiceBag {
        NiceBag::Insert { vertices, child, x }
    }

    fn new_forget(vertices: IntSet<usize>, child: Box<NiceBag>, x: usize) -> NiceBag {
        NiceBag::Forget { vertices, child, x }
    }

    fn new_join(vertices: IntSet<usize>, child1: Box<NiceBag>, child2: Box<NiceBag>) -> NiceBag {
        NiceBag::Join { vertices, child1, child2 }
    }
}



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
        .sum::<f32>()
        / num_verts as f32;
    println!("|V|, |E|/|V| = {}, {}", num_verts, num_edges);
}
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
}

impl<'a> Env<'a> {
    fn new(id_converter: IdConverter, circuit: Circuit, egraph: &'a EGraph, max_cost: Cost) -> Self {
        Env {
            id_converter,
            circuit,
            egraph,
            max_cost,
        }
    }

    fn simplify(&mut self, options: SimplifyOptions) {
        if options.verbose {
            println!("before simplify");
            print_stats(&self.circuit);
        }
    
        let mut changed = true;
        while changed {
            changed = false;
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
    
            if options.same_gate_is_tree {
                let vertices = self.circuit.get_vertices().clone();
                for u in &vertices {
                    let inputs = self.circuit.inputs_or_unreachable(u).clone();
                    let mut call_stack: Vec<usize> = Vec::default();
                    for v in inputs.iter() {
                        if self.circuit.gate_type_or_unreachable(v) == self.circuit.gate_type_or_unreachable(u) {
                            call_stack.push(*v);
                        }
                    }
                    while let Some(v) = call_stack.pop() {
                        for w in self.circuit.inputs_or_unreachable(&v).clone() {
                            if self.circuit.gate_type_or_unreachable(&w) == self.circuit.gate_type_or_unreachable(u) {
                                call_stack.push(w);
                            }
                            if inputs.contains(&w) {
                                self.circuit.remove_edge(w,* u);
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
    
            if options.distributivity {
    
            }
    
            if options.collect_variables {
                let variables: Vec<usize> = self.circuit.get_vertices().clone().into_iter().filter(|u| self.circuit.gate_type_or_unreachable(u) == &Gate::Variable).collect();
                let mut removed: FxHashSet<usize> = FxHashSet::default();
                for u in &variables {
                    if removed.contains(u) {
                        continue;
                    }
                    let outputs = self.circuit.outputs_or_unreachable(u).clone();
                    if !outputs.iter().all(|u| self.circuit.gate_type_or_unreachable(u) == &Gate::And) {
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

impl Extractor for TreewidthExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        let start_time = std::time::Instant::now();

        let max_cost = extract::faster_greedy_dag::FasterGreedyDagExtractor.boxed().extract(egraph, roots).dag_cost(egraph, roots);

        // Make ClassIds and NodeIds compatible with arboretum_td
        let mut id_converter = IdConverter::default();
        for (cid, class) in egraph.classes() {
            id_converter.get_oid_or_add_class(cid);
            for nid in class.nodes.iter() {
                id_converter.get_aid_or_add_node(nid);
                id_converter.get_vid_or_add_node(nid);
            }
        }
        println!("{}", egraph.classes().len());
        println!("{}", egraph.nodes.len());

        // Create circuit by replacing e-classes with Or gates and e-nodes
        // with both an And gate (to its input classes) and a Variable gate
        // under the And gate (selecting the e-node for extraction)
        let root_id = id_converter.reserve_id();
        println!("root id: {}", root_id);

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
            verbose: true,
        };
        let mut env = Env::new(id_converter, circuit, egraph, max_cost);

        env.simplify(options);
        env.circuit.save_cosmograph("cosmo".to_string()).expect("cosmo failed");


        let graph = env.circuit.to_graph();

        println!("preprocessing: {} us", start_time.elapsed().as_micros());
        let start_time = std::time::Instant::now();

        // Run tree decomposition and identify a bag with the root
        let td = Solver::default_heuristic().solve(&graph);

        //println!("td: {:#?}", td);

        println!("td: {} us", start_time.elapsed().as_micros());
        let start_time = std::time::Instant::now();

        let mut root_bag_id: usize = 0;
        for bag in td.bags() {
            if bag.vertex_set.contains(&root_id) {
                root_bag_id = bag.id;
                break;
            }
        }
        println!("root bag id: {}", root_bag_id);
        let nice_td = forget_until_root(to_nice_decomp(&td, &root_bag_id, None), &root_id);

        // println!("number of bags: {}", nice_td.n_bags());
        // println!("average bag size: {}", nice_td.avg_bag_size());
        println!("max bag size: {}", td.max_bag_size);

        println!("nice td: {} us", start_time.elapsed().as_micros());
        let start_time = std::time::Instant::now();

        // Find the best satisfying assignment
        let ea = nice_td.get_assignments(&env);
        let mut root_assign_map: IntMap<usize, AssignValue> = IntMap::default();
        root_assign_map.insert(root_id, AssignValue::KnownTrue);
        let root_assign = Assignment::from_map(root_assign_map);

        let mut result = ExtractionResult::default();
        if let Some(best_assign) = ea.0.get(&root_assign) {
            for (u, value) in best_assign.map.iter() {
                if let Some(nid_set) = env.id_converter.vid_to_nid_set(u) {
                    for nid in nid_set {
                        if value == &AssignValue::KnownTrue || value == &AssignValue::SupposedTrue {
                            result.choose(egraph.nid_to_cid(nid).clone(), nid.clone());
                        }
                    }
                }
            }
        }
        println!("{:#?}", ea);


        println!("extract: {} us", start_time.elapsed().as_micros());

        result
    }
}
