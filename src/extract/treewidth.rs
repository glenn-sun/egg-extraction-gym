use std::collections::BTreeMap;

use super::*;
use arboretum_td::graph::{BaseGraph, HashMapGraph, MutableGraph};
use arboretum_td::solver::Solver;
use arboretum_td::tree_decomposition::TreeDecomposition;

pub struct TreewidthExtractor;

/// Converts between usize IDs and ClassIds/NodeIds.
/// IDs for use with arboretum_td are required to be usize.
/// For convenience, this maintains maps in both directions
/// and segregates IDs for Variables, And gates, and Or gates.
#[derive(Debug)]
struct IdConverter {
    oid_to_cid: FxHashMap<usize, ClassId>,
    aid_to_nid: FxHashMap<usize, NodeId>,
    vid_to_nid: FxHashMap<usize, NodeId>,
    cid_to_oid: FxHashMap<ClassId, usize>,
    nid_to_aid: FxHashMap<NodeId, usize>,
    nid_to_vid: FxHashMap<NodeId, usize>,
    counter: usize,
}

impl IdConverter {
    fn new() -> Self {
        IdConverter {
            oid_to_cid: FxHashMap::default(),
            aid_to_nid: FxHashMap::default(),
            vid_to_nid: FxHashMap::default(),
            cid_to_oid: FxHashMap::default(),
            nid_to_aid: FxHashMap::default(),
            nid_to_vid: FxHashMap::default(),
            counter: 0,
        }
    }

    fn add_class(&mut self, cid: ClassId) {
        self.oid_to_cid.insert(self.counter, cid.clone());
        self.cid_to_oid.insert(cid, self.counter);
        self.counter += 1;
    }

    fn add_node(&mut self, nid: NodeId) {
        self.aid_to_nid.insert(self.counter, nid.clone());
        self.nid_to_aid.insert(nid.clone(), self.counter);
        self.counter += 1;

        self.vid_to_nid.insert(self.counter, nid.clone());
        self.nid_to_vid.insert(nid, self.counter);
        self.counter += 1;
    }

    /// Reserve an ID that does not correspond to any e-class or e-node.
    fn reserve_id(&mut self) -> usize {
        self.counter += 1;
        self.counter - 1
    }

    fn oid_to_cid(&self, oid: &usize) -> Option<&ClassId> {
        self.oid_to_cid.get(oid)
    }
    fn aid_to_nid(&self, aid: &usize) -> Option<&NodeId> {
        self.aid_to_nid.get(aid)
    }
    fn vid_to_nid(&self, vid: &usize) -> Option<&NodeId> {
        self.vid_to_nid.get(vid)
    }
    fn cid_to_oid(&self, cid: &ClassId) -> Option<&usize> {
        self.cid_to_oid.get(cid)
    }
    fn nid_to_aid(&self, nid: &NodeId) -> Option<&usize> {
        self.nid_to_aid.get(nid)
    }
    fn nid_to_vid(&self, nid: &NodeId) -> Option<&usize> {
        self.nid_to_vid.get(nid)
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
    vertices: FxHashSet<usize>,
    inputs: FxHashMap<usize, FxHashSet<usize>>,
    outputs: FxHashMap<usize, FxHashSet<usize>>,
    gate_type: FxHashMap<usize, Gate>,
    root_id: usize,
}

impl Circuit {
    fn new(root_id: usize) -> Self {
        Circuit {
            vertices: FxHashSet::default(),
            inputs: FxHashMap::default(),
            outputs: FxHashMap::default(),
            gate_type: FxHashMap::default(),
            root_id,
        }
    }

    fn add_vertex(&mut self, u: usize, gate_type: Gate) {
        self.vertices.insert(u);
        self.gate_type.insert(u, gate_type);
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

    fn get_vertices(&self) -> &FxHashSet<usize> {
        &self.vertices
    }
    fn inputs(&self, u: &usize) -> Option<&FxHashSet<usize>> {
        self.inputs.get(u)
    }
    fn outputs(&self, u: &usize) -> Option<&FxHashSet<usize>> {
        self.outputs.get(u)
    }
    fn gate_type(&self, u: &usize) -> Option<&Gate> {
        self.gate_type.get(u)
    }
}

/// An environment to be passed around various functions.
struct Env<'a> {
    id_converter: IdConverter,
    circuit: Circuit,
    egraph: &'a EGraph,
}

impl<'a> Env<'a> {
    fn new(id_converter: IdConverter, circuit: Circuit, egraph: &'a EGraph) -> Self {
        Env {
            id_converter,
            circuit,
            egraph,
        }
    }
}

// Assignments take values from AssignValue. We distinguish vertices known
// to be true based on other assignments made so far and vertices that we
// want to be true.
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
#[derive(PartialEq, Eq, Hash, Default, Clone, Debug)]
struct Assignment {
    assignment: BTreeMap<usize, AssignValue>,
    cost: Cost,
}

impl Assignment {
    fn new(assignment: BTreeMap<usize, AssignValue>, env: &Env) -> Self {
        let cost = Assignment::cost(&assignment, env);
        Assignment { assignment, cost }
    }

    fn get_true_vertices(assignment: &BTreeMap<usize, AssignValue>) -> FxHashSet<usize> {
        assignment.iter().filter_map(|(u, value)| 
        match value.to_bool() {
                true => Some(*u),
                false => None,
            }
        ).collect()
    }

    fn exists_cycle(true_vertices: &FxHashSet<usize>, env: &Env) -> bool {
        let mut exists_cycle = false;
        let mut current_path: FxHashSet<usize> = FxHashSet::default();
        let mut visited: FxHashSet<usize> = FxHashSet::default();

        #[derive(PartialEq, Eq, Clone, Copy, Debug)]
        enum Action {
            Insert,
            Forget,
        }

        let mut call_stack: Vec<(usize, Action)> = Vec::default();
        for u in true_vertices.iter() {
            call_stack.push((*u, Action::Forget));
            call_stack.push((*u, Action::Insert));
        }

        while let Some((u, a)) = call_stack.pop() {
            match a {
                Action::Forget => {
                    current_path.remove(&u);
                }
                Action::Insert => {
                    if visited.contains(&u) {
                        continue;
                    }
                    current_path.insert(u);
                    visited.insert(u);
                    for v in env.circuit.outputs(&u).cloned().unwrap_or_default() {
                        if true_vertices.contains(&v) {
                            if current_path.contains(&v) {
                                exists_cycle = true;
                                break;
                            }
                            if !visited.contains(&v) {
                                call_stack.push((v, Action::Forget));
                                call_stack.push((v, Action::Insert));
                            }
                        }
                    }
                }
            }
        }

        exists_cycle
    }

    fn is_deterministic(true_vertices: &FxHashSet<usize>, env: &Env) -> bool {
        for u in true_vertices {
            if env.circuit.gate_type(u) == Some(&Gate::Or) {
                if env.circuit.inputs(u).cloned().unwrap_or_default().intersection(true_vertices).count() >= 2 {
                    return false;
                }
            }
        }
        true
    }

    fn cost(assignment: &BTreeMap<usize, AssignValue>, env: &Env) -> Cost {
        // The current implementation uses DAG cost as the cost function.
        // If the assignment contains a cycle, set the cost to infinity.
        // This can be adapted to other cost functions, but they should be
        // defined on all subgraphs of e-graphs (including cyclic graphs and
        // disconnected graphs), monotone, and preserve order under common
        // unions.
        let true_vertices = Assignment::get_true_vertices(assignment);

        if !Assignment::is_deterministic(&true_vertices, env) {
            return Cost::new(f64::INFINITY).unwrap();
        }

        if Assignment::exists_cycle(&true_vertices, env) {
            return Cost::new(f64::INFINITY).unwrap();
        }

        let mut sum = Cost::default();
        for u in Assignment::get_true_vertices(assignment) {
            if env.circuit.gate_type(&u) == Some(&Gate::Variable) {
                if let Some(node) = env
                    .id_converter
                    .vid_to_nid(&u)
                    .and_then(|nid| env.egraph.nodes.get(nid))
                {
                    sum += node.cost;
                }
            }
        }
        sum
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
    fn add_if_better(&mut self, env: &Env, bag_assign: Assignment, full_assign: Assignment) {
        if full_assign.assignment.get(&env.circuit.root_id) == Some(&AssignValue::False) {
            return;
        }

        let cost = full_assign.cost;
        if cost.is_finite() {
            if let Some(self_full_assign) = self.0.get(&bag_assign) {
                if self_full_assign.cost > cost {
                    self.0.insert(bag_assign, full_assign);
                }
            } else {
                self.0.insert(bag_assign, full_assign);
            }
        }
    }
}

/// A bag for a nice tree decomposition. A nice tree decomposition is one
/// in which every bag is of the form "empty leaf", "insert u", "forget u",
/// or "join two identical bags".
trait NiceBag: std::fmt::Debug {
    fn get_assignments(&self, env: &Env) -> ExtendAssigns;
    fn vertices(&self) -> &FxHashSet<usize>;
    fn n_bags(&self) -> usize;
    fn avg_bag_size(&self) -> f32;
}

#[derive(Debug)]
struct Leaf {
    vertices: FxHashSet<usize>,
    n_bags: usize,
    avg_bag_size: f32,
}

impl Leaf {
    fn new() -> Self {
        Leaf {
            vertices: FxHashSet::default(),
            n_bags: 1,
            avg_bag_size: 0.0,
        }
    }
}

impl NiceBag for Leaf {
    fn get_assignments(&self, _env: &Env) -> ExtendAssigns {
        let mut ea = ExtendAssigns::default();
        ea.0.insert(Assignment::default(), Assignment::default());
        ea
    }

    fn vertices(&self) -> &FxHashSet<usize> {
        &self.vertices
    }

    fn n_bags(&self) -> usize {
        self.n_bags
    }

    fn avg_bag_size(&self) -> f32 {
        self.avg_bag_size
    }
}

#[derive(Debug)]
struct Insert {
    vertices: FxHashSet<usize>,
    child: Box<dyn NiceBag>,
    x: usize,
    n_bags: usize,
    avg_bag_size: f32,
}

impl Insert {
    fn new(vertices: FxHashSet<usize>, child: Box<dyn NiceBag>, x: usize) -> Self {
        let child_n_bags = child.n_bags();
        let child_bag_size = child.avg_bag_size();
        let size = vertices.len();
        Insert {
            vertices,
            child,
            x,
            n_bags: child_n_bags + 1,
            avg_bag_size: (child_n_bags as f32 * child_bag_size + size as f32)
                / (child_n_bags + 1) as f32,
        }
    }
}

struct AssignValueIndicator {
    pub False: bool,
    pub KnownTrue: bool,
    pub SupposedTrue: bool,
}

impl AssignValueIndicator {
    fn to_vec(&self) -> Vec<AssignValue> {
        let mut out: Vec<AssignValue> = Vec::default();
        if self.False {
            out.push(AssignValue::False);
        }
        if self.KnownTrue {
            out.push(AssignValue::KnownTrue);
        }
        if self.SupposedTrue {
            out.push(AssignValue::SupposedTrue);
        }
        out
    }

    // When inserting x, if some outputs/inputs of x are already assigned,
    // this restricts the values we are allowed to assign x by gate rules. 
    fn restrict_by_outputs(
        &mut self,
        env: &Env,
        bag: &Insert,
        bag_assign: &Assignment,
        full_assign: &Assignment,
    ) {
        let outputs = env.circuit.outputs(&bag.x).cloned().unwrap_or_default();
        for out in outputs {
            if !bag.vertices.contains(&out) {
                continue;
            }
            match env.circuit.gate_type(&out) {
                Some(Gate::Or) => match bag_assign.assignment.get(&out) {
                    Some(AssignValue::False) => {
                        self.KnownTrue = false;
                        self.SupposedTrue = false;
                    }
                    Some(AssignValue::KnownTrue) => {
                        self.KnownTrue = false;
                        self.SupposedTrue = false;
                    }
                    Some(AssignValue::SupposedTrue) => {
                        let siblings: FxHashSet<usize> =
                            env.circuit.inputs(&out).cloned().unwrap_or_default();
                        let mut set_gates: FxHashSet<usize> =
                            full_assign.assignment.keys().cloned().collect();
                        set_gates.insert(bag.x);
                        if siblings.is_subset(&set_gates) {
                            self.False = false;
                        }
                    }
                    None => {}
                },
                Some(Gate::And) => {
                    match bag_assign.assignment.get(&out) {
                        Some(AssignValue::False) => {}
                        Some(AssignValue::KnownTrue) => {
                            // case not possible
                        }
                        Some(AssignValue::SupposedTrue) => {
                            self.False = false;
                        }
                        None => {}
                    }
                }
                Some(Gate::Variable) => {}
                None => {}
            }
        }
    }

    fn restrict_by_inputs(
        &mut self,
        env: &Env,
        bag: &Insert,
        bag_assign: &Assignment,
        full_assign: &Assignment,
    ) {
        match env.circuit.gate_type(&bag.x) {
            Some(Gate::Variable) => {
                self.SupposedTrue = false;
            }
            Some(Gate::Or) => {
                let input_colors = FxHashSet::from_iter(
                    env.circuit
                        .inputs(&bag.x)
                        .cloned()
                        .unwrap_or_default()
                        .iter()
                        .map(|input| bag_assign.assignment.get(input)),
                );
                if input_colors.contains(&Some(&AssignValue::KnownTrue))
                    || input_colors.contains(&Some(&AssignValue::SupposedTrue))
                {
                    self.False = false;
                    self.SupposedTrue = false;
                } else if input_colors.contains(&None) {
                    self.KnownTrue = false;
                } else {
                    self.KnownTrue = false;
                    self.SupposedTrue = false;
                }
            }
            Some(Gate::And) => {
                let input_colors = FxHashSet::from_iter(
                    env.circuit
                        .inputs(&bag.x)
                        .cloned()
                        .unwrap_or_default()
                        .iter()
                        .map(|input| bag_assign.assignment.get(input)),
                );
                if input_colors.contains(&Some(&AssignValue::False)) {
                    self.KnownTrue = false;
                    self.SupposedTrue = false;
                } else if input_colors.contains(&None) {
                    self.KnownTrue = false;
                } else {
                    self.False = false;
                    self.SupposedTrue = false;
                }
            }
            None => {}
        }
    }

    fn restrict(
        &mut self,
        env: &Env,
        bag: &Insert,
        bag_assign: &Assignment,
        full_assign: &Assignment,
    ) {
        if env.circuit.gate_type(&bag.x) == Some(&Gate::Variable) {
            self.SupposedTrue = false;
        }
        self.restrict_by_outputs(env, bag, bag_assign, full_assign);
        self.restrict_by_inputs(env, bag, bag_assign, full_assign);
    }
}

impl Default for AssignValueIndicator {
    fn default() -> Self {
        AssignValueIndicator {
            False: true,
            KnownTrue: true,
            SupposedTrue: true,
        }
    }
}

impl NiceBag for Insert {
    // For an insert node, add all possible assignments to the new vertex, and upgrade
    // any SupposedTrue outputs 
    fn get_assignments<'a>(&self, env: &Env) -> ExtendAssigns {
        let mut ea = ExtendAssigns::default();
        let child_assignments = self.child.get_assignments(env).0;
        for (bag_assign, full_assign) in &child_assignments {
            let mut allowed_values = AssignValueIndicator::default();
            allowed_values.restrict(env, self, bag_assign, full_assign);

            for value in allowed_values.to_vec() {
                let mut new_bag_assign = bag_assign.assignment.clone();
                let mut new_full_assign = full_assign.assignment.clone();
                new_bag_assign.insert(self.x, value);
                new_full_assign.insert(self.x, value);

                let parents: FxHashSet<usize> =
                    env.circuit.outputs(&self.x).cloned().unwrap_or_default();
                for p in parents {
                    if bag_assign.assignment.get(&p) == Some(&AssignValue::SupposedTrue) {
                        match env.circuit.gate_type(&p) {
                            Some(&Gate::Or) => {
                                if value.to_bool() {
                                    new_bag_assign.insert(p, AssignValue::KnownTrue);
                                    new_full_assign.insert(p, AssignValue::KnownTrue);
                                }
                            }
                            Some(&Gate::And) => {
                                if env.circuit.inputs(&p).cloned().unwrap().iter().all(|u| {
                                    new_full_assign.get(u).map(|value| value.to_bool())
                                        == Some(true)
                                }) {
                                    new_bag_assign.insert(p, AssignValue::KnownTrue);
                                    new_full_assign.insert(p, AssignValue::KnownTrue);
                                }
                            }
                            Some(&Gate::Variable) => {} // case not possible
                            None => {}
                        }
                    }
                }

                ea.add_if_better(
                    env,
                    Assignment::new(new_bag_assign, env),
                    Assignment::new(new_full_assign, env),
                );
            }
        }
        ea
    }

    fn vertices(&self) -> &FxHashSet<usize> {
        &self.vertices
    }

    fn n_bags(&self) -> usize {
        self.n_bags
    }

    fn avg_bag_size(&self) -> f32 {
        self.avg_bag_size
    }
}

#[derive(Debug)]
struct Forget {
    vertices: FxHashSet<usize>,
    child: Box<dyn NiceBag>,
    x: usize,
    n_bags: usize,
    avg_bag_size: f32,
}

impl Forget {
    fn new(vertices: FxHashSet<usize>, child: Box<dyn NiceBag>, x: usize) -> Self {
        let child_n_bags = child.n_bags();
        let child_bag_size = child.avg_bag_size();
        let size = vertices.len();
        Forget {
            vertices,
            child,
            x,
            n_bags: child_n_bags + 1,
            avg_bag_size: (child_n_bags as f32 * child_bag_size + size as f32)
                / (child_n_bags + 1) as f32,
        }
    }
}

// When forgetting x, project onto the rest of the bag and keep only the best
impl NiceBag for Forget {
    fn get_assignments(&self, env: &Env) -> ExtendAssigns {
        let mut ea = ExtendAssigns::default();
        let child_assignments = self.child.get_assignments(env).0;
        for (bag_assign, full_assign) in &child_assignments {
            let mut new_bag_assign = bag_assign.clone();
            new_bag_assign.assignment.remove(&self.x);
            ea.add_if_better(env, new_bag_assign, full_assign.clone());
        }
        ea
    }

    fn vertices(&self) -> &FxHashSet<usize> {
        &self.vertices
    }

    fn n_bags(&self) -> usize {
        self.n_bags
    }

    fn avg_bag_size(&self) -> f32 {
        self.avg_bag_size
    }
}

#[derive(Debug)]
struct Join {
    vertices: FxHashSet<usize>,
    child1: Box<dyn NiceBag>,
    child2: Box<dyn NiceBag>,
    n_bags: usize,
    avg_bag_size: f32,
}

impl Join {
    fn new(vertices: FxHashSet<usize>, child1: Box<dyn NiceBag>, child2: Box<dyn NiceBag>) -> Self {
        let child1_n_bags = child1.n_bags();
        let child2_n_bags = child2.n_bags();
        let child1_bag_size = child1.avg_bag_size();
        let child2_bag_size = child2.avg_bag_size();
        let size = vertices.len();
        Join {
            vertices,
            child1,
            child2,
            n_bags: child1_n_bags + child2_n_bags + 1,
            avg_bag_size: (child1_n_bags as f32 * child1_bag_size
                + child2_n_bags as f32 * child2_bag_size
                + size as f32)
                / (child1_n_bags + child2_n_bags + 1) as f32,
        }
    }
}

impl NiceBag for Join {
    fn get_assignments(&self, env: &Env) -> ExtendAssigns {
        let mut ea = ExtendAssigns::default();
        let child1_assignments = self.child1.get_assignments(env).0;
        let child2_assignments = self.child2.get_assignments(env).0;
        for (bag_assign1, full_assign1) in &child1_assignments {
            for (bag_assign2, full_assign2) in &child2_assignments {
                let mut new_bag_assign: BTreeMap<usize, AssignValue> = BTreeMap::default();
                let mut valid = true;
                for u in self.vertices.iter() {
                    match (bag_assign1.assignment.get(u), bag_assign2.assignment.get(u)) {
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
                                Some(Gate::Variable) => {}
                                Some(Gate::And) => {
                                    if env.circuit.inputs(u).cloned().unwrap().iter().all(|v| {
                                        full_assign1.assignment.contains_key(v)
                                            || full_assign2.assignment.contains_key(v)
                                    }) {
                                        new_bag_assign.insert(*u, AssignValue::KnownTrue);
                                    } else {
                                        new_bag_assign.insert(*u, AssignValue::SupposedTrue);
                                    }
                                }
                                Some(Gate::Or) => {
                                    if env.circuit.inputs(u).cloned().unwrap().iter().all(|v| {
                                        full_assign1.assignment.contains_key(v)
                                            || full_assign2.assignment.contains_key(v)
                                    }) {
                                        valid = false;
                                        break;
                                    } else {
                                        new_bag_assign.insert(*u, AssignValue::SupposedTrue);
                                    }
                                }
                                None => {}
                            }
                        }
                        _ => {
                            valid = false;
                            break;
                        }
                    }
                }
                if valid {
                    let mut new_full_assign = new_bag_assign.clone();
                    for (u, value) in full_assign1.assignment.iter() {
                        if !self.vertices.contains(u) {
                            new_full_assign.insert(*u, *value);
                        }
                    }
                    for (u, value) in full_assign2.assignment.iter() {
                        if !self.vertices.contains(u) {
                            new_full_assign.insert(*u, *value);
                        }
                    }
                    ea.add_if_better(
                        env,
                        Assignment::new(new_bag_assign, env),
                        Assignment::new(new_full_assign, env),
                    );
                }
            }
        }
        ea
    }

    fn vertices(&self) -> &FxHashSet<usize> {
        &self.vertices
    }

    fn n_bags(&self) -> usize {
        self.n_bags
    }

    fn avg_bag_size(&self) -> f32 {
        self.avg_bag_size
    }
}

fn to_nice_decomp(
    td: &TreeDecomposition,
    root_bag_id: &usize,
    parent_id: Option<&usize>,
) -> Box<dyn NiceBag> {
    let root_bag = &td.bags[*root_bag_id];
    let mut child_ids = root_bag.neighbors.clone();
    parent_id.map(|u| child_ids.remove(u));
    if child_ids.is_empty() {
        let mut prev: Box<dyn NiceBag> = Box::new(Leaf::new());
        let mut vertices: FxHashSet<usize> = FxHashSet::default();
        for u in root_bag.vertex_set.iter() {
            vertices.insert(*u);
            let next = Insert::new(vertices.clone(), prev, *u);
            prev = Box::new(next);
        }
        return prev;
    }

    let mut same_as_root_bag: Vec<Box<dyn NiceBag>> = Vec::default();

    let child_nice_bags = child_ids
        .iter()
        .map(|u| to_nice_decomp(td, u, Some(root_bag_id)));
    for child_nice_bag in child_nice_bags {
        let mut prev = child_nice_bag;
        let mut vertices = prev.vertices().clone();
        let vertices_clone = vertices.clone();

        let root_vertices = FxHashSet::from_iter(root_bag.vertex_set.clone().into_iter());
        let root_not_child = root_vertices.difference(&vertices_clone);
        let child_not_root = vertices_clone.difference(&root_vertices);

        for u in child_not_root {
            vertices.remove(u);
            let next = Forget::new(vertices.clone(), prev, *u);
            prev = Box::new(next);
        }
        for u in root_not_child {
            vertices.insert(*u);
            let next = Insert::new(vertices.clone(), prev, *u);
            prev = Box::new(next);
        }
        same_as_root_bag.push(prev);
    }

    if let Some(mut prev) = same_as_root_bag.pop() {
        let vertices = prev.vertices().clone();
        for bag in same_as_root_bag {
            let join = Join::new(vertices.clone(), prev, bag);
            prev = Box::new(join);
        }
        return prev;
    }

    Box::new(Leaf::new())
}

fn forget_until_root(nice_td: Box<dyn NiceBag>, root_id: &usize) -> Box<dyn NiceBag> {
    let mut prev = nice_td;
    let mut vertices = prev.vertices().clone();
    let vertices_clone = vertices.clone();
    for u in vertices_clone.iter() {
        if u != root_id {
            vertices.remove(u);
            let next = Forget::new(vertices.clone(), prev, *u);
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
        .map(|u| circuit.inputs(u).cloned().unwrap_or_default().len() as f32)
        .sum::<f32>()
        / num_verts as f32;
    println!("|V|, |E|/|V| = {}, {}", num_verts, num_edges);
}
struct SimplifyOptions {
    pub remove_unreachable: bool,
    pub contract_indegree_one: bool,
    pub contract_same_gate: bool,
    pub remove_self_loops: bool,
    pub verbose: bool,
}

impl Default for SimplifyOptions {
    fn default() -> Self {
        SimplifyOptions {
            remove_unreachable: true,
            contract_indegree_one: true,
            contract_same_gate: true,
            remove_self_loops: true,
            verbose: false,
        }
    }
}

fn simplify(mut circuit: Circuit, options: SimplifyOptions) -> Circuit {
    if options.verbose {
        println!("before simplify");
        print_stats(&circuit);
    }

    let mut changed = true;
    while changed {
        changed = false;
        if options.remove_unreachable {
            let mut call_stack: Vec<usize> = Vec::default();
            call_stack.push(circuit.root_id);
            let mut visited: FxHashSet<usize> = FxHashSet::default();

            while let Some(u) = call_stack.pop() {
                if visited.contains(&u) {
                    continue;
                }
                visited.insert(u);
                if let Some(inputs) = circuit.inputs(&u) {
                    for v in inputs {
                        call_stack.push(*v);
                    }
                }
            }

            let vertices = circuit.get_vertices().clone();
            for u in vertices {
                if !visited.contains(&u) {
                    circuit.remove_vertex(u);
                    changed = true;
                }
            }

            if options.verbose {
                println!("after remove unreachable");
                print_stats(&circuit);
            }
        }

        if options.contract_indegree_one {
            let vertices = circuit.get_vertices().clone();
            for u in vertices {
                if u == circuit.root_id {
                    continue;
                }
                let inputs = circuit.inputs(&u).cloned().unwrap_or_default();
                if inputs.len() == 1 {
                    let v = inputs.into_iter().next().unwrap_or_default();
                    circuit.contract_edge_remove_out(v, u);
                    changed = true;
                }
            }
            if options.verbose {
                println!("after contract indegree one");
                print_stats(&circuit);
            }
        }

        if options.contract_same_gate {
            let vertices = circuit.get_vertices().clone();
            for u in vertices {
                if u == circuit.root_id {
                    continue;
                }
                let inputs = circuit.inputs(&u).cloned().unwrap_or_default();
                for v in inputs {
                    let outputs_v = circuit.outputs(&v).cloned().unwrap_or_default();
                    if outputs_v.len() == 1 {
                        match (circuit.gate_type(&u), circuit.gate_type(&v)) {
                            (Some(Gate::And), Some(Gate::And)) => {
                                circuit.contract_edge_remove_out(v, u);
                                changed = true;
                            }
                            (Some(Gate::Or), Some(Gate::Or)) => {
                                circuit.contract_edge_remove_out(v, u);
                                changed = true;
                            }
                            _ => {}
                        }
                    }
                }
            }

            if options.verbose {
                println!("after contract same gate");
                print_stats(&circuit);
            }
        }

        if options.remove_self_loops {
            let vertices = circuit.get_vertices().clone();
            for u in vertices {
                let inputs = circuit.inputs(&u).cloned().unwrap_or_default();
                let outputs = circuit.outputs(&u).cloned().unwrap_or_default();
                let mut intersection = inputs.intersection(&outputs);
                if intersection
                    .next()
                    .is_some_and(|u| circuit.gate_type(u) == Some(&Gate::Or))
                {
                    circuit.remove_vertex(u);
                    changed = true;
                }
            }
            if options.verbose {
                println!("after remove self loops");
                print_stats(&circuit);
            }
        }
    }
    circuit
}

impl Extractor for TreewidthExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        let start_time = std::time::Instant::now();

        // Make ClassIds and NodeIds compatible with arboretum_td
        let mut id_converter = IdConverter::new();
        for (cid, class) in egraph.classes() {
            id_converter.add_class(cid.clone());
            for nid in class.nodes.iter() {
                id_converter.add_node(nid.clone());
            }
        }

        // Create circuit by replacing e-classes with Or gates and e-nodes
        // with both an And gate (to its input classes) and a Variable gate
        // under the And gate (selecting the e-node for extraction)
        let root_id = id_converter.reserve_id();

        let mut circuit = Circuit::new(root_id);
        for (cid, class) in egraph.classes() {
            if let Some(u) = id_converter.cid_to_oid(cid) {
                circuit.add_vertex(*u, Gate::Or);

                for nid in class.nodes.iter() {
                    if let Some(v) = id_converter.nid_to_aid(nid) {
                        circuit.add_vertex(*v, Gate::And);
                        circuit.add_edge(*v, *u);

                        if let Some(node) = egraph.nodes.get(nid) {
                            for child_nid in node.children.iter() {
                                if let Some(w) =
                                    id_converter.cid_to_oid(egraph.nid_to_cid(child_nid))
                                {
                                    circuit.add_edge(*w, *v);
                                }
                            }
                        }

                        if let Some(w) = id_converter.nid_to_vid(nid) {
                            circuit.add_vertex(*w, Gate::Variable);
                            circuit.add_edge(*w, *v);
                        }
                    }
                }
            }
        }

        // Require extraction of all roots by joining all the root e-classes
        // to a new And gate at the top.
        circuit.add_vertex(root_id, Gate::And);
        for root in roots {
            if let Some(v) = id_converter.cid_to_oid(root) {
                circuit.add_edge(*v, root_id);
            }
        }

        let options = SimplifyOptions {
            remove_unreachable: true,
            contract_indegree_one: true,
            contract_same_gate: true,
            remove_self_loops: true,
            verbose: true,
        };
        circuit = simplify(circuit, options);

        // Build underlying undirected graph from circuit
        let mut graph = HashMapGraph::new();
        for u in circuit.get_vertices() {
            if let Some(vs) = circuit.inputs(u) {
                for v in vs {
                    graph.add_edge(*u, *v);
                }
            }
        }

        println!("preprocessing: {} us", start_time.elapsed().as_micros());
        let start_time = std::time::Instant::now();

        // Run tree decomposition and identify a bag with the root
        let td = Solver::default_exact().solve(&graph);

        println!("td: {} us", start_time.elapsed().as_micros());
        let start_time = std::time::Instant::now();

        let mut root_bag_id: usize = 0;
        for bag in td.bags() {
            if bag.vertex_set.contains(&root_id) {
                root_bag_id = bag.id;
                break;
            }
        }

        let nice_td = forget_until_root(to_nice_decomp(&td, &root_bag_id, None), &root_id);

        println!("number of bags: {}", nice_td.n_bags());
        println!("average bag size: {}", nice_td.avg_bag_size());
        println!("max bag size: {}", td.max_bag_size);

        println!("nice td: {} us", start_time.elapsed().as_micros());
        let start_time = std::time::Instant::now();

        // Find the best satisfying assignment
        let env = Env::new(id_converter, circuit, egraph);
        let ea = nice_td.get_assignments(&env);
        let mut root_assign_map: BTreeMap<usize, AssignValue> = BTreeMap::default();
        root_assign_map.insert(root_id, AssignValue::KnownTrue);
        let root_assign = Assignment::new(root_assign_map, &env);

        let mut result = ExtractionResult::default();
        if let Some(best_assign) = ea.0.get(&root_assign) {
            for (u, value) in best_assign.assignment.iter() {
                if let Some(nid) = env.id_converter.vid_to_nid(u) {
                    if value == &AssignValue::KnownTrue || value == &AssignValue::SupposedTrue {
                        result.choose(egraph.nid_to_cid(nid).clone(), nid.clone());
                    }
                }
            }
        }

        println!("extract: {} us", start_time.elapsed().as_micros());

        result
    }
}
