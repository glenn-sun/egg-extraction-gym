use std::collections::BTreeMap;

use super::*;
use arboretum_td::graph::{HashMapGraph, MutableGraph};
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
}

impl Circuit {
    fn new() -> Self {
        Circuit {
            vertices: FxHashSet::default(),
            inputs: FxHashMap::default(),
            outputs: FxHashMap::default(),
            gate_type: FxHashMap::default(),
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

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
enum AssignValue {
    KnownTrue,
    SupposedTrue,
    False,
}

/// A map that represents a valid partial evaluation of a circuit.
/// Uses BTreeMap so that it can be hashed.
#[derive(PartialEq, Eq, Hash, Default, Clone, Debug)]
struct Assignment(BTreeMap<usize, AssignValue>);

impl Assignment {
    fn cost(&self, env: &Env) -> Cost {
        // Detect cycles using DFS
        let true_vertices = FxHashSet::from_iter(self.0.iter().filter_map(|(u, x)| {
            if x == &AssignValue::KnownTrue || x == &AssignValue::SupposedTrue {
                Some(u)
            } else {
                None
            }
        }));

        let mut exists_cycle = false;
        let mut current_path: FxHashSet<&usize> = FxHashSet::default();
        let mut visited: FxHashSet<&usize> = FxHashSet::default();

        #[derive(PartialEq, Eq, Clone, Copy, Debug)]
        enum Action {
            Insert,
            Forget,
        }

        let mut call_stack: Vec<(&usize, Action)> = Vec::default();
        for u in true_vertices.iter() {
            call_stack.push((u, Action::Forget));
            call_stack.push((u, Action::Insert));
        }

        while let Some((u, a)) = call_stack.pop() {
            match a {
                Action::Forget => {
                    current_path.remove(u);
                }
                Action::Insert => {
                    if visited.contains(u) {
                        continue;
                    }
                    current_path.insert(u);
                    visited.insert(u);
                    if let Some(vs) = env.circuit.outputs(u) {
                        for v in vs {
                            if true_vertices.contains(v) {
                                if current_path.contains(v) {
                                    exists_cycle = true;
                                    break;
                                }
                                if !visited.contains(v) {
                                    call_stack.push((v, Action::Forget));
                                    call_stack.push((v, Action::Insert));
                                }
                            }
                        }
                    }
                }
            }
        }

        // The current implementation uses DAG cost as the cost function.
        // If the assignment contains a cycle, set the cost to infinity.
        // This can be adapted to other cost functions, but they should be
        // defined on all subgraphs of e-graphs (including cyclic graphs and
        // disconnected graphs), monotone, and preserve order under common
        // unions.
        if exists_cycle {
            return Cost::new(f64::INFINITY).unwrap();
        }

        let mut sum = Cost::default();
        for u in true_vertices {
            if env
                .circuit
                .gate_type(u)
                .is_some_and(|gate| gate == &Gate::Variable)
            {
                if let Some(nid) = env.id_converter.vid_to_nid(u) {
                    if let Some(node) = env.egraph.nodes.get(nid) {
                        sum += node.cost;
                    }
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
        let cost = full_assign.cost(env);
        if cost.is_finite() {
            if let Some(self_full_assign) = self.0.get(&bag_assign) {
                if self_full_assign.cost(env) > cost {
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
    fn current_size(&self) -> usize;
}

#[derive(Debug)]
struct Leaf {
    vertices: FxHashSet<usize>,
    current_size: usize,
}

impl Leaf {
    fn new() -> Self {
        Leaf {
            vertices: FxHashSet::default(),
            current_size: 1,
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

    fn current_size(&self) -> usize {
        self.current_size
    }
}

#[derive(Debug)]
struct Insert {
    vertices: FxHashSet<usize>,
    child: Box<dyn NiceBag>,
    x: usize,
    current_size: usize,
}

impl Insert {
    fn new(vertices: FxHashSet<usize>, child: Box<dyn NiceBag>, x: usize) -> Self {
        let size = child.current_size();
        Insert {
            vertices,
            child,
            x,
            current_size: size + 1,
        }
    }
}

impl NiceBag for Insert {
    fn get_assignments<'a>(&self, env: &Env) -> ExtendAssigns {
        let mut ea = ExtendAssigns::default();
        let child_assignments = self.child.get_assignments(env).0;
        for (bag_assign, full_assign) in &child_assignments {
            let restrict_by_outputs =
                |allowed_values: FxHashSet<&'a AssignValue>| -> FxHashSet<&'a AssignValue> {
                    let mut new_allowed_values = allowed_values.clone();
                    if let Some(outputs) = env.circuit.outputs(&self.x) {
                        for out in outputs {
                            if !self.vertices.contains(out) {
                                continue;
                            }
                            if let Some(gate) = env.circuit.gate_type(out) {
                                match gate {
                                    Gate::Or => {
                                        if let Some(value) = bag_assign.0.get(out) {
                                            match value {
                                                AssignValue::False => {
                                                    new_allowed_values
                                                        .remove(&AssignValue::KnownTrue);
                                                    new_allowed_values
                                                        .remove(&AssignValue::SupposedTrue);
                                                }
                                                AssignValue::KnownTrue => {}
                                                AssignValue::SupposedTrue => {
                                                    if let Some(siblings) = env.circuit.inputs(out)
                                                    {
                                                        let mut set_gates: FxHashSet<usize> =
                                                            full_assign.0.keys().cloned().collect();
                                                        set_gates.insert(self.x);
                                                        if siblings.is_subset(&set_gates) {
                                                            new_allowed_values
                                                                .remove(&AssignValue::False);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Gate::And => {
                                        if let Some(value) = bag_assign.0.get(out) {
                                            match value {
                                                AssignValue::False => {}
                                                AssignValue::KnownTrue => {
                                                    // case not possible
                                                }
                                                AssignValue::SupposedTrue => {
                                                    new_allowed_values.remove(&AssignValue::False);
                                                }
                                            }
                                        }
                                    }
                                    Gate::Variable => {}
                                }
                            }
                        }
                    };
                    new_allowed_values
                };
            let mut fix_outputs_and_add = |allowed_values: FxHashSet<&AssignValue>| {
                for value in allowed_values {
                    let mut new_bag_assign = bag_assign.clone();
                    let mut new_full_assign = full_assign.clone();
                    new_bag_assign.0.insert(self.x, *value);
                    new_full_assign.0.insert(self.x, *value);

                    if let Some(parents) = env.circuit.outputs(&self.x) {
                        for p in parents {
                            if self.vertices.contains(p)
                                && bag_assign
                                    .0
                                    .get(p)
                                    .is_some_and(|value| value == &AssignValue::SupposedTrue)
                            {
                                if env
                                    .circuit
                                    .gate_type(p)
                                    .is_some_and(|gate| gate == &Gate::Or)
                                    && (value == &AssignValue::KnownTrue
                                        || value == &AssignValue::SupposedTrue)
                                {
                                    new_bag_assign.0.insert(*p, AssignValue::KnownTrue);
                                    new_full_assign.0.insert(*p, AssignValue::KnownTrue);
                                }
                                if env
                                    .circuit
                                    .gate_type(p)
                                    .is_some_and(|gate| gate == &Gate::And)
                                {
                                    let all_children_true =
                                        env.circuit.inputs(p).map(|children| -> bool {
                                            let mut output = true;
                                            for child in children {
                                                if !new_full_assign.0.contains_key(child)
                                                    || new_full_assign.0.get(child).is_some_and(
                                                        |value| value == &AssignValue::False,
                                                    )
                                                {
                                                    output = false;
                                                    break;
                                                }
                                            }
                                            output
                                        });
                                    if let Some(true) = all_children_true {
                                        new_bag_assign.0.insert(*p, AssignValue::KnownTrue);
                                        new_full_assign.0.insert(*p, AssignValue::KnownTrue);
                                    }
                                }
                            }
                        }
                    };

                    ea.add_if_better(env, new_bag_assign, new_full_assign);
                }
            };

            if let Some(gate) = env.circuit.gate_type(&self.x) {
                match gate {
                    Gate::Variable => {
                        let mut allowed_values: FxHashSet<&AssignValue> = FxHashSet::default();
                        allowed_values.insert(&AssignValue::False);
                        allowed_values.insert(&AssignValue::KnownTrue);
                        allowed_values = restrict_by_outputs(allowed_values);
                        fix_outputs_and_add(allowed_values);
                    }
                    Gate::Or => {
                        let mut allowed_values: FxHashSet<&AssignValue> = FxHashSet::default();
                        allowed_values.insert(&AssignValue::False);
                        allowed_values.insert(&AssignValue::KnownTrue);
                        allowed_values.insert(&AssignValue::SupposedTrue);
                        allowed_values = restrict_by_outputs(allowed_values);

                        if let Some(inputs) = env.circuit.inputs(&self.x) {
                            let input_colors = FxHashSet::from_iter(
                                inputs.iter().map(|input| bag_assign.0.get(input)),
                            );
                            if input_colors.contains(&Some(&AssignValue::KnownTrue))
                                || input_colors.contains(&Some(&AssignValue::SupposedTrue))
                            {
                                allowed_values.remove(&AssignValue::False);
                                allowed_values.remove(&AssignValue::SupposedTrue);
                            } else if input_colors.contains(&None) {
                                allowed_values.remove(&AssignValue::KnownTrue);
                            } else {
                                allowed_values.remove(&AssignValue::KnownTrue);
                                allowed_values.remove(&AssignValue::SupposedTrue);
                            }
                        };
                        fix_outputs_and_add(allowed_values);
                    }
                    Gate::And => {
                        let mut allowed_values: FxHashSet<&AssignValue> = FxHashSet::default();
                        allowed_values.insert(&AssignValue::False);
                        allowed_values.insert(&AssignValue::KnownTrue);
                        allowed_values.insert(&AssignValue::SupposedTrue);
                        allowed_values = restrict_by_outputs(allowed_values);

                        if let Some(inputs) = env.circuit.inputs(&self.x) {
                            let input_colors = FxHashSet::from_iter(
                                inputs.iter().map(|input| bag_assign.0.get(input)),
                            );
                            if input_colors.contains(&Some(&AssignValue::False)) {
                                allowed_values.remove(&AssignValue::KnownTrue);
                                allowed_values.remove(&AssignValue::SupposedTrue);
                            } else if input_colors.contains(&None) {
                                allowed_values.remove(&AssignValue::KnownTrue);
                            } else {
                                allowed_values.remove(&AssignValue::False);
                                allowed_values.remove(&AssignValue::SupposedTrue);
                            }
                        };
                        fix_outputs_and_add(allowed_values);
                    }
                }
            }
        }
        ea
    }

    fn vertices(&self) -> &FxHashSet<usize> {
        &self.vertices
    }

    fn current_size(&self) -> usize {
        self.current_size
    }
}

#[derive(Debug)]
struct Forget {
    vertices: FxHashSet<usize>,
    child: Box<dyn NiceBag>,
    x: usize,
    current_size: usize,
}

impl Forget {
    fn new(vertices: FxHashSet<usize>, child: Box<dyn NiceBag>, x: usize) -> Self {
        let size = child.current_size();
        Forget {
            vertices,
            child,
            x,
            current_size: size + 1,
        }
    }
}

impl NiceBag for Forget {
    fn get_assignments(&self, env: &Env) -> ExtendAssigns {
        let mut ea = ExtendAssigns::default();
        let child_assignments = self.child.get_assignments(env).0;
        for (bag_assign, full_assign) in &child_assignments {
            let mut new_bag_assign = bag_assign.clone();
            new_bag_assign.0.remove(&self.x);
            ea.add_if_better(env, new_bag_assign, full_assign.clone());
        }
        ea
    }

    fn vertices(&self) -> &FxHashSet<usize> {
        &self.vertices
    }

    fn current_size(&self) -> usize {
        self.current_size
    }
}

#[derive(Debug)]
struct Join {
    vertices: FxHashSet<usize>,
    child1: Box<dyn NiceBag>,
    child2: Box<dyn NiceBag>,
    current_size: usize,
}

impl Join {
    fn new(vertices: FxHashSet<usize>, child1: Box<dyn NiceBag>, child2: Box<dyn NiceBag>) -> Self {
        let size1 = child1.current_size();
        let size2 = child2.current_size();
        Join {
            vertices,
            child1,
            child2,
            current_size: size1 + size2 + 1,
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
                let mut new_bag_assign = Assignment::default();
                let mut valid = true;
                for u in self.vertices.iter() {
                    match (bag_assign1.0.get(u), bag_assign2.0.get(u)) {
                        (Some(AssignValue::False), Some(AssignValue::False)) => {
                            new_bag_assign.0.insert(*u, AssignValue::False);
                        }
                        (Some(AssignValue::KnownTrue), Some(AssignValue::KnownTrue)) => {
                            new_bag_assign.0.insert(*u, AssignValue::KnownTrue);
                        }
                        (Some(AssignValue::SupposedTrue), Some(AssignValue::KnownTrue)) => {
                            new_bag_assign.0.insert(*u, AssignValue::KnownTrue);
                        }
                        (Some(AssignValue::KnownTrue), Some(AssignValue::SupposedTrue)) => {
                            new_bag_assign.0.insert(*u, AssignValue::KnownTrue);
                        }
                        (Some(AssignValue::SupposedTrue), Some(AssignValue::SupposedTrue)) => {
                            if let Some(gate) = env.circuit.gate_type(u) {
                                match gate {
                                    Gate::Variable => {}
                                    Gate::And => {
                                        if let Some(inputs) = env.circuit.inputs(u) {
                                            let mut all_inputs_true = true;
                                            for input in inputs.iter() {
                                                if !full_assign1.0.contains_key(input)
                                                    && !full_assign2.0.contains_key(input)
                                                {
                                                    all_inputs_true = false;
                                                    break;
                                                }
                                            }
                                            if all_inputs_true {
                                                new_bag_assign.0.insert(*u, AssignValue::KnownTrue);
                                            } else {
                                                new_bag_assign
                                                    .0
                                                    .insert(*u, AssignValue::SupposedTrue);
                                            }
                                        }
                                    }
                                    Gate::Or => {
                                        if let Some(inputs) = env.circuit.inputs(u) {
                                            let mut all_inputs_true = true;
                                            for input in inputs.iter() {
                                                if !full_assign1.0.contains_key(input)
                                                    && !full_assign2.0.contains_key(input)
                                                {
                                                    all_inputs_true = false;
                                                    break;
                                                }
                                            }
                                            if all_inputs_true {
                                                valid = false;
                                            } else {
                                                new_bag_assign
                                                    .0
                                                    .insert(*u, AssignValue::SupposedTrue);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        _ => {
                            valid = false;
                        }
                    }
                }
                if valid {
                    let mut new_full_assign = new_bag_assign.clone();
                    for u in full_assign1.0.keys() {
                        if !self.vertices.contains(u) {
                            full_assign1
                                .0
                                .get(u)
                                .map(|value| new_full_assign.0.insert(*u, *value));
                        }
                    }
                    for u in full_assign2.0.keys() {
                        if !self.vertices.contains(u) {
                            full_assign2
                                .0
                                .get(u)
                                .map(|value| new_full_assign.0.insert(*u, *value));
                        }
                    }
                    ea.add_if_better(env, new_bag_assign, new_full_assign);
                }
            }
        }
        ea
    }

    fn vertices(&self) -> &FxHashSet<usize> {
        &self.vertices
    }

    fn current_size(&self) -> usize {
        self.current_size
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

impl Extractor for TreewidthExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
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
        let mut circuit = Circuit::new();
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
        let root_id = id_converter.reserve_id();
        circuit.add_vertex(root_id, Gate::And);
        for root in roots {
            if let Some(v) = id_converter.cid_to_oid(root) {
                circuit.add_edge(*v, root_id);
            }
        }

        // Build underlying undirected graph from circuit
        let mut graph = HashMapGraph::new();
        for u in circuit.get_vertices() {
            if let Some(vs) = circuit.inputs(u) {
                for v in vs {
                    graph.add_edge(*u, *v);
                }
            }
        }

        // Run tree decomposition and identify a bag with the root
        let td = Solver::default_exact().solve(&graph);

        let mut root_bag_id: usize = 0;
        for bag in td.bags() {
            if bag.vertex_set.contains(&root_id) {
                root_bag_id = bag.id;
                break;
            }
        }

        let nice_td = forget_until_root(to_nice_decomp(&td, &root_bag_id, None), &root_id);
        let env = Env::new(id_converter, circuit, egraph);
        let ea = nice_td.get_assignments(&env);
        let mut root_assign = Assignment::default();
        root_assign.0.insert(root_id, AssignValue::KnownTrue);

        let mut result = ExtractionResult::default();
        if let Some(best_assign) = ea.0.get(&root_assign) {
            for (u, value) in best_assign.0.iter() {
                if let Some(nid) = env.id_converter.vid_to_nid(u) {
                    if value == &AssignValue::KnownTrue || value == &AssignValue::SupposedTrue {
                        result.choose(egraph.nid_to_cid(nid).clone(), nid.clone());
                    }
                }
            }
        }

        result
    }
}
