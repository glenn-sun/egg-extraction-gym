use std::collections::BTreeMap;
use std::default;

use super::*;
use arboretum_td::graph::{BaseGraph, HashMapGraph, MutableGraph};
use arboretum_td::solver::Solver;
use arboretum_td::tree_decomposition::TreeDecomposition;

pub struct Treewidth2Extractor;

#[derive(Debug, Default)]
struct IdConverter {
    oid_to_cid: FxHashMap<usize, ClassId>,
    aid_to_nid: FxHashMap<usize, NodeId>,
    cid_to_oid: FxHashMap<ClassId, usize>,
    nid_to_aid: FxHashMap<NodeId, usize>,
    counter: usize,
}

impl IdConverter {
    fn get_oid_or_add_class(&mut self, cid: ClassId) -> usize {
        if let Some(oid) = self.cid_to_oid(&cid) {
            *oid
        } else {
            self.oid_to_cid.insert(self.counter, cid.clone());
            self.cid_to_oid.insert(cid, self.counter);
            self.counter += 1;
            self.counter - 1
        }
    }

    fn get_aid_or_add_node(&mut self, nid: NodeId) -> usize {
        if let Some(aid) = self.nid_to_aid(&nid) {
            *aid
        } else {
            self.aid_to_nid.insert(self.counter, nid.clone());
            self.nid_to_aid.insert(nid.clone(), self.counter);
            self.counter += 1;
            self.counter - 1
        }
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
    fn cid_to_oid(&self, cid: &ClassId) -> Option<&usize> {
        self.cid_to_oid.get(cid)
    }
    fn nid_to_aid(&self, nid: &NodeId) -> Option<&usize> {
        self.nid_to_aid.get(nid)
    }
}

#[derive(Debug)]
struct DirectedGraph {
    vertices: FxHashSet<usize>,
    inputs: FxHashMap<usize, FxHashSet<usize>>,
    outputs: FxHashMap<usize, FxHashSet<usize>>,
    root_id: usize,
}

impl DirectedGraph {
    fn new(root_id: usize) -> Self {
        DirectedGraph {
            vertices: FxHashSet::default(),
            inputs: FxHashMap::default(),
            outputs: FxHashMap::default(),
            root_id,
        }
    }

    fn add_vertex(&mut self, u: usize) {
        self.vertices.insert(u);
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

    fn get_vertices(&self) -> &FxHashSet<usize> {
        &self.vertices
    }
    fn inputs(&self, u: &usize) -> Option<&FxHashSet<usize>> {
        self.inputs.get(u)
    }
    fn outputs(&self, u: &usize) -> Option<&FxHashSet<usize>> {
        self.outputs.get(u)
    }

    fn to_graph(&self) -> HashMapGraph {
        let mut graph = HashMapGraph::new();
        for u in self.get_vertices() {
            for v in self.inputs(u).unwrap() {
                graph.add_edge(*u, *v);
            }
        }
        graph
    }
}

/// An environment to be passed around various functions.
struct Env<'a> {
    id_converter: IdConverter,
    digraph: DirectedGraph,
    nodes_in_class: FxHashMap<usize, FxHashSet<usize>>,
    classes_in_node: FxHashMap<usize, FxHashSet<usize>>,
    egraph: &'a EGraph,
}

impl<'a> Env<'a> {
    fn new(id_converter: IdConverter, digraph: DirectedGraph, nodes_in_class: FxHashMap<usize, FxHashSet<usize>>, classes_in_node: FxHashMap<usize, FxHashSet<usize>>, egraph: &'a EGraph) -> Self {
        Env {
            id_converter,
            digraph,
            nodes_in_class,
            classes_in_node,
            egraph,
        }
    }
}


#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
enum AssignValue {
    KnownTrue(usize),
    SupposedTrue(usize),
    False,
}

impl AssignValue {
    fn class_to_bool(&self) -> bool {
        match self {
            AssignValue::False => false,
            AssignValue::KnownTrue(_) => true,
            AssignValue::SupposedTrue(_) => true,
        }
    }

    fn to_aid(&self) -> Option<usize> {
        match self {
            AssignValue::False => None,
            AssignValue::KnownTrue(aid) => Some(*aid),
            AssignValue::SupposedTrue(aid) => Some(*aid)
        }
    }
}

#[derive(Hash, Default, Clone, Debug)]
struct Assignment {
    assignment: BTreeMap<usize, AssignValue>,
    // cost: Cost,
}

impl PartialEq for Assignment {
    fn eq(&self, other: &Self) -> bool {
        self.assignment == other.assignment
    }
    fn ne(&self, other: &Self) -> bool {
        self.assignment != other.assignment
    }
}

impl Eq for Assignment {}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Action {
    Insert,
    Forget,
}

impl Assignment {
    fn new(assignment: BTreeMap<usize, AssignValue>, env: &Env) -> Self {
        // let cost = Assignment::cost(&assignment, env);
        // Assignment { assignment, cost }
        Assignment {assignment}
    }

    fn get_true_aids(assignment: &BTreeMap<usize, AssignValue>) -> FxHashSet<usize> {
        assignment.iter().filter_map(|(u, value)| value.to_aid()).collect()
    }

    fn exists_cycle(assignment: &BTreeMap<usize, AssignValue>, env: &Env) -> bool {
        let mut exists_cycle = false;
        let mut current_path: FxHashSet<usize> = FxHashSet::default();
        let mut visited: FxHashSet<usize> = FxHashSet::default();

        let mut call_stack: Vec<(usize, Action)> = Vec::default();
        let true_oids: FxHashSet<usize> = assignment.iter().filter_map(|(u, value)| value.class_to_bool().then_some(*u)).collect();
        
        // println!("{:#?}", true_oids);

        for u in true_oids.iter() {
            call_stack.push((*u, Action::Forget));
            call_stack.push((*u, Action::Insert));
        }

        while let Some((u, a)) = call_stack.pop() {
            // println!("{:?}", call_stack);
            // println!("{:?}", current_path);
            // println!("{:?}", visited);


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
                    for v in env.digraph.outputs(&u).unwrap() {
                        // println!("{:?}", v);

                        if true_oids.contains(v) {
                            if current_path.contains(v) {
                                exists_cycle = true;
                                break;
                            }
                            if !visited.contains(v) {
                                call_stack.push((*v, Action::Forget));
                                call_stack.push((*v, Action::Insert));
                            }
                        }
                    }
                }
            }
        }

        exists_cycle
    }

    fn cost(assignment: &BTreeMap<usize, AssignValue>, env: &Env) -> Cost {
        // The current implementation uses DAG cost as the cost function.
        // If the assignment contains a cycle, set the cost to infinity.
        // This can be adapted to other cost functions, but they should be
        // defined on all subgraphs of e-graphs (including cyclic graphs and
        // disconnected graphs), monotone, and preserve order under common
        // unions.
        if Assignment::exists_cycle(&assignment, env) {
            return Cost::new(f64::INFINITY).unwrap();
        }

        let mut sum = Cost::default();
        for u in Assignment::get_true_aids(assignment) {
            if let Some(nid) = env.id_converter.aid_to_nid(&u) {
                sum += env.egraph.nodes.get(nid).unwrap().cost;
            }
        }
        sum
    }
}

#[derive(Default, Debug)]
struct ExtendAssigns(FxHashMap<Assignment, Assignment>);

impl ExtendAssigns {
    fn add_if_better(&mut self, env: &Env, bag_assign: Assignment, full_assign: Assignment) {
        if full_assign.assignment.get(&env.digraph.root_id) == Some(&AssignValue::False) {
            return;
        }

        let cost = Assignment::cost(&full_assign.assignment, env);
        if cost.is_finite() {
            if let Some(self_full_assign) = self.0.get(&bag_assign) {
                let self_cost = Assignment::cost(&self_full_assign.assignment, env);
                if self_cost > cost {
                    self.0.insert(bag_assign, full_assign);
                }
            } else {
                self.0.insert(bag_assign, full_assign);
            }
        }
    }
}

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

struct PossibleValues(FxHashSet<AssignValue>);

impl PossibleValues {
    fn default_from_oid(env: &Env, oid: usize) -> Self {
        let mut pv: FxHashSet<AssignValue> = FxHashSet::default();
        pv.insert(AssignValue::False);
        for aid in env.nodes_in_class.get(&oid).unwrap() {
            pv.insert(AssignValue::KnownTrue(*aid));
            pv.insert(AssignValue::SupposedTrue(*aid));
        }
        PossibleValues(pv)
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
        let outputs = env.digraph.outputs(&bag.x).unwrap();
        for out in outputs {
            if !bag.vertices.contains(&out) {
                continue;
            }
            match bag_assign.assignment.get(&out) {
                Some(AssignValue::False) => {}
                Some(AssignValue::KnownTrue(_)) => {}
                Some(AssignValue::SupposedTrue(aid)) => {
                    if env.classes_in_node.get(aid).unwrap().contains(&bag.x) {
                        self.0.remove(&AssignValue::False);
                    }
                }
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
        for aid in env.nodes_in_class.get(&bag.x).unwrap() {
            let input_values = FxHashSet::from_iter(env.classes_in_node.get(aid).unwrap().iter().map(|input| bag_assign.assignment.get(input)));
            if input_values.contains(&Some(&AssignValue::False)) {
                self.0.remove(&AssignValue::KnownTrue(*aid));
                self.0.remove(&AssignValue::SupposedTrue(*aid));
            } else if input_values.contains(&None) {
                self.0.remove(&AssignValue::KnownTrue(*aid));
            } else {
                self.0.remove(&AssignValue::SupposedTrue(*aid));
            }
        }
    }
    

    fn restrict(
        &mut self,
        env: &Env,
        bag: &Insert,
        bag_assign: &Assignment,
        full_assign: &Assignment,
    ) {
        self.restrict_by_outputs(env, bag, bag_assign, full_assign);
        self.restrict_by_inputs(env, bag, bag_assign, full_assign);
    }
}

impl NiceBag for Insert {
    // For an insert node, add all possible assignments to the new vertex, and upgrade
    // any SupposedTrue outputs 
    fn get_assignments<'a>(&self, env: &Env) -> ExtendAssigns {
        let mut ea = ExtendAssigns::default();
        let child_assignments = self.child.get_assignments(env).0;
        for (bag_assign, full_assign) in &child_assignments {
            let mut pv = PossibleValues::default_from_oid(env, self.x);
            pv.restrict(env, self, bag_assign, full_assign);

            for value in pv.0 {
                let mut new_bag_assign = bag_assign.assignment.clone();
                let mut new_full_assign = full_assign.assignment.clone();
                new_bag_assign.insert(self.x, value);
                new_full_assign.insert(self.x, value);

                for u in env.digraph.outputs(&self.x).unwrap() {
                    if let Some(&AssignValue::SupposedTrue(aid)) = bag_assign.assignment.get(u) {
                        if env.classes_in_node.get(&aid).unwrap().iter().all(|v| {
                            new_full_assign.get(v).map(|value| value.class_to_bool())
                                == Some(true)
                        }) {
                            new_bag_assign.insert(*u, AssignValue::KnownTrue(aid));
                            new_full_assign.insert(*u, AssignValue::KnownTrue(aid));
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
                        (Some(AssignValue::KnownTrue(aid1)), Some(AssignValue::KnownTrue(aid2))) => {
                            if aid1 != aid2 {
                                break;
                            }
                            new_bag_assign.insert(*u, AssignValue::KnownTrue(*aid1));
                        }
                        (Some(AssignValue::SupposedTrue(aid1)), Some(AssignValue::KnownTrue(aid2))) => {
                            if aid1 != aid2 {
                                break;
                            }
                            new_bag_assign.insert(*u, AssignValue::KnownTrue(*aid1));
                        }
                        (Some(AssignValue::KnownTrue(aid1)), Some(AssignValue::SupposedTrue(aid2))) => {
                            if aid1 != aid2 {
                                break;
                            }
                            new_bag_assign.insert(*u, AssignValue::KnownTrue(*aid1));
                        }
                        (Some(AssignValue::SupposedTrue(aid1)), Some(AssignValue::SupposedTrue(aid2))) => {
                            if aid1 != aid2 {
                                break;
                            }

                            if env.classes_in_node.get(aid1).unwrap().iter().all(|v| {
                                full_assign1.assignment.contains_key(v)
                                    || full_assign2.assignment.contains_key(v)
                            }) {
                                new_bag_assign.insert(*u, AssignValue::KnownTrue(*aid1));
                            } else {
                                new_bag_assign.insert(*u, AssignValue::SupposedTrue(*aid1));
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

fn print_stats(digraph: &DirectedGraph) {
    let num_verts = digraph.get_vertices().len();
    let num_edges = digraph
        .get_vertices()
        .iter()
        .map(|u| digraph.inputs(u).cloned().unwrap_or_default().len() as f32)
        .sum::<f32>()
        / num_verts as f32;
    println!("|V|, |E|/|V| = {}, {}", num_verts, num_edges);
}
struct SimplifyOptions {
    pub remove_unreachable: bool,
    pub contract_indegree_one: bool,
    pub verbose: bool,
}

impl Default for SimplifyOptions {
    fn default() -> Self {
        SimplifyOptions {
            remove_unreachable: true,
            contract_indegree_one: true,
            verbose: false,
        }
    }
}

impl<'a> Env<'a> {
    fn simplify(&mut self, options: SimplifyOptions) {
        if options.verbose {
            println!("before simplify");
            print_stats(&self.digraph);
        }
    
        let mut changed = true;
        while changed {
            changed = false;
            if options.remove_unreachable {
                let mut call_stack: Vec<usize> = Vec::default();
                call_stack.push(self.digraph.root_id);
                let mut visited: FxHashSet<usize> = FxHashSet::default();
    
                while let Some(u) = call_stack.pop() {
                    if visited.contains(&u) {
                        continue;
                    }
                    visited.insert(u);
                    if let Some(inputs) = self.digraph.inputs(&u) {
                        for v in inputs {
                            call_stack.push(*v);
                        }
                    }
                }
    
                let vertices = self.digraph.get_vertices().clone();
                for u in vertices {
                    if !visited.contains(&u) {
                        self.digraph.remove_vertex(u);
                        changed = true;
                    }
                }
    
                if options.verbose {
                    println!("after remove unreachable");
                    print_stats(&self.digraph);
                }
            }
    
            // if options.contract_indegree_one {
            //     let vertices = digraph.get_vertices().clone();
            //     for u in vertices {
            //         if u == digraph.root_id {
            //             continue;
            //         }
            //         let inputs = digraph.inputs(&u).cloned().unwrap_or_default();
            //         if inputs.len() == 1 {
            //             let v = inputs.into_iter().next().unwrap_or_default();
            //             digraph.contract_edge_remove_out(v, u);
            //             changed = true;
            //         }
            //     }
            //     if options.verbose {
            //         println!("after contract indegree one");
            //         print_stats(&digraph);
            //     }
            // }
        }
    }
}

impl Extractor for Treewidth2Extractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        let start_time = std::time::Instant::now();

        let mut id_converter = IdConverter::default();
        let mut nodes_in_class: FxHashMap<usize, FxHashSet<usize>> = FxHashMap::default();
        let mut classes_in_node: FxHashMap<usize, FxHashSet<usize>> = FxHashMap::default();

        let root_oid = id_converter.reserve_id();
        let root_aid = id_converter.reserve_id();

        let mut digraph = DirectedGraph::new(root_oid);
        digraph.add_vertex(root_oid);
        nodes_in_class.entry(root_oid).or_default().insert(root_aid);
        for cid in roots {
            let oid = id_converter.get_oid_or_add_class(cid.clone());
            classes_in_node.entry(root_aid).or_default().insert(oid);
            digraph.add_vertex(oid);
            digraph.add_edge(oid, root_oid);
        }

        for (cid, class) in egraph.classes() {
            let oid = id_converter.get_oid_or_add_class(cid.clone());
            digraph.add_vertex(oid);
            for nid in class.nodes.iter() {
                let aid = id_converter.get_aid_or_add_node(nid.clone());
                classes_in_node.entry(aid).or_default();
                nodes_in_class.entry(oid).or_default().insert(aid);

                if let Some(node) = egraph.nodes.get(nid) {
                    let mut self_loop = false;
                    for input_cid in node.children.iter().map(|input_nid| egraph.nid_to_cid(input_nid)) {
                        let input_oid = id_converter.get_oid_or_add_class(input_cid.clone());
                        if input_oid == oid {
                            self_loop = true
                        }
                    }

                    if self_loop {
                        nodes_in_class.entry(oid).or_default().remove(&aid);
                    } else {
                        for input_cid in node.children.iter().map(|input_nid| egraph.nid_to_cid(input_nid)) {
                            let input_oid = id_converter.get_oid_or_add_class(input_cid.clone());
                            classes_in_node.entry(aid).or_default().insert(input_oid);
                            digraph.add_vertex(input_oid);
                            digraph.add_edge(input_oid, oid);   
                        }
                    }
                }
            }
        }

        let options = SimplifyOptions {
            remove_unreachable: true,
            contract_indegree_one: true,
            verbose: true,
        };
        let mut env = Env::new(id_converter, digraph, nodes_in_class, classes_in_node, egraph);
        env.simplify(options);

        println!("preprocessing: {} us", start_time.elapsed().as_micros());
        let start_time = std::time::Instant::now();

        let graph = env.digraph.to_graph();
        let td = Solver::default_exact().solve(&graph);

        println!("td: {} us", start_time.elapsed().as_micros());
        let start_time = std::time::Instant::now();

        let mut root_bag_id: usize = 0;
        for bag in td.bags() {
            if bag.vertex_set.contains(&root_oid) {
                root_bag_id = bag.id;
                break;
            }
        }

        let nice_td = forget_until_root(to_nice_decomp(&td, &root_bag_id, None), &root_oid);

        println!("number of bags: {}", nice_td.n_bags());
        println!("average bag size: {}", nice_td.avg_bag_size());
        println!("max bag size: {}", td.max_bag_size);

        println!("nice td: {} us", start_time.elapsed().as_micros());

        let start_time = std::time::Instant::now();

        // Find the best satisfying assignment
        let ea = nice_td.get_assignments(&env);
        let mut root_assign_map: BTreeMap<usize, AssignValue> = BTreeMap::default();
        root_assign_map.insert(root_oid, AssignValue::KnownTrue(root_aid));
        let root_assign = Assignment::new(root_assign_map, &env);

        let mut result = ExtractionResult::default();
        if let Some(best_assign) = ea.0.get(&root_assign) {
            for (u, value) in best_assign.assignment.iter() {
                if let AssignValue::KnownTrue(aid) = value {
                    if let Some(nid) = env.id_converter.aid_to_nid(aid) {
                        result.choose(egraph.nid_to_cid(nid).clone(), nid.clone());
                    }
                } 
            }
        }

        println!("extract: {} us", start_time.elapsed().as_micros());

        result
    }
}