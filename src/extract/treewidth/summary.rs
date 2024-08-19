use super::*;

/// A "summary" of an evaluation. It contains all the information
/// needed to determine whether or not we need to keep a record
/// of the evaluation moving forward.
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct Summary {
    pub bag_eval: Evaluation,
    pub is_known: KnownTag,
    pub transitive_graph: TransitiveGraph,
}

impl Summary {
    pub fn new(bag_eval: Evaluation, is_known: KnownTag, transitive_graph: TransitiveGraph) -> Self {
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
pub struct Evaluation {
    pub map: IntMap<usize, bool>,
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

impl Evaluation {
    pub fn get_or_unreachable(&self, u: &usize) -> &bool {
        self.map.get(u).unwrap_or_else(|| unreachable!())
    }

    /// Check if there exists a way to pick the inputs to u
    /// outside the evaluation's domain such that together,
    /// u meets the requirements of its gate type.
    ///  
    /// Panics if u is not in the domain.
    pub fn is_valid_at(&self, u: &usize, env: &Env) -> bool {
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
    pub fn is_valid_around(&self, u: &usize, env: &Env) -> bool {
        self.is_valid_at(u, env)
            && env
                .circuit
                .outputs_or_unreachable(u)
                .iter()
                .all(|v| !self.map.contains_key(v) || self.is_valid_at(v, env))
    }

    pub fn insert_unchecked(&self, u: usize, b: bool) -> Evaluation {
        let mut new_eval = self.clone();
        new_eval.map.insert(u, b);
        new_eval
    }

    pub fn remove(&self, u: &usize) -> Evaluation {
        let mut new_eval = self.clone();
        new_eval.map.remove(u);
        new_eval
    }

    pub fn merge(&self, other: &Evaluation) -> Evaluation {
        let mut new_eval = self.clone();
        new_eval.map.reserve(other.map.len());
        new_eval.map.extend(other.map.clone());
        new_eval
    }

    pub fn cost(&self, env: &Env) -> Cost {
        self.map
            .iter()
            .filter(|(u, b)| {
                b == &&true && env.circuit.gate_type_or_unreachable(*u) == &Gate::Variable
            })
            .map(|(u, _)| env.cost.get(u).unwrap_or_else(|| unreachable!()))
            .sum()
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
pub struct KnownTag {
    pub map: IntMap<usize, bool>,
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
    pub fn fix_tag_at(&mut self, u: &usize, full_eval: &Evaluation, env: &Env) {
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
    pub fn fix_tags_around(
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
    pub fn compute_all_tags(vertices: &IntSet<usize>, full_eval: &Evaluation, env: &Env) -> KnownTag {
        let mut new_kt = KnownTag::default();
        for u in vertices {
            if full_eval.map.get(u) == Some(&true) {
                new_kt.fix_tag_at(u, full_eval, env);
            }
        }
        new_kt
    }

    pub fn remove(&self, u: &usize) -> KnownTag {
        let mut new_kt = self.clone();
        new_kt.map.remove(u);
        new_kt
    }
}

/// A transitive graph is a graph where a path u -> v implies an
/// edge u -> v.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct TransitiveGraph {
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
    pub fn has_cycle(&self) -> bool {
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
    pub fn insert_vertex(&self, u: &usize, env: &Env) -> TransitiveGraph {
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

    pub fn remove(&self, u: &usize) -> TransitiveGraph {
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
    pub fn merge(&self, other: &TransitiveGraph) -> TransitiveGraph {
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
}