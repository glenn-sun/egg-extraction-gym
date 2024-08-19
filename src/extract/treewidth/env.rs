use super::*;

/// A variety of circuit simplification rules. See write-up for details.
pub struct SimplifyOptions {
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
pub struct Env<'a> {
    pub id_converter: IdConverter,
    pub circuit: Circuit,
    pub egraph: &'a EGraph,
    pub max_cost: Cost,
    pub cost: IntMap<usize, Cost>,
}

impl<'a> Env<'a> {
    pub fn new(
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
    pub fn save_cosmograph(&self, filename_noext: String) -> Result<(), Box<dyn Error>> {
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
    pub fn compute_costs(&mut self) {
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
    pub fn simplify(&mut self, options: SimplifyOptions) {
        if options.verbose {
            println!("before simplify");
            self.circuit.print_stats();
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
                    self.circuit.print_stats();
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
                    self.circuit.print_stats();
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
                    self.circuit.print_stats();
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
                    self.circuit.print_stats();
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
                    self.circuit.print_stats();
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
                    self.circuit.print_stats();
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
                    self.circuit.print_stats();
                }
            }
        }
    }
}
