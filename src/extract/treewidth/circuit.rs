use super::*;

/// A basic implementation of a monotone circuit. The circuit
/// may have cycles and may not be connected. Used as
/// an intermediate representation between e-graphs and
/// tree decomposition.
#[derive(Debug)]
pub struct Circuit {
    vertices: IntSet<usize>,
    inputs: IntMap<usize, IntSet<usize>>,
    outputs: IntMap<usize, IntSet<usize>>,
    gate_type: IntMap<usize, Gate>,
    pub root_id: usize,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum Gate {
    Variable,
    Or,
    And,
}

impl Circuit {
    pub fn with_root(root_id: usize) -> Self {
        Circuit {
            vertices: IntSet::default(),
            inputs: IntMap::default(),
            outputs: IntMap::default(),
            gate_type: IntMap::default(),
            root_id,
        }
    }

    pub fn add_vertex(&mut self, u: usize, gate_type: Gate) {
        self.vertices.insert(u);
        self.gate_type.insert(u, gate_type);
        self.inputs.entry(u).or_default();
        self.outputs.entry(u).or_default();
    }

    pub fn add_edge(&mut self, u: usize, v: usize) {
        self.inputs.entry(v).or_default().insert(u);
        self.outputs.entry(u).or_default().insert(v);
    }

    pub fn remove_vertex(&mut self, u: usize) {
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

    pub fn remove_edge(&mut self, u: usize, v: usize) {
        self.inputs.entry(v).or_default().remove(&u);
        self.outputs.entry(u).or_default().remove(&v);
    }

    pub fn contract_edge_remove_out(&mut self, u: usize, v: usize) {
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

    pub fn get_vertices(&self) -> &IntSet<usize> {
        &self.vertices
    }

    pub fn inputs(&self, u: &usize) -> Option<&IntSet<usize>> {
        self.inputs.get(u)
    }
    pub fn inputs_or_unreachable(&self, u: &usize) -> &IntSet<usize> {
        self.inputs.get(u).unwrap_or_else(|| unreachable!())
    }

    pub fn outputs(&self, u: &usize) -> Option<&IntSet<usize>> {
        self.outputs.get(u)
    }
    pub fn outputs_or_unreachable(&self, u: &usize) -> &IntSet<usize> {
        self.outputs.get(u).unwrap_or_else(|| unreachable!())
    }

    pub fn gate_type(&self, u: &usize) -> Option<&Gate> {
        self.gate_type.get(u)
    }
    pub fn gate_type_or_unreachable(&self, u: &usize) -> &Gate {
        self.gate_type.get(u).unwrap_or_else(|| unreachable!())
    }

    /// Convert to an undirected graph compatiable with arboretum_td
    pub fn to_graph(&self) -> HashMapGraph {
        let mut graph = HashMapGraph::new();
        for u in self.get_vertices() {
            for v in self.inputs_or_unreachable(u) {
                graph.add_edge(*u, *v);
            }
        }
        graph
    }

    pub fn print_stats(&self) {
        let num_verts = self.get_vertices().len();
        let num_edges = self
            .get_vertices()
            .iter()
            .map(|u| self.inputs_or_unreachable(u).len() as f32)
            .sum::<f32>() as f32;
        println!("|V|, |E| = {}, {}", num_verts, num_edges);
    }
}
