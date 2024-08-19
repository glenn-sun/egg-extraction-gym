use super::*;

/// Convert a tree decomposition to a nice tree decomposition.
pub fn to_nice_decomp(
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
pub fn forget_until_root(nice_td: Box<NiceBag>, root_id: &usize) -> Box<NiceBag> {
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