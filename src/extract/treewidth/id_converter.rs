use super::*;

/// Converts between usize IDs and ClassIds/NodeIds.
/// IDs for use with arboretum_td are required to be usize.
/// For convenience, this maintains maps in both directions
/// and segregates IDs for Variables, And gates, and Or gates.
/// One variable may correspond to a set of NodeIds to be
/// selected simultaneously.
#[derive(Debug, Default)]
pub struct IdConverter {
    vid_to_nid_set: IntMap<usize, FxHashSet<NodeId>>,
    cid_to_oid: FxHashMap<ClassId, usize>,
    nid_to_aid: FxHashMap<NodeId, usize>,
    nid_to_vid: FxHashMap<NodeId, usize>,
    counter: usize,
}

impl IdConverter {
    pub fn get_oid_or_add_class(&mut self, cid: &ClassId) -> usize {
        if let Some(oid) = self.cid_to_oid.get(cid) {
            *oid
        } else {
            self.cid_to_oid.insert(cid.clone(), self.counter);
            self.counter += 1;
            self.counter - 1
        }
    }

    pub fn get_aid_or_add_node(&mut self, nid: &NodeId) -> usize {
        if let Some(aid) = self.nid_to_aid.get(&nid) {
            *aid
        } else {
            self.nid_to_aid.insert(nid.clone(), self.counter);
            self.counter += 1;
            self.counter - 1
        }
    }

    pub fn get_vid_or_add_node(&mut self, nid: &NodeId) -> usize {
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

    pub fn merge_vid_keep1(&mut self, vid1: usize, vid2: usize) {
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
    pub fn reserve_id(&mut self) -> usize {
        self.counter += 1;
        self.counter - 1
    }

    pub fn vid_to_nid_set(&self, vid: &usize) -> Option<&FxHashSet<NodeId>> {
        self.vid_to_nid_set.get(vid)
    }
}
