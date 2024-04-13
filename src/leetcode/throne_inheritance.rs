#![cfg_attr(debug_assertions, allow(unused))]

use std::collections::{HashMap, HashSet};

pub struct ThroneInheritance {
    graph: HashMap<String, Vec<String>>,
    dead: HashSet<String>,
    king: String,
}

impl ThroneInheritance {
    fn new(king_name: String) -> Self {
        Self {
            graph: HashMap::new(),
            dead: HashSet::new(),
            king: king_name,
        }
    }

    fn birth(&mut self, parent_name: String, child_name: String) {
        self.graph.entry(parent_name).or_default().push(child_name);
    }

    fn death(&mut self, name: String) {
        self.dead.insert(name);
    }
    fn get_inheritance_order(&self) -> Vec<String> {
        let mut result = vec![];
        let mut stack = vec![self.king.clone()];
        while !stack.is_empty() {
            let name = stack.pop().unwrap();
            if !self.dead.contains(&name) {
                result.push(name.clone());
            }
            if let Some(children) = self.graph.get(&name) {
                stack.extend(children.iter().rev().cloned());
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::leetcode::throne_inheritance::ThroneInheritance;

    #[test]
    fn test () {
        let mut obj = ThroneInheritance::new("king".to_string());
        obj.birth("king".to_string(), "andy".to_string());
        obj.birth("king".to_string(), "bob".to_string());
        obj.birth("king".to_string(), "catherine".to_string());
        obj.birth("andy".to_string(), "matthew".to_string());
        obj.birth("bob".to_string(), "alex".to_string());
        obj.birth("bob".to_string(), "asha".to_string());
        assert_eq!(obj.get_inheritance_order(), vec!["king", "andy", "matthew", "bob", "alex", "asha", "catherine"]);
        obj.death("bob".to_string());
        assert_eq!(obj.get_inheritance_order(), vec!["king", "andy", "matthew", "alex", "asha", "catherine"]);
    }
}