#![cfg_attr(debug_assertions, allow(dead_code))]

use std::{cell::RefCell, collections::VecDeque, rc::Rc};

use crate::leetcode::data_structures::{ListNode, TreeNode};

pub fn generate_linked_list_node(nums: Vec<i32>) -> Option<Box<ListNode>> {
    if nums.is_empty() {
        return None;
    }

    let mut head = Box::new(ListNode::new(nums[0]));
    let mut dummy_head = &mut head;
    for &num in nums.iter().skip(1) {
        dummy_head.next = Some(Box::new(ListNode::new(num)));
        dummy_head = dummy_head.next.as_mut().unwrap();
    }
    Some(head)
}

pub fn generate_string_vec(strs: Vec<&str>) -> Vec<String> {
    strs.into_iter()
        .map(|s| String::from(s))
        .collect::<Vec<String>>()
}

pub fn generate_string_matrix(strs: Vec<Vec<&str>>) -> Vec<Vec<String>> {
    strs.into_iter()
        .map(|x| generate_string_vec(x.to_vec()))
        .collect::<Vec<Vec<String>>>()
}

fn build_binary_tree_from_level_order(data: String) -> Option<Rc<RefCell<TreeNode>>> {
    if data.is_empty() {
        return None;
    }

    let values: Vec<Option<i32>> = data
        .split(',')
        .map(|s| match s.trim() {
            "null" => None,
            num => num.parse::<i32>().ok(),
        })
        .collect();

    if values.is_empty() || values[0].is_none() {
        return None;
    }

    let root = Rc::new(RefCell::new(TreeNode::new(values[0].unwrap())));
    let mut queue: VecDeque<Rc<RefCell<TreeNode>>> = VecDeque::new();
    queue.push_back(root.clone());

    let mut index = 1;
    while !queue.is_empty() && index < values.len() {
        let node = queue.pop_front().unwrap();

        if index < values.len() {
            if let Some(val) = values[index] {
                let left = Rc::new(RefCell::new(TreeNode::new(val)));
                node.borrow_mut().left = Some(left.clone());
                queue.push_back(left);
            }
            index += 1;
        }

        if index < values.len() {
            if let Some(val) = values[index] {
                let right = Rc::new(RefCell::new(TreeNode::new(val)));
                node.borrow_mut().right = Some(right.clone());
                queue.push_back(right);
            }
            index += 1;
        }
    }

    Some(root)
}

pub struct BTSerializer {}

impl BTSerializer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn serialize(&self, root: Option<Rc<RefCell<TreeNode>>>) -> String {
        if root.is_none() {
            return "".to_string();
        }

        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(root);

        while !queue.is_empty() {
            let node = queue.pop_front().unwrap();
            if let Some(node) = node {
                result.push(node.borrow().val.to_string());
                queue.push_back(node.borrow().left.clone());
                queue.push_back(node.borrow().right.clone());
            } else {
                result.push("null".to_string());
            }
        }

        // Remove trailing nulls
        while result.last() == Some(&"null".to_string()) {
            result.pop();
        }

        format!("{}", result.join(","))
    }

    pub fn deserialize(&self, data: String) -> Option<Rc<RefCell<TreeNode>>> {
        if data.is_empty() {
            return None;
        }
    
        let values: Vec<Option<i32>> = data
            .split(',')
            .map(|s| match s.trim() {
                "null" => None,
                num => num.parse::<i32>().ok(),
            })
            .collect();
    
        if values.is_empty() || values[0].is_none() {
            return None;
        }
    
        let root = Rc::new(RefCell::new(TreeNode::new(values[0].unwrap())));
        let mut queue: VecDeque<Rc<RefCell<TreeNode>>> = VecDeque::new();
        queue.push_back(root.clone());
    
        let mut index = 1;
        while !queue.is_empty() && index < values.len() {
            let node = queue.pop_front().unwrap();
    
            if index < values.len() {
                if let Some(val) = values[index] {
                    let left = Rc::new(RefCell::new(TreeNode::new(val)));
                    node.borrow_mut().left = Some(left.clone());
                    queue.push_back(left);
                }
                index += 1;
            }
    
            if index < values.len() {
                if let Some(val) = values[index] {
                    let right = Rc::new(RefCell::new(TreeNode::new(val)));
                    node.borrow_mut().right = Some(right.clone());
                    queue.push_back(right);
                }
                index += 1;
            }
        }
    
        Some(root)
    }
}

#[cfg(test)]
mod unittests {
    use crate::leetcode::utils::*;

    fn setup_serialize_testcases(input: &str) {
        let coder = BTSerializer::new();
        let data = input.to_string();
        let root = build_binary_tree_from_level_order(data.clone());
        let result = coder.serialize(root);
        assert_eq!(result, data);
    }

    fn setup_deserialize_testcases(input: &str) {
        let coder = BTSerializer::new();
        let data = input.to_string();
        let root = coder.deserialize(data.clone());
        let result = coder.serialize(root);
        assert_eq!(result, data);
    }

    #[test]
    fn test_serialize() {
        setup_serialize_testcases("1,2,3,null,null,4,5");
        setup_serialize_testcases("1,2,3,null,5,null,4");
    }

    #[test]
    fn test_deserilize_btcoder() {
        setup_deserialize_testcases("1,2,3,null,null,4,5");
        setup_deserialize_testcases("1,2,3,null,5,null,4");
    }
}
