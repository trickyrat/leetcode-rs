#![cfg_attr(debug_assertions, allow(dead_code))]

use std::cmp::Ordering;
use std::mem;

#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl Ord for ListNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.val.cmp(&self.val)
    }
}

impl PartialOrd for ListNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl ListNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }

    pub fn get_last_node(&mut self) -> &mut Self {
        if let Some(ref mut box_node) = self.next {
            box_node.get_last_node()
        } else {
            self
        }
    }

    pub fn append(&mut self, val: i32) {
        let new_node = ListNode::new(val);
        self.get_last_node().next = Some(Box::new(new_node));
    }
}

enum ListNodeType {
    Empty,
    NonEmpty(ListNode),
}

impl ListNodeType {
    fn new(val: i32, next: Option<Box<ListNode>>) -> Self {
        ListNodeType::NonEmpty(ListNode { val, next })
    }

    fn take(&mut self) -> Self {
        let mut curr_node = Self::Empty;
        mem::swap(&mut curr_node, self);
        curr_node
    }
}

pub struct LinkedList {
    length: usize,
    pub head: Option<Box<ListNode>>,
    pub tail: Option<Box<ListNode>>,
}

impl LinkedList {
    pub fn new() -> Self {
        Self {
            length: 0,
            head: None,
            tail: None,
        }
    }
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn insert_at_head(&mut self, val: i32) {
        let curr_head = self.head.take();
        let new_node = Some(Box::new(ListNode {
            val,
            next: curr_head,
        }));
        self.head = new_node;
        self.length += 1;
    }

    pub fn insert_at_tail(&mut self, val: i32) {
        let curr_head = self.tail.take();
        let new_node = Some(Box::new(ListNode {
            val,
            next: curr_head,
        }));
        self.tail = new_node;
        self.length += 1;
    }
}

#[cfg(test)]
mod tests {
    use crate::leetcode::data_structures::ListNode;

    #[test]
    fn test_list_node_construct() {
        let head = Some(ListNode::new(1));
        assert!(head.is_some());
        assert_eq!(1, head.unwrap().val);
    }
}
