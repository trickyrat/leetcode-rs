#![cfg_attr(debug_assertions, allow(dead_code))]

use crate::leetcode::data_structures::ListNode;
use std::borrow::BorrowMut;

pub fn generate_linked_list_node(nums: Vec<i32>) -> Option<Box<ListNode>> {
    let mut head = ListNode::new(nums[0]);
    let dummy_head = head.borrow_mut();
    for &num in nums.iter() {
        dummy_head.append(num);
    }
    Some(Box::new(*head.next.unwrap()))
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
