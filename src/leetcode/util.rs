use crate::leetcode::data_structures::{LinkedList, ListNode};
use std::borrow::BorrowMut;

#[allow(dead_code)]
pub fn generate_linked_list_node(nums: Vec<i32>) -> Option<Box<ListNode>> {
    let mut head = ListNode::new(nums[0]);
    let dummy_head = head.borrow_mut();
    for &num in nums.iter() {
        dummy_head.append(num);
    }
    Some(Box::new(*head.next.unwrap()))
}

#[allow(dead_code)]
pub fn generate_string_vec(strs: Vec<&str>) -> Vec<String> {
    strs.iter()
        .map(|&x| String::from(x))
        .collect::<Vec<String>>()
}

#[allow(dead_code)]
pub fn generate_string_matrix(strs: Vec<Vec<&str>>) -> Vec<Vec<String>> {
    strs.iter()
        .map(|x| generate_string_vec(x.to_vec()))
        .collect::<Vec<Vec<String>>>()
}
