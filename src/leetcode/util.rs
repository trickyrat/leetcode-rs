use super::ListNode;

pub fn generate_list_node(nums: Vec<i32>) -> Option<Box<ListNode>> {
    let mut dummy_head1 = ListNode::new(nums[0]);
    for i in 1..nums.len() {
        dummy_head1.append(nums[i]);
    }
    Some(Box::new(dummy_head1))
}
