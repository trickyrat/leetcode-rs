use super::ListNode;

pub fn generate_list_node(nums: Vec<i32>) -> Option<Box<ListNode>> {
    let mut dummy_head1 = ListNode::new(nums[0]);
    for i in 1..nums.len() {
        dummy_head1.append(nums[i]);
    }
    Some(Box::new(dummy_head1))
}

pub fn generate_string_vec(strs: Vec<&str>) -> Vec<String> {
    strs.iter()
        .map(|&x| String::from(x))
        .collect::<Vec<String>>()
}

pub fn generate_string_matrix(strs: Vec<Vec<&str>>) -> Vec<Vec<String>> {
    strs.iter()
        .map(|x| generate_string_vec(x.to_vec()))
        .collect::<Vec<Vec<String>>>()
}
