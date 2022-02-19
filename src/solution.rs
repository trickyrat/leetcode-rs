impl Solution {
  pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
    let mut hashmap: std::collections::HashMap<i32, usize> =
      std::collections::HashMap::with_capacity(nums.len());
    for i in 0..nums.len() {
      if let Some(k) = hashmap.get(&(target - nums[i])) {
        if *k != i {
          return vec![*k as i32, i as i32];
        }
      }
      hashmap.insert(nums[i], i);
    }
    panic!("???")
  }
}
