#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
  pub val: i32,
  pub next: Option<Box<ListNode>>,
}

impl ListNode {
  #[inline]
  fn new(val: i32) -> Self {
    ListNode { next: None, val }
  }
}

impl Solution {
  /**
   * 1. Two Sum
   */
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

  pub fn add_two_numbers(
    l1: Option<Box<ListNode>>,
    l2: Option<Box<ListNode>>,
  ) -> Option<Box<ListNode>> {
    carried(l1, l2, 0)
  }

  fn carried(
    l1: Option<Box<ListNode>>,
    l2: Option<Box<ListNode>>,
    mut carry: i32,
  ) -> Option<Box<ListNode>> {
    if l1.is_none() && l2.is_none() && carry == 0 {
      None
    } else {
      Some(Box::new(ListNode {
        next: carried(
          l1.and_then(|x| {
            carry += x.val;
            x.next
          }),
          l2.and_then(|x| {
            carry += x.val;
            x.next
          }),
          carry / 10,
        ),
        val: carry % 10,
      }))
    }
  }
}
