#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
  pub val: i32,
  pub next: Option<Box<ListNode>>,
}
impl ListNode {
  #[inline]
  pub fn new(val: i32) -> Self {
    ListNode { next: None, val }
  }
}

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

/**
 * 2. 两数相加
 */
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
/**
 * 7. 整数转换
 */
pub fn reverse_int(x: i32) -> i32 {
  let mut x = x;
  let mut res = 0;
  while x != 0 {
    if res < i32::MIN / 10 || res > i32::MAX / 10 {
      return 0;
    }
    let digit = x % 10;
    x /= 10;
    res = res * 10 + digit;
  }
  res
}

/**
 * 537. 复数乘法
 */
pub fn complex_number_multiply(num1: String, num2: String) -> String {
  let &complex1 = &num1[..num1.len() - 1].split_once('+').unwrap();
  let &complex2 = &num2[..num2.len() - 1].split_once('+').unwrap();
  let (real1, imag1): (i32, i32) = (complex1.0.parse().unwrap(), complex1.1.parse().unwrap());
  let (real2, imag2): (i32, i32) = (complex2.0.parse().unwrap(), complex2.1.parse().unwrap());
  format!(
    "{}+{}i",
    real1 * real2 - imag1 * imag2,
    real1 * imag2 + imag1 * real2
  )
}
