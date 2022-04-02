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

pub fn generate_list_node(nums: Vec<i32>) -> Option<Box<ListNode>> {
  let n = nums.len();
  if n <= 0 {
    return None;
  } else {
    let mut head = ListNode {
      val: nums[0],
      next: None,
    };
    for i in 1..n {
      head.append(nums[i]);
    }
    Some(Box::new(head))
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
 * 172.阶乘后的零
 */
pub fn trailing_zeroes(n: i32) -> i32 {
  let mut ans = 0;
  let mut n = n;
  while n != 0 {
    n /= 5;
    ans += n;
  }
  ans
}

/**
 * 498.对角线遍历
 */
pub fn find_diagonal_order(mat: Vec<Vec<i32>>) -> Vec<i32> {
  if mat.len() == 0 {
    return Vec::new();
  }
  let n = mat.len();
  let m = mat[0].len();
  let mut row: i32 = 0;
  let mut col: i32 = 0;
  let mut direction = 1;
  let mut res: Vec<i32> = vec![0; n * m];
  let mut r = 0;
  while row < n as i32 && col < m as i32 {
    res[r] = mat[row as usize][col as usize];
    r += 1;
    let new_row = if direction == 1 { row - 1 } else { row + 1 };
    let new_col = if direction == 1 { col + 1 } else { col - 1 };
    if new_row < 0 || new_row == n as i32 || new_col < 0 || new_col == m as i32 {
      if direction == 1 {
        row += if col == m as i32 - 1 { 1 } else { 0 };
        col += if col < m as i32 - 1 { 1 } else { 0 };
      } else {
        col += if row == n as i32 - 1 { 1 } else { 0 };
        row += if row < n as i32 - 1 { 1 } else { 0 };
      }
      direction = 1 - direction;
    } else {
      row = new_row;
      col = new_col;
    }
  }
  res
}

/**
 * 504.七进制数
 */
pub fn convert_to_base7(num: i32) -> String {
  let mut num = num;
  if num == 0 {
    return String::from("0");
  }
  let negative = num < 0;
  num = i32::abs(num);
  let mut digits = String::from("");
  while num > 0 {
    digits += &(num % 7).to_string();
    num /= 7;
  }
  if negative {
    digits += &"-".to_string();
  }
  digits.chars().rev().collect()
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

/**
 * 682.棒球比赛
 */
pub fn cal_points(ops: Vec<String>) -> i32 {
  ops
    .iter()
    .map(|x| x.as_str())
    .fold((vec![0; ops.len()], 0), |(mut cache, i), op| match op {
      "C" => {
        cache[i - 1] = 0;
        (cache, i - 1)
      }
      "D" => {
        cache[i] = cache[i - 1] * 2;
        (cache, i + 1)
      }
      "+" => {
        cache[i] = cache[i - 1] + cache[i - 2];
        (cache, i + 1)
      }
      _ => {
        cache[i] = op.parse().unwrap();
        (cache, i + 1)
      }
    })
    .0
    .iter()
    .sum()
}

/**
 * 693.交替位二进制数
 */
pub fn has_alternating_bits(n: i32) -> bool {
  let a = n ^ (n >> 1);
  a & (a + 1) == 0
}

/**
 * 728.自除数
 */
pub fn self_dividing_numbers(left: i32, right: i32) -> Vec<i32> {
  let mut ans = Vec::new();
  for i in left..=right {
    if is_self_dividing(i) {
      ans.push(i);
    }
  }
  ans
}
fn is_self_dividing(num: i32) -> bool {
  let mut tmp = num;
  while tmp > 0 {
    let digit = tmp % 10;
    if digit == 0 || num % digit != 0 {
      return false;
    }
    tmp /= 10;
  }
  true
}

/**
 * 744.寻找比目标字母大的最小字母
 */
pub fn next_greatest_letter(letters: Vec<char>, target: char) -> char {
  let len = letters.len();
  if target >= letters[len - 1] {
    return letters[0];
  }
  let mut low = 0;
  let mut high = len - 1;
  while low < high {
    let mid = (high - low) / 2 + low;
    if letters[mid] > target {
      high = mid;
    } else {
      low = mid + 1;
    }
  }
  return letters[low];
}

/**
 * 1991.寻找数组的中间位置
 */
pub fn pivot_index(nums: Vec<i32>) -> i32 {
  let total: i32 = nums.iter().sum();
  let mut sum: i32 = 0;
  for i in 0..nums.len() {
    if 2 * sum + nums[i] == total {
      return i as i32;
    }
    sum += nums[i];
  }
  -1
}
