pub mod datastructures;

use datastructures::list_node::*;

use std::collections::HashMap;
use std::collections::HashSet;

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
  let mut hashmap: HashMap<i32, usize> = HashMap::with_capacity(nums.len());
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

#[test]
fn test_two_sum() {
  assert_eq!(two_sum(vec![2, 7, 11, 15], 9), vec![0, 1]);
  assert_eq!(two_sum(vec![3, 2, 4], 6), vec![1, 2]);
  assert_eq!(two_sum(vec![3, 3], 6), vec![0, 1]);
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

#[test]
fn test_add_two_numbers() {
  assert_eq!(
    add_two_numbers(
      generate_list_node(vec![2, 4, 3]),
      generate_list_node(vec![5, 6, 4]),
    ),
    generate_list_node(vec![7, 0, 8])
  );
  assert_eq!(
    add_two_numbers(generate_list_node(vec![0]), generate_list_node(vec![0])),
    generate_list_node(vec![0])
  );
  assert_eq!(
    add_two_numbers(
      generate_list_node(vec![9, 9, 9, 9, 9, 9, 9]),
      generate_list_node(vec![9, 9, 9, 9]),
    ),
    generate_list_node(vec![8, 9, 9, 9, 0, 0, 0, 1])
  );
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

#[test]
fn test_reverse_int() {
  assert_eq!(reverse_int(123), 321);
  assert_eq!(reverse_int(-123), -321);
  assert_eq!(reverse_int(120), 21);
  assert_eq!(reverse_int(100), 1);
  assert_eq!(reverse_int(2147483647), 0);
  assert_eq!(reverse_int(-2147483648), 0);
}

/**
 * 27.移除元素
 */
pub fn remove_element(nums: &mut Vec<i32>, val: i32) -> i32 {
  let mut left = 0;
  let n = nums.len();
  for right in 0..n {
    if nums[right] != val {
      nums[left] = nums[right];
      left += 1;
    }
  }
  left as i32
}

#[test]
fn test_remove_element() {
  assert_eq!(remove_element(&mut vec![3, 2, 2, 3], 3), 2);
  assert_eq!(remove_element(&mut vec![0, 1, 2, 2, 3, 0, 4, 2], 2), 5);
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

#[test]
fn test_trailing_zeroes() {
  assert_eq!(trailing_zeroes(3), 0);
  assert_eq!(trailing_zeroes(5), 1);
  assert_eq!(trailing_zeroes(0), 0);
}

/**
 * 357. 统计各位数字都不同的数字个数
 */
pub fn count_numbers_with_unique_digits(n: i32) -> i32 {
  if n == 0 {
    return 1;
  }
  if n == 1 {
    return 10;
  }
  let mut res = 10;
  let mut cur = 9;
  for i in 0..n - 1 {
    cur *= 9 - i;
    res += cur;
  }
  res
}

#[test]
fn test_count_numbers_with_unique_digits() {
  assert_eq!(count_numbers_with_unique_digits(2), 91);
  assert_eq!(count_numbers_with_unique_digits(0), 1);
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

#[test]
fn test_find_diagonal_order() {
  assert_eq!(
    find_diagonal_order(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]),
    vec![1, 2, 4, 7, 5, 3, 6, 8, 9]
  );
  assert_eq!(
    find_diagonal_order(vec![vec![1, 2], vec![3, 4]]),
    vec![1, 2, 3, 4]
  );
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

#[test]
fn test_convert_to_base7() {
  assert_eq!(convert_to_base7(100), "202");
  assert_eq!(convert_to_base7(-7), "-10");
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

#[test]
fn test_complex_number_multiply() {
  assert_eq!(
    complex_number_multiply(String::from("1+1i"), String::from("1+1i")),
    String::from("0+2i")
  );
  assert_eq!(
    complex_number_multiply(String::from("1+-1i"), String::from("1+-1i")),
    String::from("0+-2i")
  );
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

#[test]
fn test_cal_points() {
  assert_eq!(
    cal_points(vec![
      String::from("5"),
      String::from("2"),
      String::from("C"),
      String::from("D"),
      String::from("+"),
    ]),
    30
  );
  assert_eq!(
    cal_points(vec![
      String::from("5"),
      String::from("-2"),
      String::from("4"),
      String::from("C"),
      String::from("D"),
      String::from("9"),
      String::from("+"),
      String::from("+"),
    ]),
    27
  );
  assert_eq!(cal_points(vec![String::from("1")]), 1);
}

/**
 * 693.交替位二进制数
 */
pub fn has_alternating_bits(n: i32) -> bool {
  let a = n ^ (n >> 1);
  a & (a + 1) == 0
}

#[test]
fn test_has_alternating_bits() {
  assert_eq!(has_alternating_bits(5), true);
  assert_eq!(has_alternating_bits(7), false);
  assert_eq!(has_alternating_bits(11), false);
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

#[test]
fn test_self_dividing_numbers() {
  assert_eq!(
    self_dividing_numbers(1, 22),
    vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]
  );
  assert_eq!(self_dividing_numbers(47, 85), vec![48, 55, 66, 77]);
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

#[test]
fn test_next_greatest_letter() {
  assert_eq!(next_greatest_letter(vec!['c', 'f', 'j'], 'a'), 'c');
  assert_eq!(next_greatest_letter(vec!['c', 'f', 'j'], 'c'), 'f');
  assert_eq!(next_greatest_letter(vec!['c', 'f', 'j'], 'd'), 'f');
}

/**
 * 762.二进制表示中质数个计算置位
 */
pub fn count_prime_set_bits(left: i32, right: i32) -> i32 {
  (left..=right).fold(0, |ret, i| ret + (665772 >> i.count_ones() & 1))
}

#[test]
fn test_count_prime_set_bits() {
  assert_eq!(count_prime_set_bits(6, 10), 4);
  assert_eq!(count_prime_set_bits(10, 15), 5);
}

/**
 * 804. 唯一摩尔斯密码词
 */
pub fn unique_morse_representations(words: Vec<String>) -> i32 {
  let morse = vec![
    ".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--",
    "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--..",
  ];
  words
    .iter()
    .fold(HashSet::new(), |mut unique, word| {
      let mut s = String::new();
      word
        .bytes()
        .for_each(|ch| s = format!("{}{}", s, morse[(ch - 'a' as u8) as usize]));
      unique.insert(s);
      unique
    })
    .len() as i32
}

#[test]
fn test_unique_morse_representations() {
  assert_eq!(
    unique_morse_representations(vec![
      String::from("gin"),
      String::from("zen"),
      String::from("gig"),
      String::from("msg"),
    ]),
    2
  );
  assert_eq!(unique_morse_representations(vec![String::from("a")]), 1);
}

/**
 * 806. 写字符串需要的行数
 */
pub fn number_of_lines(widths: Vec<i32>, s: String) -> Vec<i32> {
  let max_wdith = 100;
  let mut lines = 1;
  let mut width = 0;
  for c in s.chars() {
    let need = widths[(c as u8 - b'a') as usize];
    width += need;
    if width > max_wdith {
      width = need;
      lines += 1;
    }
  }
  vec![lines, width]
}

#[test]
fn test_number_of_lines() {
  assert_eq!(
    number_of_lines(
      vec![
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10,
      ],
      String::from("abcdefghijklmnopqrstuvwxyz"),
    ),
    vec![3, 60]
  );
  assert_eq!(
    number_of_lines(
      vec![
        4, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10,
      ],
      String::from("bbbcccdddaaa"),
    ),
    vec![2, 4]
  );
}

/**
 * 1672. 最富有客户的资产总量
 */
pub fn maximum_wealth(accounts: Vec<Vec<i32>>) -> i32 {
  accounts.iter().map(|x| x.iter().sum()).max().unwrap()
}

#[test]
fn test_maximum_wealth() {
  assert_eq!(maximum_wealth(vec![vec![1, 2, 3], vec![3, 2, 1]]), 6);
  assert_eq!(maximum_wealth(vec![vec![1, 5], vec![7, 3], vec![3, 5]]), 10);
  assert_eq!(
    maximum_wealth(vec![vec![2, 8, 7], vec![7, 1, 3], vec![1, 9, 5]]),
    17
  );
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

#[test]
fn test_pivot_index() {
  assert_eq!(pivot_index(vec! {2, 3, -1, 8, 4}), 3);
  assert_eq!(pivot_index(vec! {1, -1, 4}), 2);
  assert_eq!(pivot_index(vec! {2, 5}), -1);
}
