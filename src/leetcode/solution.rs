use crate::leetcode::data_structures::ListNode;
use crate::leetcode::TreeNode;
use std::cell::RefCell;
use std::cmp::max;
use std::collections::HashSet;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;


/// 1. Two Sum
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

/// 2. Add Two Numbers
pub fn add_two_numbers(
    l1: Option<Box<ListNode>>,
    l2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
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
    carried(l1, l2, 0)
}

/// 7. Convert Integer
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

/// 27. Remove Element
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

/// 172. Factorial Trailing Zeroes
pub fn trailing_zeroes(n: i32) -> i32 {
    let mut ans = 0;
    let mut n = n;
    while n != 0 {
        n /= 5;
        ans += n;
    }
    ans
}

/// 357. Count Numbers with Unique Digits
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

/// 386. Lexicographical Numbers
pub fn lexical_order(n: i32) -> Vec<i32> {
    let mut ret: Vec<i32> = Vec::with_capacity(n as usize);
    let mut num = 1;
    for _ in 0..n {
        ret.push(num);
        if num * 10 <= n {
            num *= 10;
        } else {
            while num % 10 == 9 || num + 1 > n {
                num /= 10;
            }
            num += 1;
        }
    }
    ret
}

/// 498. Diagonal Traverse
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

/// 504. Base 7
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

/// 537. Complex Number Multiplication
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

/// 636. Exclusive Time of Functions
pub fn exclusive_time(n: i32, logs: Vec<String>) -> Vec<i32> {
    let mut stack: Vec<Vec<i32>> = Vec::new();
    let mut res: Vec<i32> = vec![0; n as usize];
    let start_command = "start";
    for log in logs {
        let data: Vec<&str> = log.as_str().split(':').collect();
        let index = data[0].parse::<i32>().unwrap();
        let timestamp = data[2].parse::<i32>().unwrap();
        if data[1] == start_command {
            if !stack.is_empty() {
                let top = stack.last().unwrap();
                res[top[0] as usize] += timestamp - top[1];
            }
            stack.push(vec![index, timestamp]);
        } else {
            let pair = stack.pop().unwrap();
            res[pair[0] as usize] += timestamp - pair[1] + 1;
            if !stack.is_empty() {
                let last = stack.len() - 1;
                stack[last][1] = timestamp + 1;
            }
        }
    }
    res
}

/// 646. Maximum Length of Pair Chain
pub fn find_longest_chain(mut pairs: Vec<Vec<i32>>) -> i32 {
    pairs.sort_by(|a, b| a[1].cmp(&b[1]));
    let mut curr = i32::MIN;
    let mut res = 0;
    for pair in pairs {
        if curr < pair[0] {
            curr = pair[1];
            res += 1;
        }
    }
    res
}

/// 658. Find K Closest Elements
pub fn find_closest_elements(arr: Vec<i32>, k: i32, x: i32) -> Vec<i32> {
    let mut right = match arr.binary_search(&x) {
        Ok(i) => i as i32,
        Err(i) => i as i32,
    };
    let n = arr.len() as i32;
    let mut left = right as i32 - 1;
    let mut k = k;
    while k > 0 {
        if left < 0 {
            right += 1;
        } else if right >= n || x - arr[left as usize] <= arr[right as usize] - x {
            left -= 1;
        } else {
            right += 1;
        }
        k -= 1;
    }
    let (left, right) = ((left + 1) as usize, right as usize);
    arr[left..right].to_vec()
}

/// 662. Maximum Width of Binary Tree
pub fn width_of_binary_tree(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
    let mut level_min: HashMap<usize, usize> = HashMap::new();
    fn dfs(
        node: &Option<Rc<RefCell<TreeNode>>>,
        depth: usize,
        index: usize,
        level_min: &mut HashMap<usize, usize>,
    ) -> i32 {
        if node.is_none() {
            return 0;
        }
        if !level_min.contains_key(&depth) {
            level_min.insert(depth, index);
        }
        max(
            (index - level_min[&depth] + 1) as i32,
            max(
                dfs(
                    &node.as_ref().unwrap().borrow().left,
                    depth + 1,
                    index * 2,
                    level_min,
                ),
                dfs(
                    &node.as_ref().unwrap().borrow().right,
                    depth + 1,
                    index * 2 + 1,
                    level_min,
                ),
            ),
        )
    }
    dfs(&root, 1, 1, &mut level_min)
}

/// 682. Baseball Game
pub fn cal_points(ops: Vec<String>) -> i32 {
    ops.iter()
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

/// 693. Binary Number with Alternating bits
pub fn has_alternating_bits(n: i32) -> bool {
    let a = n ^ (n >> 1);
    a & (a + 1) == 0
}

/// 724. Find Pivot Index
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

/// 728. Self Dividing Numbers
pub fn self_dividing_numbers(left: i32, right: i32) -> Vec<i32> {
    let mut ans = Vec::new();
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

    for i in left..=right {
        if is_self_dividing(i) {
            ans.push(i);
        }
    }
    ans
}

/// 744. Find Smallest Letter Greater Than Target
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

/// 762. Prime Number of Set Bits in Binary Representation
pub fn count_prime_set_bits(left: i32, right: i32) -> i32 {
    (left..=right).fold(0, |ret, i| ret + (665772 >> i.count_ones() & 1))
}

/// 793. Preimage Size of Factorial Zeroes Function
pub fn preimage_size_fzf(k: i32) -> i32 {
    fn zeta(mut n: i32) -> i32 {
        let mut res = 0;
        while n != 0 {
            n /= 5;
            res += n;
        }
        res
    }

    fn nx(n: i32) -> i32 {
        let mut right: i64 = 5 * (n as i64);
        let mut left: i64 = 0;
        while left <= right {
            let mid = left + (right - left) / 2;
            if zeta(mid as i32) < n {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return (right + 1) as i32;
    }

    nx(k + 1) - nx(k)
}

/// 804. Unique Morse Code Words
pub fn unique_morse_representations(words: Vec<String>) -> i32 {
    let morse = vec![
        ".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..",
        "--", "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-",
        "-.--", "--..",
    ];
    words
        .iter()
        .fold(HashSet::new(), |mut unique, word| {
            let mut s = String::new();
            word.bytes()
                .for_each(|ch| s = format!("{}{}", s, morse[(ch - 'a' as u8) as usize]));
            unique.insert(s);
            unique
        })
        .len() as i32
}

/// 806. Number of Lines To Write String
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

/// 821. Shortest Distance to a Character
pub fn shortest_to_char(s: String, c: char) -> Vec<i32> {
    let cmp = |initial: &mut i32, ch: char| {
        if ch == c {
            *initial = 0;
        } else {
            *initial += 1;
        }
        Some(*initial)
    };
    let mut a: Vec<_> = s.chars().rev().scan(s.len() as i32, cmp).collect();
    s.chars()
        .scan(s.len() as i32, cmp)
        .zip(a.drain(..).rev())
        .map(|e| e.0.min(e.1))
        .collect()
}

///  883. Projection Area of 3D Shapes
pub fn projection_area(grid: Vec<Vec<i32>>) -> i32 {
    let max_row = grid
        .iter()
        .map(|rows| rows.iter().max().unwrap())
        .sum::<i32>();
    let max_col = (0..grid.len())
        .map(|i| grid.iter().map(|col| col[i]).max().unwrap())
        .sum::<i32>();
    grid.iter()
        .map(|r| r.iter().filter(|c| **c != 0).count() as i32)
        .sum::<i32>()
        + max_row
        + max_col
}

/// 905. Sort Array By Parity
pub fn sort_array_by_parity(mut nums: Vec<i32>) -> Vec<i32> {
    let (mut left, mut right) = (0, nums.len() - 1);
    while left < right {
        while left < right && nums[left] % 2 == 0 {
            left += 1;
        }

        while left < right && nums[right] % 2 == 1 {
            right -= 1;
        }

        if left < right {
            nums.swap(left, right);
            left += 1;
            right -= 1;
        }
    }
    nums
}

/// 942. DI String Match
pub fn di_string_match(s: String) -> Vec<i32> {
    let n = s.len();
    let mut lo = 0;
    let mut hi = n as i32;
    let mut res: Vec<i32> = vec![0; n + 1];
    s.chars().enumerate().for_each(|(i, c)| {
        if c == 'I' {
            res[i] = lo;
            lo += 1
        } else {
            res[i] = hi;
            hi -= 1
        }
    });
    res[n] = lo;
    res
}

/// 944. Delete Columns to Make Sorted
pub fn min_deletion_size(strs: Vec<String>) -> i32 {
    let strs_arr = strs
        .iter()
        .map(|s| s.chars().collect::<Vec<char>>())
        .collect::<Vec<_>>();
    let mut ans = 0;
    for j in 0..strs[0].len() {
        for i in 1..strs.len() {
            if strs_arr[i - 1][j] > strs_arr[i][j] {
                ans += 1;
                break;
            }
        }
    }
    ans
}

/// 946. Validate Stack Sequences
pub fn validate_stack_sequences(pushed: Vec<i32>, popped: Vec<i32>) -> bool {
    let mut stack: VecDeque<i32> = VecDeque::new();
    let n = pushed.len();
    let mut j: usize = 0;
    for i in 0..n {
        stack.push_back(pushed[i]);
        while !stack.is_empty() && stack.back().cloned().unwrap() == popped[j] {
            stack.pop_back();
            j += 1;
        }
    }
    stack.is_empty()
}

/// 998. Maximum Binary Tree II
pub fn insert_into_max_tree(
    root: Option<Rc<RefCell<TreeNode>>>,
    val: i32,
) -> Option<Rc<RefCell<TreeNode>>> {
    let (parent, mut curr) = (Rc::new(RefCell::new(TreeNode::new(val))), root.clone());
    if root.is_some() && val > root.as_ref().unwrap().borrow().val {
        parent.borrow_mut().left = root;
        return Some(parent);
    }
    while curr.as_ref().unwrap().borrow().right.is_some()
        && curr
        .as_ref()
        .unwrap()
        .borrow()
        .right
        .as_ref()
        .unwrap()
        .borrow()
        .val
        > val
    {
        let right = curr.as_ref().unwrap().borrow().right.clone();
        curr = right;
    }
    parent.borrow_mut().left = curr.as_ref().unwrap().borrow().right.clone();
    curr.as_ref().unwrap().borrow_mut().right = Some(parent.clone());
    root
}

/// 1403. Minimum Subsequence in Non-Increasing Order
pub fn min_subsequence(mut nums: Vec<i32>) -> Vec<i32> {
    let total: i32 = nums.iter().sum();
    nums.sort_by(|a, b| b.cmp(a));
    let mut curr = 0;
    let mut ans: Vec<i32> = Vec::new();
    for num in nums {
        curr += num;
        ans.push(num);
        if total - curr < curr {
            break;
        }
    }
    ans
}

/// 1408. String Matching in an Array
pub fn string_matching(words: Vec<String>) -> Vec<String> {
    let mut res: Vec<String> = vec![];
    for i in 0..words.len() {
        let word: &String = &words[i];
        for j in 0..words.len() {
            let word2: &String = &words[j];
            if i != j && word2.contains(word.as_str()) {
                res.push(word.clone());
                break;
            }
        }
    }
    res
}

/// 1450. Number of Students Doing Homework at a Given Time
pub fn busy_student(start_time: Vec<i32>, end_time: Vec<i32>, query_time: i32) -> i32 {
    let mut res = 0;
    for i in 0..start_time.len() {
        if start_time[i] <= query_time && query_time <= end_time[i] {
            res += 1;
        }
    }
    res
}

/// 1455. Check If a Word Occurs As a Prefix of Any Word in a Sentence
pub fn is_prefix_of_word(sentence: String, search_word: String) -> i32 {
    let words = sentence.as_str().split_ascii_whitespace();
    let mut index = 1;
    for word in words {
        if word.starts_with(search_word.as_str()) {
            return index;
        }
        index += 1;
    }
    -1
}

/// 1460. Make Two Arrays Equal by Reversing Sub-arrays
pub fn can_be_equal(mut target: Vec<i32>, mut arr: Vec<i32>) -> bool {
    target.sort();
    arr.sort();
    target == arr
}

/// 1464. Maximum Product of Two Elements in an Array
pub fn max_product(nums: Vec<i32>) -> i32 {
    let mut a = nums[0];
    let mut b = nums[1];
    if a < b {
        let temp = a;
        a = b;
        b = temp;
    }
    for i in 2..nums.len() {
        if a < nums[i] {
            b = a;
            a = nums[i];
        } else if nums[i] > b {
            b = nums[i];
        }
    }
    (a - 1) * (b - 1)
}

/// 1470. Shuffle the Array
pub fn shuffle(nums: Vec<i32>, n: i32) -> Vec<i32> {
    let mut res: Vec<i32> = vec![0; (n * 2) as usize];
    let n = n as usize;
    for i in 0..n {
        res[2 * i] = nums[i];
        res[2 * i + 1] = nums[n + i];
    }
    res
}

/// 1475. Final Prices With a Special Discount in a Shop
pub fn final_prices(prices: Vec<i32>) -> Vec<i32> {
    let n = prices.len();
    let mut res: Vec<i32> = vec![0; n];
    let mut stack: Vec<i32> = vec![];
    for i in (0..n).rev() {
        while !stack.is_empty() && stack[stack.len() - 1] > prices[i] {
            stack.pop();
        }
        res[i] = match stack.is_empty() {
            true => prices[i],
            false => prices[i] - stack[stack.len() - 1],
        };
        stack.push(prices[i]);
    }
    res
}

/// 1582. Special Positions in a Binary Matrix
pub fn num_special(mut mat: Vec<Vec<i32>>) -> i32 {
    let m = mat.len();
    let n = mat[0].len();
    for i in 0..m {
        let mut count = 0;
        for j in 0..n {
            if mat[i][j] == 1 {
                count += 1;
            }
        }
        if i == 0 {
            count -= 1;
        }
        if count > 0 {
            for j in 0..n {
                if mat[i][j] == 1 {
                    mat[0][j] += count;
                }
            }
        }
    }
    let mut sum = 0;
    for num in mat[0].to_vec() {
        if num == 1 {
            sum += 1;
        }
    }
    sum
}

/// 1672. Richest Customer Wealth
pub fn maximum_wealth(accounts: Vec<Vec<i32>>) -> i32 {
    accounts.iter().map(|x| x.iter().sum()).max().unwrap()
}

/// 1823. Find the Winner of the Circular Game
pub fn find_the_winner(n: i32, k: i32) -> i32 {
    let mut winner = 1;
    for i in 2..=n {
        winner = (winner + k - 1) % i + 1;
    }
    winner
}

/// 1991. Find the Middle Index in Array
pub fn find_middle_index(nums: Vec<i32>) -> i32 {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_list_node(nums: Vec<i32>) -> Option<Box<ListNode>> {
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

    #[test]
    fn test_two_sum() {
        assert_eq!(two_sum(vec![2, 7, 11, 15], 9), vec![0, 1]);
        assert_eq!(two_sum(vec![3, 2, 4], 6), vec![1, 2]);
        assert_eq!(two_sum(vec![3, 3], 6), vec![0, 1]);
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

    #[test]
    fn test_reverse_int() {
        assert_eq!(reverse_int(123), 321);
        assert_eq!(reverse_int(-123), -321);
        assert_eq!(reverse_int(120), 21);
        assert_eq!(reverse_int(100), 1);
        assert_eq!(reverse_int(2147483647), 0);
        assert_eq!(reverse_int(-2147483648), 0);
    }

    #[test]
    fn test_remove_element() {
        assert_eq!(remove_element(&mut vec![3, 2, 2, 3], 3), 2);
        assert_eq!(
            remove_element(&mut vec![0, 1, 2, 2, 3, 0, 4, 2], 2),
            5
        );
    }

    #[test]
    fn test_trailing_zeroes() {
        assert_eq!(trailing_zeroes(3), 0);
        assert_eq!(trailing_zeroes(5), 1);
        assert_eq!(trailing_zeroes(0), 0);
    }

    #[test]
    fn test_count_numbers_with_unique_digits() {
        assert_eq!(count_numbers_with_unique_digits(2), 91);
        assert_eq!(count_numbers_with_unique_digits(0), 1);
    }

    #[test]
    fn test_lexical_order() {
        assert_eq!(
            lexical_order(13),
            vec![1, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9]
        );
        assert_eq!(lexical_order(2), vec![1, 2]);
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

    #[test]
    fn test_convert_to_base7() {
        assert_eq!(convert_to_base7(100), "202");
        assert_eq!(convert_to_base7(-7), "-10");
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

    #[test]
    fn test_exclusive_time() {
        assert_eq!(
            vec![3, 4],
            exclusive_time(
                2,
                vec!["0:start:0", "1:start:2", "1:end:5", "0:end:6"]
                    .iter()
                    .map(|&x| x.to_string())
                    .collect(),
            )
        );
        assert_eq!(
            vec![8],
            exclusive_time(
                1,
                vec![
                    "0:start:0",
                    "0:start:2",
                    "0:end:5",
                    "0:start:6",
                    "0:end:6",
                    "0:end:7",
                ]
                    .iter()
                    .map(|&x| x.to_string())
                    .collect(),
            )
        );
        assert_eq!(
            vec![7, 1],
            exclusive_time(
                2,
                vec![
                    "0:start:0",
                    "0:start:2",
                    "0:end:5",
                    "1:start:6",
                    "1:end:6",
                    "0:end:7",
                ]
                    .iter()
                    .map(|&x| x.to_string())
                    .collect(),
            )
        );
        assert_eq!(
            vec![8, 1],
            exclusive_time(
                2,
                vec![
                    "0:start:0",
                    "0:start:2",
                    "0:end:5",
                    "1:start:7",
                    "1:end:7",
                    "0:end:8",
                ]
                    .iter()
                    .map(|&x| x.to_string())
                    .collect(),
            )
        );
        assert_eq!(
            vec![1],
            exclusive_time(
                1,
                vec!["0:start:0", "0:end:0"]
                    .iter()
                    .map(|&x| x.to_string())
                    .collect(),
            )
        );
    }

    #[test]
    fn test_find_longest_chain() {
        assert_eq!(find_longest_chain(vec![vec![1, 2], vec![2, 3], vec![3, 4]]), 2);
        assert_eq!(find_longest_chain(vec![vec![1, 2], vec![7, 8], vec![4, 5]]), 3);
    }

    #[test]
    fn test_find_closest_elements() {
        assert_eq!(
            find_closest_elements(vec![1, 2, 3, 4, 5], 4, 3),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            find_closest_elements(vec![1, 2, 3, 4, 5], 4, -1),
            vec![1, 2, 3, 4]
        );
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

    #[test]
    fn test_has_alternating_bits() {
        assert_eq!(has_alternating_bits(5), true);
        assert_eq!(has_alternating_bits(7), false);
        assert_eq!(has_alternating_bits(11), false);
    }

    #[test]
    fn test_pivot_index() {
        assert_eq!(
            self_dividing_numbers(1, 22),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]
        );
        assert_eq!(self_dividing_numbers(47, 85), vec![48, 55, 66, 77]);
    }

    #[test]
    fn test_self_dividing_numbers() {
        assert_eq!(
            self_dividing_numbers(1, 22),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]
        );
        assert_eq!(self_dividing_numbers(47, 85), vec![48, 55, 66, 77]);
    }

    #[test]
    fn test_next_greatest_letter() {
        assert_eq!(next_greatest_letter(vec!['c', 'f', 'j'], 'a'), 'c');
        assert_eq!(next_greatest_letter(vec!['c', 'f', 'j'], 'c'), 'f');
        assert_eq!(next_greatest_letter(vec!['c', 'f', 'j'], 'd'), 'f');
    }

    #[test]
    fn test_count_prime_set_bits() {
        assert_eq!(count_prime_set_bits(6, 10), 4);
        assert_eq!(count_prime_set_bits(10, 15), 5);
    }

    #[test]
    fn test_preimage_size_fzf() {
        assert_eq!(preimage_size_fzf(0), 5);
        assert_eq!(preimage_size_fzf(5), 0);
        assert_eq!(preimage_size_fzf(3), 5);
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
        assert_eq!(
            unique_morse_representations(vec![String::from("a")]),
            1
        );
    }

    #[test]
    fn test_number_of_lines() {
        assert_eq!(
            number_of_lines(
                vec![
                    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                    10, 10, 10, 10, 10, 10,
                ],
                String::from("abcdefghijklmnopqrstuvwxyz"),
            ),
            vec![3, 60]
        );
        assert_eq!(
            number_of_lines(
                vec![
                    4, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                    10, 10, 10, 10, 10, 10,
                ],
                String::from("bbbcccdddaaa"),
            ),
            vec![2, 4]
        );
    }

    #[test]
    fn test_shortest_to_char() {
        assert_eq!(
            shortest_to_char(String::from("loveleetcode"), 'e'),
            vec![3, 2, 1, 0, 1, 0, 0, 1, 2, 2, 1, 0]
        );
        assert_eq!(
            shortest_to_char(String::from("aaab"), 'b'),
            vec![3, 2, 1, 0]
        );
    }

    #[test]
    fn test_projection_area() {
        assert_eq!(projection_area(vec![vec![1, 2], vec![3, 4]]), 17);
        assert_eq!(projection_area(vec![vec![2]]), 5);
        assert_eq!(projection_area(vec![vec![1, 0], vec![0, 2]]), 8);
    }

    #[test]
    fn test_sort_array_by_parity() {
        assert_eq!(
            sort_array_by_parity(vec![3, 1, 2, 4]),
            vec![4, 2, 1, 3]
        );
        assert_eq!(sort_array_by_parity(vec![0]), vec![0]);
    }

    #[test]
    fn test_di_string_match() {
        assert_eq!(
            di_string_match(String::from("IDID")),
            vec![0, 4, 1, 3, 2]
        );
        assert_eq!(
            di_string_match(String::from("III")),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            di_string_match(String::from("DDI")),
            vec![3, 2, 0, 1]
        );
    }

    #[test]
    fn test_min_deletion_size() {
        assert_eq!(
            min_deletion_size(vec![
                "cba".to_string(),
                "daf".to_string(),
                "ghi".to_string(),
            ]),
            1
        );
        assert_eq!(
            min_deletion_size(vec!["a".to_string(), "b".to_string()]),
            0
        );
        assert_eq!(
            min_deletion_size(vec![
                "zyx".to_string(),
                "wvu".to_string(),
                "tsr".to_string(),
            ]),
            3
        );
    }

    #[test]
    fn test_min_subsequence() {
        assert_eq!(min_subsequence(vec![4, 3, 10, 9, 8]), vec![10, 9]);
        assert_eq!(min_subsequence(vec![4, 4, 7, 6, 7]), vec![7, 7, 6]);
        assert_eq!(min_subsequence(vec![6]), vec![6]);
    }

    #[test]
    fn test_string_matching() {
        let words1: Vec<String> = vec!["mass", "as", "hero", "superhero"]
            .iter()
            .map(|&x| x.to_string())
            .collect();
        let words2: Vec<String> = vec!["leetcode", "et", "code"]
            .iter()
            .map(|&x| x.to_string())
            .collect();
        let words3: Vec<String> = vec!["blue", "green", "bu"]
            .iter()
            .map(|&x| x.to_string())
            .collect();
        let expected1: Vec<String> = vec!["as", "hero"].iter().map(|&x| x.to_string()).collect();
        let expected2: Vec<String> = vec!["et", "code"].iter().map(|&x| x.to_string()).collect();
        let expected3: Vec<String> = vec![];
        assert_eq!(string_matching(words1), expected1);
        assert_eq!(string_matching(words2), expected2);
        assert_eq!(string_matching(words3), expected3);
    }

    #[test]
    fn test_busy_student() {
        assert_eq!(busy_student(vec![1, 2, 3], vec![3, 2, 7], 4), 1);
        assert_eq!(busy_student(vec![4], vec![4], 4), 1);
    }

    #[test]
    fn test_is_prefix_of_word() {
        assert_eq!(
            is_prefix_of_word("i love eating burger".to_string(), "burg".to_string()),
            4
        );
        assert_eq!(
            is_prefix_of_word(
                "this problem is an easy problem".to_string(),
                "pro".to_string(),
            ),
            2
        );
        assert_eq!(
            is_prefix_of_word("i am tired".to_string(), "you".to_string()),
            -1
        );
    }

    #[test]
    fn test_can_be_equal() {
        assert_eq!(
            can_be_equal(vec![1, 2, 3, 4], vec![2, 1, 3, 4]),
            true
        );
        assert_eq!(can_be_equal(vec![7], vec![7]), true);
        assert_eq!(can_be_equal(vec![3, 7, 9], vec![3, 7, 11]), false);
    }

    #[test]
    fn test_max_product() {
        assert_eq!(max_product(vec![3, 4, 5, 2]), 12);
        assert_eq!(max_product(vec![1, 5, 4, 5]), 16);
        assert_eq!(max_product(vec![3, 7]), 12);
    }

    #[test]
    fn test_shuffle() {
        assert_eq!(
            vec![2, 3, 5, 4, 1, 7],
            shuffle(vec![2, 5, 1, 3, 4, 7], 3)
        );
        assert_eq!(
            vec![1, 4, 2, 3, 3, 2, 4, 1],
            shuffle(vec![1, 2, 3, 4, 4, 3, 2, 1], 4)
        );
        assert_eq!(vec![1, 2, 1, 2], shuffle(vec![1, 1, 2, 2], 2));
    }

    #[test]
    fn test_final_prices() {
        assert_eq!(
            vec![4, 2, 4, 2, 3],
            final_prices(vec![8, 4, 6, 2, 3])
        );
        assert_eq!(
            vec![1, 2, 3, 4, 5],
            final_prices(vec![1, 2, 3, 4, 5])
        );
        assert_eq!(vec![9, 0, 1, 6], final_prices(vec![10, 1, 1, 6]));
    }

    #[test]
    fn test_num_special() {
        assert_eq!(num_special(vec![vec![1, 0, 0], vec![0, 0, 1], vec![1, 0, 0]]), 1);
        assert_eq!(num_special(vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]), 3);
    }

    #[test]
    fn test_maximum_wealth() {
        assert_eq!(
            maximum_wealth(vec![vec![1, 2, 3], vec![3, 2, 1]]),
            6
        );
        assert_eq!(
            maximum_wealth(vec![vec![1, 5], vec![7, 3], vec![3, 5]]),
            10
        );
        assert_eq!(
            maximum_wealth(vec![vec![2, 8, 7], vec![7, 1, 3], vec![1, 9, 5]]),
            17
        );
    }

    #[test]
    fn test_find_the_winner() {
        assert_eq!(find_the_winner(5, 2), 3);
        assert_eq!(find_the_winner(6, 5), 1);
    }

    #[test]
    fn test_find_middle_index() {
        assert_eq!(find_middle_index(vec! {2, 3, -1, 8, 4}), 3);
        assert_eq!(find_middle_index(vec! {1, -1, 4}), 2);
        assert_eq!(find_middle_index(vec! {2, 5}), -1);
    }
}
