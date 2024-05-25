#![cfg_attr(debug_assertions, allow(unused))]

use crate::leetcode::data_structures::ListNode;
use crate::leetcode::TreeNode;
use std::cell::RefCell;
use std::cmp::{max, min, Ordering};
use std::collections::hash_map::Entry::Vacant;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::env::temp_dir;
use std::iter::Sum;
use std::mem::swap;
use std::rc::Rc;

/// 1.Two Sum
pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
    let mut map: HashMap<i32, usize> = HashMap::with_capacity(nums.len());
    for (i, num) in nums.iter().enumerate() {
        if let Some(k) = map.get(&(target - num)) {
            if *k != i {
                return vec![*k as i32, i as i32];
            }
        }
        map.insert(nums[i], i);
    }
    vec![]
}

/// 2.Add Two Numbers
pub fn add_two_numbers(
    l1: Option<Box<ListNode>>,
    l2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    let (mut l1, mut l2) = (l1, l2);
    let mut dummy = ListNode::new(0);
    let mut curr = &mut dummy;
    let mut carry = 0;

    while l1.is_some() || l2.is_some() || carry != 0 {
        let mut sum = carry;
        if let Some(node) = l1 {
            sum += node.val;
            l1 = node.next;
        }
        if let Some(node) = l2 {
            sum += node.val;
            l2 = node.next;
        }
        carry = sum / 10;
        curr.next = Some(Box::new(ListNode::new(sum % 10)));
        curr = curr.next.as_mut().unwrap();
    }
    dummy.next
}

/// 3.Longest Substring Without Repeating Characters
pub fn length_of_longest_substring(s: String) -> i32 {
    let (n, mut res) = (s.len(), 0);
    // let mut map = HashMap::<u8, i32>::new();
    // let mut i = 0;
    // let chars = s.as_bytes();
    // for j in 0..n {
    //     if map.contains_key(&chars[j]) {
    //         i = max(*map.get(&chars[j]).unwrap(), i);
    //         map.entry(chars[j]).and_modify(|e| *e = j as i32 + 1);
    //     } else {
    //         map.insert(chars[j], (j + 1) as i32);
    //     }
    //     res = max(res, j as i32 - i + 1);
    // }

    let mut map = vec![0; 128];
    let chars = s.as_bytes();
    let mut i = 0;
    for j in 0..n {
        i = max(map[chars[j] as usize], i);
        res = max(res, j as i32 - i + 1);
        map[chars[j] as usize] = j as i32 + 1;
    }
    res
}

/// 7.Convert Integer
pub fn reverse_int(x: i32) -> i32 {
    let mut x = x;
    let mut res = 0;
    while x != 0 {
        if !(i32::MIN / 10..=i32::MAX / 10).contains(&res) {
            return 0;
        }
        let digit = x % 10;
        x /= 10;
        res = res * 10 + digit;
    }
    res
}

/// 19.Remove Nth Node From End of List
pub fn remove_nth_from_end(head: Option<Box<ListNode>>, n: i32) -> Option<Box<ListNode>> {
    let mut dummy = Box::new(ListNode {
        val: -1,
        next: head,
    });
    let mut fast = &mut dummy.clone();
    let mut slow = &mut dummy;
    let n = n as usize;
    for _ in 0..n {
        fast = fast.next.as_mut().unwrap();
    }

    while fast.next.is_some() {
        fast = fast.next.as_mut().unwrap();
        slow = slow.next.as_mut().unwrap();
    }
    slow.next = slow.next.take().unwrap().next.clone();
    dummy.next
}

/// 21.Merge Two Sorted List
pub fn merge_two_sorted_lists(
    mut list1: Option<Box<ListNode>>,
    mut list2: Option<Box<ListNode>>,
) -> Option<Box<ListNode>> {
    let mut dummy_head = ListNode::new(-1);
    let mut p = &mut dummy_head;
    while let (Some(p1), Some(p2)) = (list1.as_ref(), list2.as_ref()) {
        let curr = if p1.val < p2.val {
            &mut list1
        } else {
            &mut list2
        };
        p.next = curr.take();
        p = p.next.as_mut().unwrap();
        *curr = p.next.take();
    }
    p.next = list1.or(list2);
    dummy_head.next
}

/// 23.Merge K Sorted Lists
pub fn merge_k_lists(lists: Vec<Option<Box<ListNode>>>) -> Option<Box<ListNode>> {
    if lists.is_empty() {
        return None;
    }
    let mut dummy = ListNode::new(-1);
    let mut p = &mut dummy;
    let mut pq = BinaryHeap::with_capacity(lists.len());
    for head in lists {
        match head {
            Some(v) => pq.push(v),
            None => {}
        }
    }

    while !pq.is_empty() {
        let mut node = pq.pop().unwrap();
        if pq.is_empty() {
            p.next = Some(node);
            break;
        }
        let next = node.next.take();
        if next.is_some() {
            pq.push(next.unwrap());
        }
        p.next = Some(node);
        p = p.next.as_mut().unwrap();
    }

    dummy.next
}

/// 26.Remove Duplicates from Sorted Array
pub fn remove_duplicates(nums: &mut Vec<i32>) -> i32 {
    let mut slow = 1;
    for fast in (1..nums.len()) {
        if nums[fast] != nums[fast - 1] {
            nums[slow] = nums[fast];
            slow += 1;
        }
    }
    slow as i32
}

/// 27.Remove Element
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

/// 33.Search in Rotated Sorted Array
pub fn search(nums: Vec<i32>, target: i32) -> i32 {
    let n = nums.len();
    if n == 0 {
        return -1;
    }
    if n == 1 {
        return if nums[0] == target { 0 } else { -1 };
    }
    let (mut l, mut r) = (0, (n - 1) as i32);
    while l <= r {
        let mid = (l + r) / 2;
        if nums[mid as usize] == target {
            return mid;
        }
        if (nums[0] <= nums[mid as usize]) {
            if (nums[0] <= target && target < nums[mid as usize]) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        } else {
            if (nums[mid as usize] < target && target <= nums[n - 1]) {
                l = mid + 1;
            } else {
                r = mid - 1;
            }
        }
    }
    -1
}

/// 35.Search Insert Position
pub fn search_insert(nums: Vec<i32>, target: i32) -> i32 {
    let n = nums.len() as i32;
    let (mut left, mut right, mut res) = (0, n - 1, n);
    while left <= right {
        let mid = left + (right - left) / 2;
        if nums[mid as usize] >= target {
            res = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    res
}

/// 39.Combination Sum
pub fn combination_sum(candidates: &Vec<i32>, target: i32) -> Vec<Vec<i32>> {
    fn dfs(
        _candidates: &Vec<i32>,
        _target: i32,
        _ans: &mut Vec<Vec<i32>>,
        _combine: &mut Vec<i32>,
        start_index: usize,
    ) {
        if start_index == _candidates.len() {
            return;
        }
        if _target == 0 {
            _ans.push(_combine.clone());
            return;
        }
        dfs(_candidates, _target, _ans, _combine, start_index + 1);
        if _target - _candidates[start_index] >= 0 {
            _combine.push(_candidates[start_index]);
            dfs(
                _candidates,
                _target - _candidates[start_index],
                _ans,
                _combine,
                start_index,
            );
            _combine.pop();
        }
    }
    let mut ans: Vec<Vec<i32>> = vec![];
    let mut combine: Vec<i32> = vec![];
    dfs(&candidates, target, &mut ans, &mut combine, 0);
    ans
}

/// 86.Partition List
pub fn partition(mut head: Option<Box<ListNode>>, x: i32) -> Option<Box<ListNode>> {
    let (mut dummy_head1, mut dummy_head2) = (ListNode::new(-1), ListNode::new(-1));
    let (mut p1, mut p2) = (&mut dummy_head1, &mut dummy_head2);
    while let Some(mut p) = head {
        head = p.next.take();

        if p.val >= x {
            p2.next = Some(p);
            p2 = p2.next.as_mut().unwrap();
        } else {
            p1.next = Some(p);
            p1 = p1.next.as_mut().unwrap();
        }
    }
    p1.next = dummy_head2.next;
    dummy_head1.next
}

// 118.Pascal's Triangle
pub fn generate(num_rows: i32) -> Vec<Vec<i32>> {
    let mut res = vec![];
    for i in 0..num_rows as usize {
        res.push(vec![1; i + 1]);
        for j in 1..i {
            res[i][j] = res[i - 1][j - 1] + res[i - 1][j];
        }
    }
    res
}

/// 172.Factorial Trailing Zeroes
pub fn trailing_zeroes(n: i32) -> i32 {
    let mut ans = 0;
    let mut n = n;
    while n != 0 {
        n /= 5;
        ans += n;
    }
    ans
}

/// 189.Rotate Array
pub fn rotate(nums: &mut Vec<i32>, mut k: i32) {
    fn reverse(arr: &mut Vec<i32>, mut start: i32, mut end: i32) {
        while start < end {
            arr.swap(start as usize, end as usize);
            start += 1;
            end -= 1;
        }
    }
    let n = nums.len();
    k %= (n as i32);
    reverse(nums, 0, (n - 1) as i32);
    reverse(nums, 0, k - 1);
    reverse(nums, k, (n - 1) as i32);
}

/// 216. Combination Sum III
pub fn combination_sum3(k: i32, n: i32) -> Vec<Vec<i32>> {
    let mut temp: Vec<i32> = vec![];
    let mut ans: Vec<Vec<i32>> = vec![];

    fn dfs(curr: i32, n: i32, k: i32, sum: i32, _temp: &mut Vec<i32>, _ans: &mut Vec<Vec<i32>>) {
        let length = _temp.len() as i32;
        if length + (n - curr + 1) < k || length > k {
            return;
        }

        if length == k && _temp.iter().sum::<i32>() == sum {
            _ans.push(_temp.clone());
            return;
        }

        _temp.push(curr);
        dfs(curr + 1, n, k, sum, _temp, _ans);
        _temp.pop();
        dfs(curr + 1, n, k, sum, _temp, _ans);
    }
    dfs(1, 9, k, n, &mut temp, &mut ans);
    ans
}

/// 283.Move Zeroes
pub fn move_zeroes(nums: &mut Vec<i32>) {
    let n = nums.len();
    let (mut left, mut right) = (0, 0);
    while right < n {
        if nums[right] != 0 {
            nums.swap(left, right);
            left += 1;
        }
        right += 1;
    }
}

/// 331.Verify Preorder Serialization of a Binary Tree
pub fn is_valid_serialization(preorder: String) -> bool {
    let n = preorder.len();
    let (mut i, mut slots) = (0, 1);

    for ch in preorder.chars().into_iter() {
        if slots == 0 {
            return false;
        }
        if ch == ',' {
            i += 1;
        } else if ch == '#' {
            slots -= 1;
            i += 1;
        } else {
            while i < n && ch != ',' {
                i += 1;
            }
            slots += 1;
        }
    }
    slots == 0
}

/// 357.Count Numbers with Unique Digits
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

/// 377. Combination Sum IV
pub fn combination_sum4(nums: Vec<i32>, target: i32) -> i32 {
    let mut dp = vec![0; (target + 1) as usize];
    dp[0] = 1;
    for i in 1..=target {
        for num in nums.iter() {
            if *num <= i {
                dp[i as usize] += dp[(i - num) as usize];
            }
        }
    }
    dp[target as usize]
}

/// 386.Lexicographical Numbers
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

/// 396.Rotate Function
pub fn max_rotate_function(nums: Vec<i32>) -> i32 {
    let (mut f, mut n) = (0, nums.len());
    let nums_sum: i32 = nums.iter().sum();
    for i in 0..n {
        f += (i as i32) * nums[i];
    }
    let mut res = f;
    for i in (0..n).rev() {
        f += nums_sum - (n as i32) * nums[i];
        res = max(res, f);
    }
    res
}

/// 453.Minimum Moves to Equal Array Elements
pub fn min_moves(nums: Vec<i32>) -> i32 {
    let min_num = nums.iter().min().unwrap();
    let mut res = 0;
    for num in nums.iter() {
        res += num - min_num;
    }
    res
}

/// 498.Diagonal Traverse
pub fn find_diagonal_order(mat: Vec<Vec<i32>>) -> Vec<i32> {
    if mat.is_empty() {
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
                row += i32::from(col == m as i32 - 1);
                col += i32::from(col < m as i32 - 1);
            } else {
                col += i32::from(row == n as i32 - 1);
                row += i32::from(row < n as i32 - 1);
            }
            direction = 1 - direction;
        } else {
            row = new_row;
            col = new_col;
        }
    }
    res
}

/// 504.Base 7
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
        digits.push('-');
    }
    digits.chars().rev().collect()
}

/// 518.Coin Change II
pub fn change(amount: i32, coins: Vec<i32>) -> i32 {
    let mut dp = vec![0; amount as usize + 1];
    dp[0] = 1;
    coins.iter().for_each(|&coin| {
        for i in coin..=amount {
            dp[i as usize] += dp[(i - coin) as usize];
        }
    });
    dp[amount as usize]
}

/// 537.Complex Number Multiplication
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

/// 636.Exclusive Time of Functions
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

/// 646.Maximum Length of Pair Chain
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

/// 658.Find K Closest Elements
pub fn find_closest_elements(arr: Vec<i32>, k: i32, x: i32) -> Vec<i32> {
    let mut right = match arr.binary_search(&x) {
        Ok(i) => i as i32,
        Err(i) => i as i32,
    };
    let n = arr.len() as i32;
    let mut left = right - 1;
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

/// 662.Maximum Width of Binary Tree
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
        level_min.entry(depth).or_insert(index);
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

/// 665.Non-decreasing Array
pub fn check_possibility(mut nums: Vec<i32>) -> bool {
    let mut count = 0;
    let n = nums.len();
    for i in 0..n - 1 {
        let (x, y) = (nums[i], nums[i + 1]);
        if x > y {
            count += 1;
            if count > 1 {
                return false;
            }
            if i > 0 && y < nums[i - 1] {
                nums[i + 1] = x;
            }
        }
    }
    true
}

/// 670.Maximum Swap
pub fn maximum_swap(num: i32) -> i32 {
    let mut chars = num.to_string().bytes().collect::<Vec<u8>>();
    let n = chars.len();
    let (mut index1, mut index2) = (n, n);
    let (mut max, mut max_index) = (b'0' - 1, n);
    for (i, &ch) in chars.iter().enumerate().rev() {
        match ch.cmp(&max) {
            Ordering::Greater => {
                max = ch;
                max_index = i;
            }
            Ordering::Less => {
                index1 = max_index;
                index2 = i;
            }
            _ => {}
        }
    }
    if index1 != n && index2 != n {
        chars.swap(index1, index2);
        String::from_utf8(chars).unwrap().parse::<i32>().unwrap()
    } else {
        num
    }
}

/// 672.Bulb Switcher II
pub fn flip_lights(n: i32, presses: i32) -> i32 {
    let mut seen: HashSet<i32> = HashSet::new();
    let bits = vec![0, 1, 2, 3];
    for i in 0..16 {
        let mut press_array: Vec<i32> = vec![0; 4];
        for &j in bits.iter() {
            press_array[j] = (i >> j) & 1;
        }
        let sum: i32 = press_array.iter().sum();
        if sum % 2 == presses % 2 && sum <= presses {
            let mut status = press_array[0] ^ press_array[1] ^ press_array[3];
            if n >= 2 {
                status |= (press_array[0] ^ press_array[1]) << 1;
            }
            if n >= 3 {
                status |= (press_array[0] ^ press_array[2]) << 2;
            }
            if n >= 4 {
                status |= (press_array[0] ^ press_array[1] ^ press_array[3]) << 3;
            }
            seen.insert(status);
        }
    }
    seen.len() as i32
}

/// 682.Baseball Game
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

/// 693.Binary Number with Alternating bits
pub fn has_alternating_bits(n: i32) -> bool {
    let a = n ^ (n >> 1);
    a & (a + 1) == 0
}

/// 724.Find Pivot Index
pub fn pivot_index(nums: Vec<i32>) -> i32 {
    let total: i32 = nums.iter().sum();
    let mut sum: i32 = 0;
    for (i, num) in nums.iter().enumerate() {
        if 2 * sum + num == total {
            return i as i32;
        }
        sum += num;
    }
    -1
}

/// 728.Self Dividing Numbers
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

/// 744.Find Smallest Letter Greater Than Target
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
    letters[low]
}

/// 762.Prime Number of Set Bits in Binary Representation
pub fn count_prime_set_bits(left: i32, right: i32) -> i32 {
    (left..=right).fold(0, |ret, i| ret + (665772 >> i.count_ones() & 1))
}

/// 769.Max Chunks To Make Sorted
pub fn max_chunks_to_sorted(arr: Vec<i32>) -> i32 {
    let (mut res, mut maximum) = (0, 0);
    for (i, v) in arr.iter().enumerate() {
        maximum = max(maximum, *v);
        if maximum == i as i32 {
            res += 1
        }
    }
    res
}

/// 779.K-th Symbol in Grammar
pub fn kth_grammar(_n: i32, mut k: i32) -> i32 {
    k -= 1;
    let mut res = 0;
    while k > 0 {
        k &= k - 1;
        res ^= 1;
    }
    res
}

/// 784.Letter Case Permutation
pub fn letter_case_permutation(s: String) -> Vec<String> {
    let (mut m, mut res) = (0, Vec::<String>::new());
    for ch in s.chars() {
        if ch.is_alphabetic() {
            m += 1;
        }
    }
    for mask in 0..1 << m {
        let (mut t, mut k) = (Vec::<String>::new(), 0);
        for c in s.chars() {
            if c.is_alphabetic() {
                t.push(if mask >> k & 1 != 0 {
                    c.to_uppercase().to_string()
                } else {
                    c.to_lowercase().to_string()
                });
                k += 1;
            } else {
                t.push(c.to_string());
            }
        }
        res.push(t.join(""));
    }
    res
}

/// 793.Preimage Size of Factorial Zeroes Function
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
        (right + 1) as i32
    }

    nx(k + 1) - nx(k)
}

/// 801.Minimum Swaps To Make Sequences Increasing
pub fn min_swap(nums1: Vec<i32>, nums2: Vec<i32>) -> i32 {
    let n = nums1.len();
    let (mut a, mut b) = (0, 1);
    for i in 1..n {
        let (at, bt) = (a, b);
        a = n;
        b = n;
        if nums1[i] > nums1[i - 1] && nums2[i] > nums2[i - 1] {
            a = min(a, at);
            b = min(b, bt + 1);
        }
        if nums1[i] > nums2[i - 1] && nums2[i] > nums1[i - 1] {
            a = min(a, bt);
            b = min(b, at + 1);
        }
    }
    min(a, b) as i32
}

/// 804.Unique Morse Code Words
pub fn unique_morse_representations(words: Vec<String>) -> i32 {
    let morse = vec![
        ".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--",
        "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--",
        "--..",
    ];
    words
        .iter()
        .fold(HashSet::new(), |mut unique, word| {
            let mut s = String::new();
            word.bytes()
                .for_each(|ch| s = format!("{}{}", s, morse[(ch - b'a') as usize]));
            unique.insert(s);
            unique
        })
        .len() as i32
}

/// 806.Number of Lines To Write String
pub fn number_of_lines(widths: Vec<i32>, s: String) -> Vec<i32> {
    let max_width = 100;
    let mut lines = 1;
    let mut width = 0;
    for c in s.chars() {
        let need = widths[(c as u8 - b'a') as usize];
        width += need;
        if width > max_width {
            width = need;
            lines += 1;
        }
    }
    vec![lines, width]
}

/// 811.Subdomain Visit Count
pub fn subdomain_visits(cpdomains: Vec<String>) -> Vec<String> {
    let mut map = HashMap::<&str, usize>::new();
    let mut count = 0;
    for cpdomain in &cpdomains {
        for (i, v) in cpdomain.as_bytes().iter().enumerate() {
            if *v == b' ' {
                count = cpdomain[..i].parse::<usize>().unwrap();
                map.entry(&cpdomain[i + 1..])
                    .and_modify(|x| *x += count)
                    .or_insert(count);
                continue;
            }
            if *v == b'.' {
                map.entry(&cpdomain[i + 1..])
                    .and_modify(|x| *x += count)
                    .or_insert(count);
            }
        }
    }
    map.iter().map(|(s, n)| format!("{} {}", n, s)).collect()
}

/// 817.Linked List Components
pub fn num_components(head: Option<Box<ListNode>>, nums: Vec<i32>) -> i32 {
    let set = nums.iter().fold(HashSet::new(), |mut set, num| {
        set.insert(num);
        set
    });
    let mut in_set = false;
    let mut res = 0;
    let mut head = &head;
    while let Some(node) = head {
        if set.contains(&node.val) {
            if !in_set {
                in_set = true;
                res += 1;
            }
        } else {
            in_set = false;
        }
        head = &node.next;
    }
    res
}

/// 821.Shortest Distance to a Character
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

/// 828.Count Unique Characters of All Substrings of a Given String
pub fn unique_letter_string(s: String) -> i32 {
    let mut index: HashMap<char, Vec<i32>> = HashMap::new();
    for i in 0..s.len() {
        let c = s.as_bytes()[i] as char;
        index.entry(c).or_insert_with(|| vec![-1]);
        index.get_mut(&c).unwrap().push(i as i32);
    }
    let mut res = 0;
    for pair in index.iter() {
        let mut arr = pair.1.clone();
        arr.push(s.len() as i32);
        for i in 1..arr.len() - 1 {
            res += (arr[i] - arr[i - 1]) * (arr[i + 1] - arr[i]);
        }
    }
    res
}

/// 856.Score of Parentheses
pub fn score_of_parentheses(s: String) -> i32 {
    let chars = s.as_bytes();
    chars
        .iter()
        .enumerate()
        .fold((0, 0), |(mut l, ret), (i, &ch)| {
            l += if ch == b'(' { 1 } else { -1 };
            (
                l,
                if ch == b')' && chars[i - 1] == b'(' {
                    ret + (1 << l)
                } else {
                    ret
                },
            )
        })
        .1
}

/// 862.Shortest Subarray with Sum at Least K
pub fn shortest_subarray(nums: Vec<i32>, k: i32) -> i32 {
    let (mut ret, mut pre_sum, mut queue) = (i64::MAX, 0, VecDeque::new());
    queue.push_back((0, -1));
    for (i, &num) in nums.iter().enumerate() {
        pre_sum += num as i64;
        while !queue.is_empty() && pre_sum <= queue[queue.len() - 1].0 {
            queue.pop_back();
        }
        while !queue.is_empty() && pre_sum - queue[0].0 >= k as i64 {
            ret = ret.min(i as i64 - queue.pop_front().unwrap().1 as i64);
        }
        queue.push_back((pre_sum, i as i32));
    }
    if ret == i64::MAX {
        -1
    } else {
        ret as i32
    }
}

///  883.Projection Area of 3D Shapes
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

/// 886.Possible Bipartition
pub fn possible_bipartition(n: i32, dislikes: Vec<Vec<i32>>) -> bool {
    let mut color = vec![0; (n + 1) as usize];
    let mut group = vec![vec![]; (n + 1) as usize];
    for p in dislikes.iter() {
        group[p[0] as usize].push(p[1] as usize);
        group[p[1] as usize].push(p[0] as usize);
    }
    fn dfs(curr: usize, now_color: i32, color: &mut Vec<i32>, group: &Vec<Vec<usize>>) -> bool {
        color[curr] = now_color;
        for &next in group[curr].iter() {
            if color[next] != 0 && color[next] == color[curr] {
                return false;
            }
            if color[next] == 0 && !dfs(next, 3 ^ now_color, color, group) {
                return false;
            }
        }
        true
    }

    for i in 1..=n as usize {
        if color[i] == 0 && !dfs(i, 1, &mut color, &group) {
            return false;
        }
    }
    true
}

/// 905.Sort Array By Parity
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

/// 907.Sum of Subarray Minimums
pub fn sum_subarray_mins(arr: Vec<i32>) -> i32 {
    let mod_number = 1000000007;
    let n = arr.len();
    let (mut mono_stack, mut dp, mut res) = (Vec::<usize>::new(), vec![0; n], 0);
    for (i, &v) in arr.iter().enumerate() {
        while !mono_stack.is_empty() && arr[mono_stack[mono_stack.len() - 1]] > v {
            mono_stack.pop();
        }
        let k = if mono_stack.is_empty() {
            i + 1
        } else {
            i - mono_stack[mono_stack.len() - 1]
        };
        dp[i] = if mono_stack.is_empty() {
            k * (v as usize)
        } else {
            k * (v as usize) + (dp[i - k])
        };
        res = (res + dp[i]) % mod_number;
        mono_stack.push(i);
    }
    res as i32
}

/// 915.Partition Array into Disjoint Intervals
pub fn partition_disjoint(nums: Vec<i32>) -> i32 {
    let (mut curr_max, mut left_max) = (nums[0], nums[0]);
    let mut res = 0;
    for (i, &num) in nums
        .iter()
        .enumerate()
        .filter(|(i, _)| i > &0 && i < &(nums.len() - 1))
    {
        curr_max = max(curr_max, num);
        if nums[i] < left_max {
            left_max = curr_max;
            res = i as i32
        }
    }
    res + 1
}

/// 921.Minimum Add to Make Parentheses Valid
pub fn min_add_to_make_valid(s: String) -> i32 {
    let mut res = 0;
    let mut left_count = 0;
    for ch in s.chars() {
        if ch == '(' {
            left_count += 1;
        } else if left_count > 0 {
            left_count -= 1;
        } else {
            res += 1;
        }
    }
    res += left_count;
    res
}

/// 927.Three Equal Parts
pub fn three_equal_parts(arr: Vec<i32>) -> Vec<i32> {
    let sum: i32 = arr.iter().sum();
    if sum % 3 != 0 {
        return vec![-1, -1];
    }
    if sum == 0 {
        return vec![0, 2];
    }
    let partial = sum / 3;
    let (mut first, mut second, mut third, mut curr) = (0, 0, 0, 0);
    for (i, v) in arr.iter().enumerate() {
        if *v == 1 {
            if curr == 0 {
                first = i;
            } else if curr == partial {
                second = i;
            } else if curr == 2 * partial {
                third = i;
            }
            curr += 1;
        }
    }
    let n = arr.len() - third;
    if first + n <= second && second + n <= third {
        let mut i = 0;
        while third + i < arr.len() {
            if (arr[first + i] != arr[second + i]) || (arr[first + i] != arr[third + i]) {
                return vec![-1, -1];
            }
            i += 1;
        }
        return vec![(first + n - 1) as i32, (second + n) as i32];
    }
    vec![-1, -1]
}

/// 934.Shortest Bridge
pub fn shortest_bridge(mut grid: Vec<Vec<i32>>) -> i32 {
    let n = grid.len();
    let dirs = vec![vec![-1, 0], vec![1, 0], vec![0, 1], vec![0, -1]];
    let mut island = Vec::<(i32, i32)>::new();
    let mut queue = VecDeque::<(i32, i32)>::new();
    let dir_loop: Vec<usize> = vec![0, 1, 2, 3];
    for i in 0..n {
        for j in 0..n {
            if grid[i][j] == 1 {
                queue.push_back((i as i32, j as i32));
                grid[i][j] = -1;
                while !queue.is_empty() {
                    let cell = queue.pop_front().unwrap();
                    island.push((cell.0, cell.1));
                    for &k in dir_loop.iter() {
                        let nx = cell.0 + dirs[k][0];
                        let ny = cell.1 + dirs[k][1];
                        if nx >= 0
                            && ny >= 0
                            && nx < n as i32
                            && ny < n as i32
                            && grid[nx as usize][ny as usize] == 1
                        {
                            queue.push_back((nx, ny));
                            grid[nx as usize][ny as usize] = -1;
                        }
                    }
                }
                island.iter().for_each(|&x| queue.push_back(x));
                let mut step = 0;
                while !queue.is_empty() {
                    for _k in 0..queue.len() {
                        let cell = queue.pop_front().unwrap();
                        for &d in dir_loop.iter() {
                            let nx = cell.0 + dirs[d][0];
                            let ny = cell.1 + dirs[d][1];
                            if nx >= 0 && ny >= 0 && nx < n as i32 && ny < n as i32 {
                                if grid[nx as usize][ny as usize] == 0 {
                                    queue.push_back((nx, ny));
                                    grid[nx as usize][ny as usize] = -1;
                                } else if grid[nx as usize][ny as usize] == 1 {
                                    return step;
                                }
                            }
                        }
                    }
                    step += 1;
                }
            }
        }
    }
    0
}

/// 940.Distinct Subsequences II
pub fn distinct_subseq_ii(s: String) -> i32 {
    let mod_number = 1000000007;
    let mut alphas = vec![0; 26];
    let mut res = 0;
    s.as_bytes().iter().for_each(|x| {
        let index = (x - b'a') as usize;
        let prev = alphas[index];
        alphas[index] = (res + 1) % mod_number;
        res = ((res + alphas[index] - prev) % mod_number + mod_number) % mod_number;
    });
    res
}

/// 942.DI String Match
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

/// 944.Delete Columns to Make Sorted
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

/// 946.Validate Stack Sequences
pub fn validate_stack_sequences(pushed: Vec<i32>, popped: Vec<i32>) -> bool {
    let mut stack: VecDeque<i32> = VecDeque::new();
    let mut j: usize = 0;
    for (_, &v) in pushed.iter().enumerate() {
        stack.push_back(v);
        while !stack.is_empty() && stack.back().cloned().unwrap() == popped[j] {
            stack.pop_back();
            j += 1;
        }
    }
    stack.is_empty()
}

/// 998.Maximum Binary Tree II
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
    curr.as_ref().unwrap().borrow_mut().right = Some(parent);
    root
}

/// 1017. Convert to Base -2
pub fn base_neg2(n: i32) -> String {
    let mut val = 0x55555555 ^ (0x55555555 - n);
    if val == 0 {
        return "0".to_string();
    }
    let mut res: Vec<String> = Vec::new();
    while val != 0 {
        res.push((val & 1).to_string());
        val >>= 1;
    }
    res.reverse();
    res.join("")
}

/// 1052. Grumpy Bookstore Owner
pub fn max_satisfied(customers: Vec<i32>, grumpy: Vec<i32>, minutes: i32) -> i32 {
    let mut total = 0;
    let n = customers.len();
    for i in 0..n {
        if grumpy[i] == 0 {
            total += customers[i];
        }
    }
    let mut increase = 0;
    for i in 0..minutes as usize {
        increase = increase + customers[i] * grumpy[i];
    }

    let mut max_increase = increase;
    for i in minutes as usize..n {
        increase = increase + customers[i] * grumpy[i]
            - customers[i - minutes as usize] * grumpy[i - minutes as usize];
        max_increase = max_increase.max(increase);
    }

    total + max_increase
}

/// 1403.Minimum Subsequence in Non-Increasing Order
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

/// 1408.String Matching in an Array
pub fn string_matching(words: Vec<String>) -> Vec<String> {
    let mut res: Vec<String> = vec![];
    for (i, word1) in words.iter().enumerate() {
        for (j, word2) in words.iter().enumerate() {
            if i != j && word2.contains(word1.as_str()) {
                res.push(word1.clone());
                break;
            }
        }
    }
    res
}

/// 1441.Build an Array With Stack Operations
pub fn build_array(target: Vec<i32>, _n: i32) -> Vec<String> {
    let mut res = Vec::<String>::new();
    let mut prev = 0;
    target.iter().for_each(|&x| {
        for _i in 0..(x - prev - 1) {
            res.push(String::from("Push"));
            res.push(String::from("Pop"));
        }
        res.push(String::from("Push"));
        prev = x;
    });
    res
}

/// 1450.Number of Students Doing Homework at a Given Time
pub fn busy_student(start_time: Vec<i32>, end_time: Vec<i32>, query_time: i32) -> i32 {
    let mut res = 0;
    for i in 0..start_time.len() {
        if start_time[i] <= query_time && query_time <= end_time[i] {
            res += 1;
        }
    }
    res
}

/// 1455.Check If a Word Occurs As a Prefix of Any Word in a Sentence
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

/// 1460.Make Two Arrays Equal by Reversing Sub-arrays
pub fn can_be_equal(mut target: Vec<i32>, mut arr: Vec<i32>) -> bool {
    target.sort();
    arr.sort();
    target == arr
}

/// 1464.Maximum Product of Two Elements in an Array
pub fn max_product(nums: Vec<i32>) -> i32 {
    let mut a = nums[0];
    let mut b = nums[1];
    if a < b {
        swap(&mut a, &mut b);
    }
    for &num in nums.iter().skip(2) {
        if a < num {
            b = a;
            a = num;
        } else if num > b {
            b = num;
        }
    }
    (a - 1) * (b - 1)
}

/// 1470.Shuffle the Array
pub fn shuffle(nums: Vec<i32>, n: i32) -> Vec<i32> {
    let mut res: Vec<i32> = vec![0; (n * 2) as usize];
    let n = n as usize;
    for i in 0..n {
        res[2 * i] = nums[i];
        res[2 * i + 1] = nums[n + i];
    }
    res
}

/// 1475.Final Prices With a Special Discount in a Shop
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

/// 1582.Special Positions in a Binary Matrix
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
    for num in mat[0].iter().copied() {
        if num == 1 {
            sum += 1;
        }
    }
    sum
}

/// 1608.Special Array With X Elements Greater Than or Equal X
pub fn special_array(mut nums: Vec<i32>) -> i32 {
    let n = nums.len();
    nums.sort_by(|a, b| b.cmp(a));
    for i in 1..=n {
        if nums[i - 1] >= i as i32 && (i == n || nums[i] < i as i32) {
            return i as i32;
        }
    }
    -1
}

/// 1619.Mean of Array After Removing Some Elements
pub fn trim_mean(mut arr: Vec<i32>) -> f64 {
    let n = arr.len();
    arr.sort();
    arr[n / 20..(19 * n / 20)].iter().sum::<i32>() as f64 / (n as f64 * 0.9)
}

/// 1624.Largest Substring Between Two Equal Characters
pub fn max_length_between_equal_characters(s: String) -> i32 {
    let mut map: HashMap<char, usize> = HashMap::new();
    let mut res: i32 = -1;
    for (i, ch) in s.chars().enumerate() {
        if let Vacant(e) = map.entry(ch) {
            e.insert(i);
        } else {
            res = max(res, (i - map[&ch] - 1) as i32);
        }
    }
    res
}

/// 1636.Sort Array by Increasing Frequency
pub fn frequency_sort(mut nums: Vec<i32>) -> Vec<i32> {
    let mut count = HashMap::<i32, i32>::new();
    nums.iter()
        .for_each(|num| *count.entry(*num).or_insert(0) += 1);
    nums.sort_by(|x, y| count[x].cmp(&count[y]).then(y.cmp(x)));
    nums
}

/// 1658.Minimum Operations to Reduce X to Zero
pub fn min_operations(nums: Vec<i32>, x: i32) -> i32 {
    let n = nums.len();
    let sum = nums.iter().sum::<i32>();
    if sum < x {
        return -1;
    }
    let (mut left, mut right, mut left_sum, mut right_sum, mut res) = (-1, 0, 0, sum, n as i32 + 1);
    while left < n as i32 {
        if left != -1 {
            left_sum += nums[left as usize]
        }
        while right < n && left_sum + right_sum > x {
            right_sum -= nums[right];
            right += 1;
        }
        if left_sum + right_sum == x {
            res = min(res, (left + 1) + (n - right) as i32);
        }
        left += 1;
    }
    if res > n as i32 {
        -1
    } else {
        res
    }
}

/// 1672.Richest Customer Wealth
pub fn maximum_wealth(accounts: Vec<Vec<i32>>) -> i32 {
    accounts.iter().map(|x| x.iter().sum()).max().unwrap()
}

/// 1673.Find the Most Competitive Subsequence
pub fn most_competitive(nums: Vec<i32>, k: i32) -> Vec<i32> {
    let mut res: Vec<i32> = vec![];
    let n = nums.len();
    for i in 0..n {
        while res.len() > 0 && (n - i + res.len()) as i32 > k && *res.last().unwrap() > nums[i] {
            res.pop();
        }
        res.push(nums[i]);
    }
    res.truncate(k as usize);
    res
}

/// 1694.Reformat Phone Number
pub fn reformat_number(number: String) -> String {
    let mut digits = String::new();
    for ch in number.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
        }
    }
    let mut n = digits.len();
    let mut res = String::new();
    let mut pt = 0;
    while n > 0 {
        if n > 4 {
            res.push_str(&digits[pt..pt + 3]);
            res.push('-');
            pt += 3;
            n -= 3;
        } else {
            if n == 4 {
                res.push_str(&digits[pt..pt + 2]);
                res.push('-');
                res.push_str(&digits[pt + 2..pt + 4]);
            } else {
                res.push_str(&digits[pt..pt + n]);
            }
            break;
        }
    }
    res.to_string()
}

/// 1700.Number of Students Unable to Eat Lunch
pub fn count_students(students: Vec<i32>, sandwiches: Vec<i32>) -> i32 {
    let mut square: i32 = students.iter().sum();
    let mut circular = students.len() as i32 - square;
    for sandwich in sandwiches {
        if sandwich == 1 && square > 0 {
            square -= 1;
        } else if sandwich == 0 && circular > 0 {
            circular -= 1;
        } else {
            break;
        }
    }
    square + circular
}

/// 1702.Maximum Binary String After Change
pub fn maximum_binary_string(binary: String) -> String {
    let n = binary.len();
    let mut res: Vec<char> = binary.chars().collect();
    let (mut zero_index, mut zero_count) = (n, 0);
    for i in 0..n {
        if res[i] == '0' {
            if (zero_index == n) {
                zero_index = i;
            }
            zero_count += 1;
        }
        res[i] = '1';
    }
    if zero_count == 0 {
        return binary;
    }
    res[zero_index + zero_count - 1] = '0';
    res.iter().collect()
}

/// 1750.Minimum Length of String After Deleting Similar Ends
pub fn minimum_length(s: String) -> i32 {
    let (mut left, mut right, s) = (0, s.len() - 1, s.as_bytes());
    while left < right && s[left] == s[right] {
        let c = s[left];
        while left <= right && s[left] == c {
            left += 1;
        }
        while left <= right && s[right] == c {
            right -= 1;
        }
    }
    (right + 1 - left) as i32
}

/// 1768.Merge Strings Alternately
pub fn merge_alternately(word1: String, word2: String) -> String {
    let (mut word1_iter, mut word2_iter, mut res) = (
        word1.chars().peekable(),
        word2.chars().peekable(),
        String::from(""),
    );
    while word1_iter.peek().is_some() || word2_iter.peek().is_some() {
        if let Some(ch) = word1_iter.next() {
            res.push(ch);
        }
        if let Some(ch) = word2_iter.next() {
            res.push(ch);
        }
    }
    res
}

/// 1773.Count Items Matching a Rule
pub fn count_matches(items: Vec<Vec<String>>, rule_key: String, rule_value: String) -> i32 {
    let mut rules = HashMap::<String, usize>::new();
    rules.insert(String::from("type"), 0);
    rules.insert(String::from("color"), 1);
    rules.insert(String::from("name"), 2);
    let index = *rules.get(&rule_key).unwrap();
    let mut res = 0;
    for item in items {
        if item[index] == rule_value {
            res += 1;
        }
    }
    res
}

/// 1779.Find Nearest Point That Has the Same X or Y Coordinate
pub fn nearest_valid_point(x: i32, y: i32, points: Vec<Vec<i32>>) -> i32 {
    let (mut min, mut res) = (i32::MAX, -1);
    for (i, p) in points.iter().enumerate() {
        if p[0] == x || p[1] == y {
            let distance = (p[0] - x).abs() + (p[1] - y).abs();
            if distance < min {
                min = distance;
                res = i as i32;
            }
        }
    }
    res
}

/// 1784.Check if Binary String Has at Most One Segment of Ones
pub fn check_ones_segment(s: String) -> bool {
    !s.contains("01")
}

/// 1790.Check if One String Swap Can Make Strings Equal
pub fn are_almost_equal(s1: String, s2: String) -> bool {
    let mut diff: Vec<usize> = vec![];
    let chars1 = s1.chars().collect::<Vec<char>>();
    let chars2 = s2.chars().collect::<Vec<char>>();
    for i in 0..s1.len() {
        if chars1[i] != chars2[i] {
            if diff.len() > 2 {
                return false;
            }
            diff.push(i);
        }
    }
    let n = diff.len();
    if n == 0 {
        return true;
    }

    if n != 2 {
        return false;
    }
    chars1[diff[0]] == chars2[diff[1]] && chars1[diff[1]] == chars2[diff[0]]
}

/// 1800.Maximum Ascending Subarray Sum
pub fn max_ascending_sum(nums: Vec<i32>) -> i32 {
    let (mut res, mut i) = (0, 0);
    let n = nums.len();
    while i < n {
        let mut curr = nums[i];
        i += 1;
        while i < n && nums[i] > nums[i - 1] {
            curr += nums[i];
            i += 1
        }
        res = max(res, curr);
    }
    res
}

/// 1802.Maximum Value at a Given Index in a Bounded Array
pub fn max_value(n: i32, index: i32, max_sum: i32) -> i32 {
    let (mut left, mut right) = (index as f64, (n - index - 1) as f64);
    let max_sum = max_sum as f64;
    if left > right {
        swap(&mut left, &mut right);
    }
    let mut upper = ((left + 1.0) * (left + 1.0) - 3.0 * (left + 1.0)) / 2.0
        + left
        + 1.0
        + (left + 1.0)
        + ((left + 1.0) * (left + 1.0) - 3.0 * (left + 1.0)) / 2.0
        + right
        + 1.0;
    if upper >= max_sum {
        let (a, b, c) = (1.0, -2.0, left + right + 2.0 - max_sum);
        return ((-b + (b * b - 4.0 * a * c).sqrt()) / (2.0 * a)).floor() as i32;
    }
    upper = (2.0 * (right + 1.0) - left - 1.0) * left / 2.0
        + (right + 1.0)
        + ((right + 1.0) * (right + 1.0) - 3.0 * (right + 1.0)) / 2.0
        + right
        + 1.0;

    if upper >= max_sum {
        let (a, b, c) = (
            1.0 / 2.0,
            left + 1.0 - 3.0 / 2.0,
            right + 1.0 + (-left - 1.0) * left / 2.0 - max_sum,
        );
        ((-b + (b * b - 4.0 * a * c).sqrt()) / (2.0 * a)).floor() as i32
    } else {
        let (a, b) = (
            left + right + 1.0,
            (-left * left - left - right * right - right) / 2.0 - max_sum,
        );
        (-b / a).floor() as i32
    }
}

/// 1822.Sign of the Product of an Array
pub fn array_sign(nums: Vec<i32>) -> i32 {
    let mut sign = 1;
    for num in nums {
        if num == 0 {
            return 0;
        }
        if num < 0 {
            sign = -sign;
        }
    }
    sign
}

/// 1823.Find the Winner of the Circular Game
pub fn find_the_winner(n: i32, k: i32) -> i32 {
    let mut winner = 1;
    for i in 2..=n {
        winner = (winner + k - 1) % i + 1;
    }
    winner
}

/// 1991.Find the Middle Index in Array
pub fn find_middle_index(nums: Vec<i32>) -> i32 {
    let total: i32 = nums.iter().sum();
    let mut sum: i32 = 0;
    for (i, num) in nums.iter().enumerate() {
        if 2 * sum + num == total {
            return i as i32;
        }
        sum += num;
    }
    -1
}

/// 1997.First Day Where You Have Been in All the Rooms
pub fn first_day_been_in_all_rooms(next_visit: Vec<i32>) -> i32 {
    const MOD: i32 = 1_000_000_007;
    let n = next_visit.len();
    let mut dp: Vec<i32> = vec![0; n];
    dp[0] = 2;
    for i in 1..n {
        let to = next_visit[i] as usize;
        dp[i] = 2 + dp[i - 1];
        if to != 0 {
            dp[i] = (dp[i] - dp[to - 1] + MOD) % MOD;
        }
        dp[i] = dp[i] + dp[i - 1] % MOD;
    }
    dp[n - 2]
}

/// 2011.Final Value of Variable After Performing Operations
pub fn final_value_after_operations(operations: Vec<String>) -> i32 {
    operations.iter().fold(0, |acc, op| {
        acc + if op.as_bytes()[1] as char == '+' {
            1
        } else {
            -1
        }
    })
}

/// 2027.Minimum Moves to Convert String
pub fn minimum_moves(s: String) -> i32 {
    let (mut res, mut count) = (0, -1);
    for (i, ch) in s.chars().enumerate() {
        if ch == 'X' && i as i32 > count {
            res += 1;
            count = i as i32 + 2;
        }
    }
    res
}

/// 2032.Two Out of Three
pub fn two_out_of_three(nums1: Vec<i32>, nums2: Vec<i32>, nums3: Vec<i32>) -> Vec<i32> {
    let mut map = HashMap::<i32, i32>::new();
    for num in nums1 {
        map.insert(num, 1);
    }
    for num in nums2 {
        *map.entry(num).or_insert(0) |= 2;
    }
    for num in nums3 {
        *map.entry(num).or_insert(0) |= 4;
    }
    let mut res = Vec::<i32>::new();
    for (k, v) in map.iter() {
        if (v & (v - 1)) != 0 {
            res.push(*k);
        }
    }
    res
}

/// 2037.Minimum Number of Moves to Seat Everyone
pub fn min_moves_to_seat(mut seats: Vec<i32>, mut students: Vec<i32>) -> i32 {
    seats.sort();
    students.sort();
    seats
        .iter()
        .zip(students.iter())
        .fold(0, |res, (i, j)| res + (i - j).abs())
}

/// 2042.Check if Numbers Are Ascending in a Sentence
pub fn are_number_ascending(s: String) -> bool {
    let (mut prev, mut pos, n) = (0, 0, s.len());
    let chars = s.as_bytes();
    while pos < n {
        if chars[pos].is_ascii_digit() {
            let mut curr = 0;
            while pos < n && chars[pos].is_ascii_digit() {
                curr = curr * 10 + chars[pos] - b'0';
                pos += 1;
            }
            if curr <= prev {
                return false;
            }
            prev = curr;
        } else {
            pos += 1;
        }
    }
    true
}

/// 2180.Count Integers With Even Digit Sum
pub fn count_even(num: i32) -> i32 {
    let (mut y, x) = (num / 10, num % 10);
    let (mut res, mut y_sum) = (y * 5, 0);
    while y > 0 {
        y_sum += y % 10;
        y /= 10;
    }
    if y_sum % 2 == 0 {
        res += x / 2 + 1;
    } else {
        res += (x + 1) / 2;
    }
    res - 1
}

/// 2185.Counting Words With a Given Prefix
pub fn prefix_count(words: Vec<String>, pref: String) -> i32 {
    words
        .into_iter()
        .filter(|word| word.starts_with(&pref))
        .count() as i32
}

/// 2244.Minimum Rounds to Complete All Tasks
pub fn minimum_rounds(tasks: Vec<i32>) -> i32 {
    let mut counts = HashMap::<i32, i32>::new();
    for &task in tasks.iter() {
        *counts.entry(task).or_insert(0) += 1;
    }
    let mut res = 0;
    for (_, count) in counts {
        if count == 1 {
            return -1;
        } else if count % 3 == 0 {
            res += count / 3;
        } else {
            res += count / 3 + 1;
        }
    }
    res
}

/// 2351.First Letter to Appear Twice
pub fn repeated_character(s: String) -> char {
    let mut seen = 0;
    for ch in s.chars() {
        let x = ch as u8 - b'a';
        if (seen & (1 << x)) > 0 {
            return ch;
        }
        seen |= 1 << x;
    }
    ' '
}

/// 2391.Minimum Amount of Time to Collect Garbage
pub fn garbage_collection(garbage: Vec<String>, travel: Vec<i32>) -> i32 {
    let mut distance = HashMap::<String, i32>::new();
    let mut res = 0;
    let mut current_distance = 0;
    for i in 0..garbage.len() {
        res += garbage[i].len() as i32;
        if (i > 0) {
            current_distance += travel[i - 1];
        }

        for c in garbage[i].chars() {
            distance.insert(c.to_string(), current_distance);
        }
    }
    for (k, v) in distance.iter() {
        res += *v;
    }
    res
}

/// 2580.Count Ways to Group Overlapping Ranges
pub fn count_ways(ranges: Vec<Vec<i32>>) -> i32 {
    let mut ranges = ranges;
    const MOD: i64 = 1_000_000_007;
    ranges.sort_by(|a, b| a[0].cmp(&b[0]));

    let n = ranges.len();
    let mut res: i64 = 1;
    let mut i = 0;
    while i < n {
        let (mut r, mut j) = (ranges[i][1], i + 1);
        while j < n && ranges[j][0] <= r {
            r = r.max(ranges[j][1]);
            j += 1;
        }

        res = (res * 2) % MOD;
        i = j;
    }
    res as i32
}

/// 2903.Find Indices With Index and Value Difference I
pub fn find_indices(nums: Vec<i32>, index_difference: i32, value_difference: i32) ->  Vec<i32> {
    let (mut min_index, mut max_index) = (0, 0);
    for j in (index_difference as usize)..nums.len() {
        let i = j - index_difference as usize;
        if nums[i] < nums[min_index] {
            min_index = i;
        }
        if nums[j] - nums[min_index] >= value_difference {
            return vec![min_index as i32, j as i32];
        }

        if nums[i] > nums[max_index] {
            max_index = i;
        }

        if nums[max_index] - nums[j] >= value_difference {
            return vec![max_index as i32, j as i32];
        }
    }
    vec![-1, -1]
}

/// 2908.Minimum Sum of Mountain Triplets I
pub fn minimum_sum(nums: Vec<i32>) -> i32 {
    let n = nums.len();
    let (mut res, mut mini) = (1000, 1000);
    let (mut left, mut right) = (vec![0; n], nums[n - 1]);
    for i in 1..n {
        mini = nums[i - 1].min(mini);
        left[i] = mini;
    }
    for i in (1..n - 1).rev() {
        if left[i] < nums[i] && nums[i] > right {
            res = res.min(left[i] + nums[i] + right);
        }
        right = right.min(nums[i]);
    }
    if res < 1000 {
        res
    } else {
        -1
    }
}

/// 2952.Minimum Number of Coins to be Added
pub fn minimum_added_coins(coins: Vec<i32>, target: i32) -> i32 {
    let mut coins = coins;
    coins.sort_by(|a, b| a.cmp(b));
    let mut res = 0;
    let mut x = 1;
    let n = coins.len();
    let mut index = 0;
    while x <= target {
        if index < n && coins[index] <= x {
            x += coins[index];
            index += 1;
        } else {
            x *= 2;
            res += 1;
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use crate::leetcode::solution::*;
    use crate::leetcode::util::*;

    fn test_string_vec_equal_base(mut expected: Vec<String>, mut actual: Vec<String>) {
        expected.sort();
        actual.sort();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_two_sum() {
        assert_eq!(vec![0, 1], two_sum(vec![2, 7, 11, 15], 9));
        assert_eq!(vec![1, 2], two_sum(vec![3, 2, 4], 6));
        assert_eq!(vec![0, 1], two_sum(vec![3, 3], 6));
    }

    #[test]
    fn test_add_two_numbers() {
        assert_eq!(
            add_two_numbers(
                generate_linked_list_node(vec![2, 4, 3]),
                generate_linked_list_node(vec![5, 6, 4]),
            ),
            generate_linked_list_node(vec![7, 0, 8])
        );
        assert_eq!(
            add_two_numbers(
                generate_linked_list_node(vec![0]),
                generate_linked_list_node(vec![0]),
            ),
            generate_linked_list_node(vec![0])
        );
        assert_eq!(
            add_two_numbers(
                generate_linked_list_node(vec![9, 9, 9, 9, 9, 9, 9]),
                generate_linked_list_node(vec![9, 9, 9, 9]),
            ),
            generate_linked_list_node(vec![8, 9, 9, 9, 0, 0, 0, 1])
        );
    }

    #[test]
    fn test_length_of_longest_substring() {
        assert_eq!(length_of_longest_substring(String::from("abcabcbb")), 3);
        assert_eq!(length_of_longest_substring(String::from("bbbbb")), 1);
        assert_eq!(length_of_longest_substring(String::from("pwwkew")), 3);
    }

    #[test]
    fn test_merge_two_sorted_lists() {
        assert_eq!(
            merge_two_sorted_lists(
                generate_linked_list_node(vec![1, 2, 4]),
                generate_linked_list_node(vec![1, 3, 4]),
            ),
            generate_linked_list_node(vec![1, 1, 2, 3, 4, 4])
        );
        assert_eq!(merge_two_sorted_lists(None, None), None);
        assert_eq!(
            merge_two_sorted_lists(None, generate_linked_list_node(vec![0])),
            generate_linked_list_node(vec![0])
        );
    }

    #[test]
    fn test_merge_k_sorted_lists() {
        let lists = vec![
            generate_linked_list_node(vec![1, 4, 5]),
            generate_linked_list_node(vec![1, 3, 4]),
            generate_linked_list_node(vec![2, 6]),
        ];
        assert_eq!(
            merge_k_lists(lists),
            generate_linked_list_node(vec![1, 1, 2, 3, 4, 4, 5, 6])
        );
    }

    #[test]
    fn test_remove_duplicates() {
        assert_eq!(remove_duplicates(&mut vec![1, 1, 2]), 2);
        assert_eq!(
            remove_duplicates(&mut vec![0, 0, 1, 1, 1, 2, 2, 3, 3, 4]),
            5
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
    fn test_remove_nth_from_end() {
        assert_eq!(
            remove_nth_from_end(generate_linked_list_node(vec![1, 2, 3, 4, 5]), 2),
            generate_linked_list_node(vec![1, 2, 3, 5])
        );
        assert_eq!(
            remove_nth_from_end(generate_linked_list_node(vec![1]), 1),
            None
        );
        assert_eq!(
            remove_nth_from_end(generate_linked_list_node(vec![1, 2]), 1),
            generate_linked_list_node(vec![1])
        );
    }

    #[test]
    fn test_remove_element() {
        assert_eq!(remove_element(&mut vec![3, 2, 2, 3], 3), 2);
        assert_eq!(remove_element(&mut vec![0, 1, 2, 2, 3, 0, 4, 2], 2), 5);
    }

    #[test]
    fn test_search() {
        assert_eq!(search(vec![4, 5, 6, 7, 0, 1, 2], 0), 4);
        assert_eq!(search(vec![4, 5, 6, 7, 0, 1, 2], 3), -1);
        assert_eq!(search(vec![1], 0), -1);
    }

    #[test]
    fn test_search_insert() {
        assert_eq!(search_insert(vec![1, 3, 5, 6], 0), 0);
        assert_eq!(search_insert(vec![1, 3, 5, 6], 5), 2);
        assert_eq!(search_insert(vec![1, 3, 5, 6], 2), 1);
        assert_eq!(search_insert(vec![1, 3, 5, 6], 7), 4);
    }

    macro_rules! combination_sum_tests {
        ($($name:ident: $value:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    let (candidates, target, mut expected) = $value;
                    let mut actual = combination_sum(&candidates, target);
                    actual.sort_by(|a, b| a.iter().collect::<Vec<_>>().cmp(&b.iter().collect::<Vec<_>>()));
                    expected.sort_by(|a: &Vec<i32>, b: &Vec<i32>| a.iter().collect::<Vec<_>>().cmp(&b.iter().collect::<Vec<_>>()));
                    assert_eq!(actual, expected);
                }
            )*
        };
    }
    combination_sum_tests! {
        test_combination_sum_case0: (vec![2, 3, 6, 7], 7, vec![vec![2, 2, 3], vec![7]]),
        test_combination_sum_case1: (vec![2, 3, 5], 8, vec![vec![2, 2, 2, 2], vec![2, 3, 3], vec![3, 5]]),
        test_combination_sum_case2: (vec![2], 1, vec![]),
    }

    #[test]
    fn test_partition() {
        assert_eq!(
            partition(generate_linked_list_node(vec![1, 4, 3, 2, 5, 2]), 3),
            generate_linked_list_node(vec![1, 2, 2, 4, 3, 5])
        );
        assert_eq!(
            partition(generate_linked_list_node(vec![2, 1]), 2),
            generate_linked_list_node(vec![1, 2])
        );
    }

    #[test]
    fn test_trailing_zeroes() {
        assert_eq!(trailing_zeroes(3), 0);
        assert_eq!(trailing_zeroes(5), 1);
        assert_eq!(trailing_zeroes(0), 0);
    }

    #[test]
    fn test_combination_sum3() {
        assert_eq!(combination_sum3(3, 7), vec![[1, 2, 4]]);
        assert_eq!(
            combination_sum3(3, 9),
            vec![[1, 2, 6], [1, 3, 5], [2, 3, 4]]
        );
        assert_eq!(combination_sum3(4, 1), Vec::<Vec<i32>>::new());
    }

    #[test]
    fn test_is_valid_serialization() {
        assert_eq!(
            is_valid_serialization(String::from("9,3,4,#,#,1,#,#,2,#,6,#,#")),
            true
        );
        assert_eq!(is_valid_serialization(String::from("1,#")), false);
        assert_eq!(is_valid_serialization(String::from("9,#,#,1")), false);
    }

    #[test]
    fn test_count_numbers_with_unique_digits() {
        assert_eq!(count_numbers_with_unique_digits(2), 91);
        assert_eq!(count_numbers_with_unique_digits(0), 1);
    }

    #[test]
    fn test_combination_sum4() {
        assert_eq!(combination_sum4(vec![1, 2, 3], 4), 7);
        assert_eq!(combination_sum4(vec![], 3), 0);
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
    fn test_max_rotate_function() {
        assert_eq!(max_rotate_function(vec![4, 3, 2, 6]), 26);
        assert_eq!(max_rotate_function(vec![100]), 0);
    }

    #[test]
    fn test_min_moves() {
        assert_eq!(min_moves(vec![1, 2, 3]), 3);
        assert_eq!(min_moves(vec![1, 1, 1]), 0);
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
        assert_eq!(convert_to_base7(100), String::from("202"));
        assert_eq!(convert_to_base7(-7), String::from("-10"));
    }

    #[test]
    fn test_change() {
        assert_eq!(change(5, vec![1, 2, 5]), 4);
        assert_eq!(change(3, vec![32]), 0);
        assert_eq!(change(10, vec![10]), 1);
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
                generate_string_vec(vec!["0:start:0", "1:start:2", "1:end:5", "0:end:6"]),
            )
        );
        assert_eq!(
            vec![8],
            exclusive_time(
                1,
                generate_string_vec(vec![
                    "0:start:0",
                    "0:start:2",
                    "0:end:5",
                    "0:start:6",
                    "0:end:6",
                    "0:end:7",
                ]),
            )
        );
        assert_eq!(
            vec![7, 1],
            exclusive_time(
                2,
                generate_string_vec(vec![
                    "0:start:0",
                    "0:start:2",
                    "0:end:5",
                    "1:start:6",
                    "1:end:6",
                    "0:end:7",
                ]),
            )
        );
        assert_eq!(
            vec![8, 1],
            exclusive_time(
                2,
                generate_string_vec(vec![
                    "0:start:0",
                    "0:start:2",
                    "0:end:5",
                    "1:start:7",
                    "1:end:7",
                    "0:end:8",
                ]),
            )
        );
        assert_eq!(
            vec![1],
            exclusive_time(1, generate_string_vec(vec!["0:start:0", "0:end:0"]))
        );
    }

    #[test]
    fn test_find_longest_chain() {
        assert_eq!(
            find_longest_chain(vec![vec![1, 2], vec![2, 3], vec![3, 4]]),
            2
        );
        assert_eq!(
            find_longest_chain(vec![vec![1, 2], vec![7, 8], vec![4, 5]]),
            3
        );
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
    fn test_check_possibility() {
        assert_eq!(true, check_possibility(vec![4, 2, 3]));
        assert_eq!(false, check_possibility(vec![4, 2, 1]));
    }

    #[test]
    fn test_maximum_swap() {
        assert_eq!(7236, maximum_swap(2736));
        assert_eq!(9973, maximum_swap(9973));
    }

    #[test]
    fn test_flip_lights() {
        assert_eq!(2, flip_lights(1, 1));
        assert_eq!(3, flip_lights(2, 1));
        assert_eq!(4, flip_lights(3, 1));
    }

    #[test]
    fn test_cal_points() {
        assert_eq!(
            cal_points(generate_string_vec(vec!["5", "2", "C", "D", "+"])),
            30
        );
        assert_eq!(
            cal_points(generate_string_vec(vec![
                "5", "-2", "4", "C", "D", "9", "+", "+",
            ])),
            27
        );
        assert_eq!(cal_points(generate_string_vec(vec!["1"])), 1);
    }

    #[test]
    fn test_has_alternating_bits() {
        assert!(has_alternating_bits(5));
        assert!(!has_alternating_bits(7));
        assert!(!has_alternating_bits(11));
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
    fn test_max_chunks_to_sorted() {
        assert_eq!(1, max_chunks_to_sorted(vec![4, 3, 2, 1, 0]));
        assert_eq!(4, max_chunks_to_sorted(vec![1, 0, 2, 3, 4]));
    }

    #[test]
    fn test_kth_grammar() {
        assert_eq!(0, kth_grammar(1, 1));
        assert_eq!(0, kth_grammar(2, 1));
        assert_eq!(1, kth_grammar(2, 2));
    }

    macro_rules! letter_case_permutation_tests {
        ($($name:ident: $value:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    let (mut s, mut expected) = $value;
                    let mut actual = letter_case_permutation(s);
                    actual.sort();
                    let mut expected_strings = generate_string_vec(expected);
                    expected_strings.sort();
                    assert_eq!(actual, expected_strings);
                }
            )*
        };
    }

    letter_case_permutation_tests! {
        test_letter_case_permutation_case0: (String::from("a1b2"), vec!["a1b2", "a1B2", "A1b2", "A1B2"]),
        test_letter_case_permutation_case1: (String::from("3z4"), vec!["3z4", "3Z4"]),
    }

    #[test]
    fn test_preimage_size_fzf() {
        assert_eq!(preimage_size_fzf(0), 5);
        assert_eq!(preimage_size_fzf(5), 0);
        assert_eq!(preimage_size_fzf(3), 5);
    }

    #[test]
    fn test_min_swap() {
        assert_eq!(1, min_swap(vec![1, 3, 5, 4], vec![1, 2, 3, 7]));
        assert_eq!(1, min_swap(vec![0, 3, 5, 8, 9], vec![2, 1, 4, 6, 9]));
    }

    #[test]
    fn test_unique_morse_representations() {
        assert_eq!(
            unique_morse_representations(generate_string_vec(vec!["gin", "zen", "gig", "msg"])),
            2
        );
        assert_eq!(
            unique_morse_representations(generate_string_vec(vec!["a"])),
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

    macro_rules! subdomain_visits_tests {
        ($($name:ident: $value:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    let (mut domains, mut expected) = $value;
                    let cpdomains = generate_string_vec(domains);
                    let mut actual = subdomain_visits(cpdomains);
                    actual.sort();
                    let mut expected_strings = generate_string_vec(expected);
                    expected_strings.sort();
                    assert_eq!(actual, expected_strings);
                }
            )*
        };
    }

    subdomain_visits_tests! {
        test_subdomain_visits_case0: (vec!["9001 discuss.leetcode.com"], vec!["9001 discuss.leetcode.com", "9001 com", "9001 leetcode.com"]),
        test_subdomain_visits_case1: (vec![
            "900 google.mail.com",
            "50 yahoo.com",
            "1 intel.mail.com",
            "5 wiki.org"],
            vec![
                "901 mail.com",
                "50 yahoo.com",
                "900 google.mail.com",
                "5 wiki.org",
                "5 org",
                "1 intel.mail.com",
                "951 com"]),
    }

    #[test]
    fn test_num_components() {
        assert_eq!(
            num_components(generate_linked_list_node(vec![0, 1, 2, 3]), vec![0, 1, 3]),
            2
        );
        assert_eq!(
            num_components(
                generate_linked_list_node(vec![0, 1, 2, 3, 4]),
                vec![0, 3, 1, 4],
            ),
            2
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
    fn test_unique_letter_string() {
        assert_eq!(unique_letter_string(String::from("ABC")), 10);
        assert_eq!(unique_letter_string(String::from("ABA")), 8);
        assert_eq!(unique_letter_string(String::from("LEETCODE")), 92);
    }

    #[test]
    fn test_score_of_parentheses() {
        assert_eq!(1, score_of_parentheses(String::from("()")));
        assert_eq!(2, score_of_parentheses(String::from("(())")));
        assert_eq!(2, score_of_parentheses(String::from("()()")));
    }

    #[test]
    fn test_shortest_subarray() {
        assert_eq!(1, shortest_subarray(vec![1], 1));
        assert_eq!(-1, shortest_subarray(vec![1, 2], 4));
        assert_eq!(3, shortest_subarray(vec![2, -1, 2], 3));
    }

    #[test]
    fn test_projection_area() {
        assert_eq!(projection_area(vec![vec![1, 2], vec![3, 4]]), 17);
        assert_eq!(projection_area(vec![vec![2]]), 5);
        assert_eq!(projection_area(vec![vec![1, 0], vec![0, 2]]), 8);
    }

    #[test]
    fn test_possible_bipartiition() {
        assert!(possible_bipartition(
            4,
            vec![vec![1, 2], vec![1, 3], vec![2, 4]],
        ));
        assert!(!possible_bipartition(
            3,
            vec![vec![1, 2], vec![1, 3], vec![2, 3]],
        ));
        assert!(!possible_bipartition(
            5,
            vec![vec![1, 2], vec![2, 3], vec![3, 4], vec![4, 5], vec![1, 5]],
        ));
    }

    #[test]
    fn test_sort_array_by_parity() {
        assert_eq!(sort_array_by_parity(vec![3, 1, 2, 4]), vec![4, 2, 1, 3]);
        assert_eq!(sort_array_by_parity(vec![0]), vec![0]);
    }

    #[test]
    fn test_sum_subarray_mins() {
        assert_eq!(17, sum_subarray_mins(vec![3, 1, 2, 4]));
        assert_eq!(444, sum_subarray_mins(vec![11, 81, 94, 43, 3]));
    }

    #[test]
    fn test_partition_disjoint() {
        assert_eq!(3, partition_disjoint(vec![5, 0, 3, 8, 6]));
        assert_eq!(4, partition_disjoint(vec![1, 1, 1, 0, 6, 12]));
    }

    #[test]
    fn test_generate() {
        assert_eq!(generate(1), vec![vec![1]]);
        assert_eq!(
            generate(5),
            vec![
                vec![1],
                vec![1, 1],
                vec![1, 2, 1],
                vec![1, 3, 3, 1],
                vec![1, 4, 6, 4, 1]
            ]
        );
    }

    #[test]
    fn test_min_add_to_make_valid() {
        assert_eq!(min_add_to_make_valid(String::from("())")), 1);
        assert_eq!(min_add_to_make_valid(String::from("(((")), 3);
    }

    #[test]
    fn test_three_equal_parts() {
        assert_eq!(three_equal_parts(vec![1, 0, 1, 0, 1]), vec![0, 3]);
        assert_eq!(three_equal_parts(vec![1, 1, 0, 1, 1]), vec![-1, -1]);
        assert_eq!(three_equal_parts(vec![1, 1, 0, 0, 1]), vec![0, 2]);
    }

    #[test]
    fn test_shortest_bridge() {
        assert_eq!(1, shortest_bridge(vec![vec![0, 1], vec![1, 0]]));
        assert_eq!(
            2,
            shortest_bridge(vec![vec![0, 1, 0], vec![0, 0, 0], vec![0, 0, 1]])
        );
        assert_eq!(
            1,
            shortest_bridge(vec![
                vec![1, 1, 1, 1, 1],
                vec![1, 0, 0, 0, 1],
                vec![1, 0, 1, 0, 1],
                vec![1, 0, 0, 0, 1],
                vec![1, 1, 1, 1, 1],
            ])
        );
    }

    #[test]
    fn test_distinct_subseq_ii() {
        assert_eq!(distinct_subseq_ii(String::from("abc")), 7);
        assert_eq!(distinct_subseq_ii(String::from("aba")), 6);
        assert_eq!(distinct_subseq_ii(String::from("aaa")), 3);
    }

    #[test]
    fn test_di_string_match() {
        assert_eq!(di_string_match(String::from("IDID")), vec![0, 4, 1, 3, 2]);
        assert_eq!(di_string_match(String::from("III")), vec![0, 1, 2, 3]);
        assert_eq!(di_string_match(String::from("DDI")), vec![3, 2, 0, 1]);
    }

    #[test]
    fn test_min_deletion_size() {
        assert_eq!(
            min_deletion_size(generate_string_vec(vec!["cba", "daf", "ghi"])),
            1
        );
        assert_eq!(min_deletion_size(generate_string_vec(vec!["a", "b"])), 0);
        assert_eq!(
            min_deletion_size(generate_string_vec(vec!["zyx", "wvu", "tsr"])),
            3
        );
    }

    #[test]
    fn test_base_neg2() {
        assert_eq!(base_neg2(2), "110".to_string());
        assert_eq!(base_neg2(3), "111".to_string());
        assert_eq!(base_neg2(4), "100".to_string());
    }

    #[test]
    fn test_max_satisfied() {
        assert_eq!(
            max_satisfied(
                vec![1, 0, 1, 2, 1, 1, 7, 5],
                vec![0, 1, 0, 1, 0, 1, 0, 1],
                3
            ),
            16
        );
        assert_eq!(max_satisfied(vec![1], vec![0], 1), 1);
    }

    #[test]
    fn test_min_subsequence() {
        assert_eq!(min_subsequence(vec![4, 3, 10, 9, 8]), vec![10, 9]);
        assert_eq!(min_subsequence(vec![4, 4, 7, 6, 7]), vec![7, 7, 6]);
        assert_eq!(min_subsequence(vec![6]), vec![6]);
    }

    #[test]
    fn test_string_matching() {
        assert_eq!(
            string_matching(generate_string_vec(vec!["mass", "as", "hero", "superhero"])),
            generate_string_vec(vec!["as", "hero"])
        );
        assert_eq!(
            string_matching(generate_string_vec(vec!["leetcode", "et", "code"])),
            generate_string_vec(vec!["et", "code"])
        );
        let expected: Vec<String> = vec![];
        assert_eq!(
            string_matching(generate_string_vec(vec!["blue", "green", "bu"])),
            expected
        );
    }

    #[test]
    fn test_build_array() {
        assert_eq!(
            build_array(vec![1, 3], 3),
            generate_string_vec(vec!["Push", "Push", "Pop", "Push"])
        );
        assert_eq!(
            build_array(vec![1, 2, 3], 3),
            generate_string_vec(vec!["Push", "Push", "Push"])
        );
        assert_eq!(
            build_array(vec![1, 2], 4),
            generate_string_vec(vec!["Push", "Push"])
        );
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
        assert!(can_be_equal(vec![1, 2, 3, 4], vec![2, 1, 3, 4]));
        assert!(can_be_equal(vec![7], vec![7]));
        assert!(!can_be_equal(vec![3, 7, 9], vec![3, 7, 11]));
    }

    #[test]
    fn test_max_product() {
        assert_eq!(max_product(vec![3, 4, 5, 2]), 12);
        assert_eq!(max_product(vec![1, 5, 4, 5]), 16);
        assert_eq!(max_product(vec![3, 7]), 12);
    }

    #[test]
    fn test_shuffle() {
        assert_eq!(vec![2, 3, 5, 4, 1, 7], shuffle(vec![2, 5, 1, 3, 4, 7], 3));
        assert_eq!(
            vec![1, 4, 2, 3, 3, 2, 4, 1],
            shuffle(vec![1, 2, 3, 4, 4, 3, 2, 1], 4)
        );
        assert_eq!(vec![1, 2, 1, 2], shuffle(vec![1, 1, 2, 2], 2));
    }

    #[test]
    fn test_final_prices() {
        assert_eq!(vec![4, 2, 4, 2, 3], final_prices(vec![8, 4, 6, 2, 3]));
        assert_eq!(vec![1, 2, 3, 4, 5], final_prices(vec![1, 2, 3, 4, 5]));
        assert_eq!(vec![9, 0, 1, 6], final_prices(vec![10, 1, 1, 6]));
    }

    #[test]
    fn test_num_special() {
        assert_eq!(
            num_special(vec![vec![1, 0, 0], vec![0, 0, 1], vec![1, 0, 0]]),
            1
        );
        assert_eq!(
            num_special(vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]),
            3
        );
    }

    #[test]
    fn test_special_array() {
        assert_eq!(special_array(vec![3, 5]), 2);
        assert_eq!(special_array(vec![0, 0]), -1);
        assert_eq!(special_array(vec![0, 4, 3, 0, 4]), 3);
    }

    #[test]
    fn test_trim_mean() {
        assert!(
            (trim_mean(vec![
                1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
            ]) - 2.00000)
                <= 0.00001
        );
        assert!(
            (trim_mean(vec![
                6, 2, 7, 5, 1, 2, 0, 3, 10, 2, 5, 0, 5, 5, 0, 8, 7, 6, 8, 0,
            ]) - 4.00000)
                <= 0.00001
        );
        assert!(
            (trim_mean(vec![
                6, 0, 7, 0, 7, 5, 7, 8, 3, 4, 0, 7, 8, 1, 6, 8, 1, 1, 2, 4, 8, 1, 9, 5, 4, 3, 8, 5,
                10, 8, 6, 6, 1, 0, 6, 10, 8, 2, 3, 4,
            ]) - 4.77778)
                <= 0.00001
        );
    }

    #[test]
    fn test_max_length_between_equal_characters() {
        assert_eq!(0, max_length_between_equal_characters(String::from("aa")));
        assert_eq!(2, max_length_between_equal_characters(String::from("abca")));
        assert_eq!(
            -1,
            max_length_between_equal_characters(String::from("cbzyx"))
        );
    }

    #[test]
    fn test_frequency_sort() {
        assert_eq!(
            vec![3, 1, 1, 2, 2, 2],
            frequency_sort(vec![1, 1, 2, 2, 2, 3])
        );
        assert_eq!(vec![1, 3, 3, 2, 2], frequency_sort(vec![2, 3, 1, 3, 2]));
        assert_eq!(
            vec![5, -1, 4, 4, -6, -6, 1, 1, 1],
            frequency_sort(vec![-1, 1, -6, 4, 5, -6, 1, 4, 1])
        );
    }

    #[test]
    fn test_min_operations() {
        assert_eq!(min_operations(vec![1, 1, 4, 2, 3], 5), 2);
        assert_eq!(min_operations(vec![5, 6, 7, 8, 9], 4), -1);
        assert_eq!(min_operations(vec![3, 2, 20, 1, 1, 3], 10), 5);
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

    #[test]
    fn test_most_competitive() {
        assert_eq!(most_competitive(vec![3, 5, 2, 6], 2), vec![2, 6]);
        assert_eq!(most_competitive(vec![2, 4, 3, 3, 5, 4, 9, 6], 4), vec![2, 3, 3, 4]);
    }

    #[test]
    fn test_reformat_number() {
        assert_eq!(
            reformat_number(String::from("1-23-45 6")),
            String::from("123-456")
        );
        assert_eq!(
            reformat_number(String::from("123 4-567")),
            String::from("123-45-67")
        );
        assert_eq!(
            reformat_number(String::from("123 4-5678")),
            String::from("123-456-78")
        );
    }

    #[test]
    fn test_count_students() {
        assert_eq!(count_students(vec![1, 1, 0, 0], vec![0, 1, 0, 1]), 0);
        assert_eq!(
            count_students(vec![1, 1, 1, 0, 0, 1], vec![1, 0, 0, 0, 1, 1]),
            3
        );
    }

    #[test]
    fn test_maximum_binary_string() {
        assert_eq!(
            maximum_binary_string("000110".to_string()),
            String::from("111011")
        );
        assert_eq!(maximum_binary_string("01".to_string()), String::from("01"));
        assert_eq!(maximum_binary_string("11".to_string()), String::from("11"));
        assert_eq!(
            maximum_binary_string("1000".to_string()),
            String::from("1110")
        );
    }

    #[test]
    fn test_minimum_length() {
        assert_eq!(2, minimum_length(String::from("ca")));
        assert_eq!(0, minimum_length(String::from("cabaabac")));
        assert_eq!(3, minimum_length(String::from("aabccabba")));
    }

    #[test]
    fn test_merge_alternately() {
        assert_eq!(
            String::from("apbqcr"),
            merge_alternately(String::from("abc"), String::from("pqr"))
        );
        assert_eq!(
            String::from("apbqrs"),
            merge_alternately(String::from("ab"), String::from("pqrs"))
        );
        assert_eq!(
            String::from("apbqcd"),
            merge_alternately(String::from("abcd"), String::from("pq"))
        );
    }

    #[test]
    fn test_count_matches() {
        assert_eq!(
            1,
            count_matches(
                generate_string_matrix(vec![
                    vec!["phone", "blue", "pixel"],
                    vec!["computer", "silver", "lenovo"],
                    vec!["phone", "gold", "iphone"],
                ]),
                String::from("color"),
                String::from("silver"),
            )
        );
        assert_eq!(
            2,
            count_matches(
                generate_string_matrix(vec![
                    vec!["phone", "blue", "pixel"],
                    vec!["computer", "silver", "phone"],
                    vec!["phone", "gold", "iphone"],
                ]),
                String::from("type"),
                String::from("phone"),
            )
        );
    }

    #[test]
    fn test_nearest_valid_point() {
        assert_eq!(
            nearest_valid_point(
                3,
                4,
                vec![vec![1, 2], vec![3, 1], vec![2, 4], vec![2, 3], vec![4, 4]],
            ),
            2
        );
        assert_eq!(nearest_valid_point(3, 4, vec![vec![3, 4]]), 0);
        assert_eq!(nearest_valid_point(3, 4, vec![vec![2, 3]]), -1);
    }

    #[test]
    fn test_check_ones_segment() {
        assert!(!check_ones_segment(String::from("1001")));
        assert!(check_ones_segment(String::from("110")));
    }

    #[test]
    fn test_are_almost_equal() {
        assert!(are_almost_equal(String::from("bank"), String::from("kanb")));
        assert!(!are_almost_equal(
            String::from("attack"),
            String::from("defend"),
        ));
        assert!(are_almost_equal(
            String::from("kelb"),
            String::from("kelbf"),
        ));
    }

    #[test]
    fn test_max_value() {
        assert_eq!(2, max_value(4, 2, 6));
        assert_eq!(3, max_value(6, 1, 10));
    }

    #[test]
    fn test_array_sign() {
        assert_eq!(array_sign(vec![-1, -2, -3, -4, 3, 2, 1]), 1);
        assert_eq!(array_sign(vec![1, 5, 0, 2, -3]), 0);
        assert_eq!(array_sign(vec![-1, 1, -1, 1, -1]), -1);
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

    #[test]
    fn test_first_day_been_in_all_rooms() {
        assert_eq!(first_day_been_in_all_rooms(vec![0, 0]), 2);
        assert_eq!(first_day_been_in_all_rooms(vec![0, 0, 2]), 6);
        assert_eq!(first_day_been_in_all_rooms(vec![0, 1, 2, 0]), 6);
    }

    #[test]
    fn test_final_value_after_operations() {
        assert_eq!(
            final_value_after_operations(generate_string_vec(vec!["--X", "X++", "X++"])),
            1
        );
        assert_eq!(
            final_value_after_operations(generate_string_vec(vec!["++X", "++X", "X++"])),
            3
        );
        assert_eq!(
            final_value_after_operations(generate_string_vec(vec!["X++", "++X", "--X", "X--"])),
            0
        );
    }

    #[test]
    fn test_minimum_moves() {
        assert_eq!(minimum_moves(String::from("XXX")), 1);
        assert_eq!(minimum_moves(String::from("XXOX")), 2);
        assert_eq!(minimum_moves(String::from("OOOO")), 0);
    }

    macro_rules! two_out_of_three_tests {
        ($($name:ident: $value:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    let (nums1, nums2, nums3, mut expected) = $value;
                    let mut actual = two_out_of_three(nums1, nums2, nums3);
                    actual.sort();
                    expected.sort_by(|a: &i32, b: &i32| a.cmp(b));
                    assert_eq!(actual, expected);
                }
            )*
        };
    }

    two_out_of_three_tests! {
        test_two_out_of_three_case0: (vec![1, 1, 3, 2], vec![2, 3], vec![3], vec![3, 2]),
        test_two_out_of_three_case1: (vec![3, 1], vec![2, 3], vec![1, 2], vec![2, 3, 1]),
        test_two_out_of_three_case2: (vec![1, 2, 2], vec![4, 3, 3], vec![5], vec![]),
    }

    #[test]
    fn test_min_moves_to_seat() {
        assert_eq!(min_moves_to_seat(vec![3, 1, 5], vec![2, 7, 4]), 4);
        assert_eq!(min_moves_to_seat(vec![4, 1, 5, 9], vec![1, 3, 2, 6]), 7);
        assert_eq!(min_moves_to_seat(vec![2, 2, 6, 6], vec![1, 3, 2, 6]), 4);
    }

    #[test]
    fn test_are_number_ascending() {
        assert!(are_number_ascending(String::from(
            "1 box has 3 blue 4 red 6 green and 12 yellow marbles"
        )));
        assert!(!are_number_ascending(String::from("hello world 5 x 5")));
        assert!(!are_number_ascending(String::from(
            "sunset is at 7 51 pm overnight lows will be in the low 50 and 60 s"
        )));
    }

    #[test]
    fn test_count_even() {
        assert_eq!(count_even(4), 2);
        assert_eq!(count_even(30), 14);
    }

    #[test]
    fn test_prefix_count() {
        assert_eq!(
            prefix_count(
                generate_string_vec(vec!["pay", "attention", "practice", "attend"]),
                String::from("at"),
            ),
            2
        );
        assert_eq!(
            prefix_count(
                generate_string_vec(vec!["leetcode", "win", "loops", "success"]),
                String::from("code"),
            ),
            0
        );
    }

    #[test]
    fn test_minimum_rounds() {
        assert_eq!(minimum_rounds(vec![2, 2, 3, 3, 2, 4, 4, 4, 4, 4]), 4);
        assert_eq!(minimum_rounds(vec![2, 3, 3]), -1);
    }

    #[test]
    fn test_repeated_character() {
        assert_eq!(repeated_character(String::from("abccbaacz")), 'c');
        assert_eq!(repeated_character(String::from("abcdd")), 'd');
        assert_eq!(repeated_character(String::from("aa")), 'a');
        assert_eq!(repeated_character(String::from("zz")), 'z');
    }

    #[test]
    fn test_garbage_collection() {
        assert_eq!(
            garbage_collection(
                generate_string_vec(vec!["G", "P", "GP", "GG"]),
                vec![2, 4, 3]
            ),
            21
        );
        assert_eq!(
            garbage_collection(generate_string_vec(vec!["MMM", "PGM", "GP"]), vec![3, 10]),
            37
        );
    }

    #[test]
    fn test_count_ways() {
        assert_eq!(count_ways(vec![vec![6, 10], vec![5, 15]]), 2);
        assert_eq!(
            count_ways(vec![vec![1, 3], vec![10, 20], vec![2, 5], vec![4, 8]]),
            4
        );
    }

    #[test]
    fn test_find_indices() {
        assert_eq!(find_indices(vec![5, 1, 4, 1], 2, 4), vec![0, 3]);
        assert_eq!(find_indices(vec![2, 1], 0, 0), vec![0, 0]);
        assert_eq!(find_indices(vec![1, 2, 3], 2, 4), vec![-1, -1]);
    }

    #[test]
    fn test_minimum_sum() {
        assert_eq!(9, minimum_sum(vec![8, 6, 1, 5, 3]));
        assert_eq!(13, minimum_sum(vec![5, 4, 8, 7, 10, 2]));
        assert_eq!(-1, minimum_sum(vec![6, 5, 4, 3, 4, 5]));
    }

    #[test]
    fn test_minimum_added_coins() {
        assert_eq!(2, minimum_added_coins(vec![1, 4, 10], 19));
        assert_eq!(1, minimum_added_coins(vec![1, 4, 10, 5, 7, 19], 19));
        assert_eq!(3, minimum_added_coins(vec![1, 1, 1], 20));
    }
}
