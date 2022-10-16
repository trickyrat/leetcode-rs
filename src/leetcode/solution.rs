use crate::leetcode::data_structures::ListNode;
use crate::leetcode::TreeNode;
use std::cell::RefCell;
use std::cmp::{max, min, Ordering};
use std::collections::HashSet;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

/// 1.Two Sum
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

/// 2.Add Two Numbers
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

/// 7.Convert Integer
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

/// 498.Diagonal Traverse
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
        digits += &"-".to_string();
    }
    digits.chars().rev().collect()
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

/// 670.Maximum Swap
pub fn maximum_swap(num: i32) -> i32 {
    let mut chars = num.to_string().chars().collect::<Vec<char>>();
    let n = chars.len();
    let mut max_index = n - 1;
    let mut index1 = 0;
    let mut index2 = 0;
    for i in (0..n).rev() {
        if chars[i] > chars[max_index] {
            max_index = i;
        } else if chars[i] < chars[max_index] {
            index1 = i;
            index2 = max_index;
        }
    }
    if index1 >= 0 {
        chars.swap(index1, index2);
        return chars.iter().collect::<String>().parse::<i32>().unwrap();
    }
    num
}

/// 672.Bulb Switcher II
pub fn flip_lights(n: i32, presses: i32) -> i32 {
    let mut seen: HashSet<i32> = HashSet::new();
    for i in 0..16 {
        let mut press_array: Vec<i32> = vec![0; 4];
        for j in 0..4 {
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
    for i in 0..nums.len() {
        if 2 * sum + nums[i] == total {
            return i as i32;
        }
        sum += nums[i];
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
    return letters[low];
}

/// 762.Prime Number of Set Bits in Binary Representation
pub fn count_prime_set_bits(left: i32, right: i32) -> i32 {
    (left..=right).fold(0, |ret, i| ret + (665772 >> i.count_ones() & 1))
}

/// 769. Max Chunks To Make Sorted
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
        return (right + 1) as i32;
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
                .for_each(|ch| s = format!("{}{}", s, morse[(ch - 'a' as u8) as usize]));
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
            if *v == ' ' as u8 {
                count = cpdomain[..i].parse::<usize>().unwrap();
                map.entry(&cpdomain[i + 1..])
                    .and_modify(|x| *x += count)
                    .or_insert(count);
                continue;
            }
            if *v == '.' as u8 {
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
        if !index.contains_key(&c) {
            index.insert(c, vec![-1]);
        }
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

/// 1441. Build an Array With Stack Operations
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
        if color[i as usize] == 0 && !dfs(i, 1, &mut color, &group) {
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

/// 921.Minimum Add to Make Parentheses Valid
pub fn min_add_to_make_valid(s: String) -> i32 {
    let mut res = 0;
    let mut left_count = 0;
    for ch in s.chars() {
        if ch == '(' {
            left_count += 1;
        } else {
            if left_count > 0 {
                left_count -= 1;
            } else {
                res += 1;
            }
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

/// 940. Distinct Subsequences II
pub fn distinct_subseq_ii(s: String) -> i32 {
    let MOD = 1000000007;
    let mut alphas = vec![0; 26];
    let mut res = 0;
    s.chars().for_each(|x| {
        let index = ((x as u8) - b'a') as usize;
        let prev = alphas[index];
        alphas[index] = (res + 1) % MOD;
        res = ((res + alphas[index] - prev) % MOD + MOD) % MOD;
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
    curr.as_ref().unwrap().borrow_mut().right = Some(parent.clone());
    root
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

/// 1441. Build an Array With Stack Operations
pub fn build_array(target: Vec<i32>, n: i32) -> Vec<String> {
    let mut res = Vec::<String>::new();
    let mut prev = 0;
    target.iter().for_each(|&x| {
        for i in 0..(x - prev - 1) {
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
    for num in mat[0].to_vec() {
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
        if !map.contains_key(&ch) {
            map.insert(ch, i);
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
        .for_each(|&num| *count.entry(num).or_default() += 1);
    nums.sort_by(
        |x, y| match count.get(x).unwrap().cmp(count.get(y).unwrap()) {
            Ordering::Greater => Ordering::Greater,
            Ordering::Less => Ordering::Less,
            _ => y.cmp(x),
        },
    );
    nums
}

/// 1672.Richest Customer Wealth
pub fn maximum_wealth(accounts: Vec<Vec<i32>>) -> i32 {
    accounts.iter().map(|x| x.iter().sum()).max().unwrap()
}

/// 1694.Reformat Phone Number
pub fn reformat_number(number: String) -> String {
    let mut digits = String::new();
    for ch in number.chars() {
        if ch.is_digit(10) {
            digits.push(ch);
        }
    }
    let mut n = digits.len();
    let mut res = String::new();
    let mut pt = 0;
    while n > 0 {
        if n > 4 {
            res.push_str(&digits[pt..pt + 3]);
            res.push_str("-");
            pt += 3;
            n -= 3;
        } else {
            if n == 4 {
                res.push_str(&digits[pt..pt + 2]);
                res.push_str("-");
                res.push_str(&digits[pt + 2..pt + 4]);
            } else {
                res.push_str(&digits[pt..pt + n]);
            }
            break;
        }
    }
    res.to_string()
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
    for i in 0..nums.len() {
        if 2 * sum + nums[i] == total {
            return i as i32;
        }
        sum += nums[i];
    }
    -1
}
