use crate::leetcode::data_structures::ListNode;
use crate::leetcode::TreeNode;
use std::cell::RefCell;
use std::cmp::{max, Ordering};
use std::collections::HashSet;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::thread::scope;

// 1. Two Sum
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

// 2. Add Two Numbers
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

// 7. Convert Integer
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

// 27. Remove Element
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

// 172. Factorial Trailing Zeroes
pub fn trailing_zeroes(n: i32) -> i32 {
    let mut ans = 0;
    let mut n = n;
    while n != 0 {
        n /= 5;
        ans += n;
    }
    ans
}

// 357. Count Numbers with Unique Digits
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

// 386. Lexicographical Numbers
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

// 498. Diagonal Traverse
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

// 504. Base 7
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

// 537. Complex Number Multiplication
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

// 636. Exclusive Time of Functions
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

// 646. Maximum Length of Pair Chain
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

// 658. Find K Closest Elements
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

// 662. Maximum Width of Binary Tree
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

// 670. Maximum Swap
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

// 672. Bulb Switcher II
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

// 682. Baseball Game
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

// 693. Binary Number with Alternating bits
pub fn has_alternating_bits(n: i32) -> bool {
    let a = n ^ (n >> 1);
    a & (a + 1) == 0
}

// 724. Find Pivot Index
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

// 728. Self Dividing Numbers
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

// 744. Find Smallest Letter Greater Than Target
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

// 762. Prime Number of Set Bits in Binary Representation
pub fn count_prime_set_bits(left: i32, right: i32) -> i32 {
    (left..=right).fold(0, |ret, i| ret + (665772 >> i.count_ones() & 1))
}

// 793. Preimage Size of Factorial Zeroes Function
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

// 804. Unique Morse Code Words
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

// 806. Number of Lines To Write String
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

// 811. Subdomain Visit Count
pub fn subdomain_visits(cpdomains: Vec<String>) -> Vec<String> {
    let mut map = HashMap::<&str, usize>::new();
    let mut count = 0;
    for cpdomain in &cpdomains {
        for (i, v) in cpdomain.as_bytes().iter().enumerate() {
            if *v == ' ' as u8 {
                count = cpdomain[..i].parse::<usize>().unwrap();
                map.entry(&cpdomain[i + 1..]).and_modify(|x| *x += count).or_insert(count);
                continue;
            }
            if *v == '.' as u8 {
                map.entry(&cpdomain[i + 1..]).and_modify(|x| *x += count).or_insert(count);
            }
        }
    }
    map.iter().map(|(s, n)| format!("{} {}", n, s)).collect()
}

// 821. Shortest Distance to a Character
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

// 828. Count Unique Characters of All Substrings of a Given String
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

// 856. Score of Parentheses
pub fn score_of_parentheses(s: String) -> i32 {
    let mut chars = s.as_bytes();
    chars.iter().enumerate().fold((0, 0), |(mut l, ret), (i, &ch)| {
        l += if ch == b'(' { 1 } else { -1 };
        (l, if ch == b')' && chars[i - 1] == b'(' { ret + (1 << l) } else { ret })
    }).1
}

//  883. Projection Area of 3D Shapes
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

// 905. Sort Array By Parity
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

// 921. Minimum Add to Make Parentheses Valid
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

// 927. Three Equal Parts
pub fn three_equal_parts(arr: Vec<i32>) -> Vec<i32> {
    let sum: i32 = arr.iter().sum();
    if sum % 3 != 0 {
        return vec![-1, -1];
    }
    if sum == 0 {
        return vec![0, 2];
    }
    let mut partial = sum / 3;
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

// 942. DI String Match
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

// 944. Delete Columns to Make Sorted
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

// 946. Validate Stack Sequences
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

// 998. Maximum Binary Tree II
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

// 1403. Minimum Subsequence in Non-Increasing Order
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

// 1408. String Matching in an Array
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

// 1450. Number of Students Doing Homework at a Given Time
pub fn busy_student(start_time: Vec<i32>, end_time: Vec<i32>, query_time: i32) -> i32 {
    let mut res = 0;
    for i in 0..start_time.len() {
        if start_time[i] <= query_time && query_time <= end_time[i] {
            res += 1;
        }
    }
    res
}

// 1455. Check If a Word Occurs As a Prefix of Any Word in a Sentence
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

// 1460. Make Two Arrays Equal by Reversing Sub-arrays
pub fn can_be_equal(mut target: Vec<i32>, mut arr: Vec<i32>) -> bool {
    target.sort();
    arr.sort();
    target == arr
}

// 1464. Maximum Product of Two Elements in an Array
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

// 1470. Shuffle the Array
pub fn shuffle(nums: Vec<i32>, n: i32) -> Vec<i32> {
    let mut res: Vec<i32> = vec![0; (n * 2) as usize];
    let n = n as usize;
    for i in 0..n {
        res[2 * i] = nums[i];
        res[2 * i + 1] = nums[n + i];
    }
    res
}

// 1475. Final Prices With a Special Discount in a Shop
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

// 1582. Special Positions in a Binary Matrix
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

// 1608. Special Array With X Elements Greater Than or Equal X
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

// 1619. Mean of Array After Removing Some Elements
pub fn trim_mean(mut arr: Vec<i32>) -> f64 {
    let n = arr.len();
    arr.sort();
    arr[n / 20..(19 * n / 20)].iter().sum::<i32>() as f64 / (n as f64 * 0.9)
}

// 1624. Largest Substring Between Two Equal Characters
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

// 1636. Sort Array by Increasing Frequency
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

// 1672. Richest Customer Wealth
pub fn maximum_wealth(accounts: Vec<Vec<i32>>) -> i32 {
    accounts.iter().map(|x| x.iter().sum()).max().unwrap()
}

// 1694. Reformat Phone Number
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

// 1784. Check if Binary String Has at Most One Segment of Ones
pub fn check_ones_segment(s: String) -> bool {
    !s.contains("01")
}

// 1800. Maximum Ascending Subarray Sum
pub fn max_ascending_sum(nums: Vec<i32>) -> i32 {
    let (mut res, mut i) = (0, 0);
    let n = nums.len();
    while i < n {
        let mut curr = nums[i];
        i += 1;
        while i < n && nums[i] > nums[i-1] {
            curr += nums[i];
            i += 1
        }
        res = max(res, curr);
    }
    res
}

// 1823. Find the Winner of the Circular Game
pub fn find_the_winner(n: i32, k: i32) -> i32 {
    let mut winner = 1;
    for i in 2..=n {
        winner = (winner + k - 1) % i + 1;
    }
    winner
}

// 1991. Find the Middle Index in Array
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
        assert_eq!(remove_element(&mut vec![0, 1, 2, 2, 3, 0, 4, 2], 2), 5);
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
        let v1 = vec!["5", "2", "C", "D", "+"]
            .iter()
            .map(|&x| x.to_string())
            .collect();
        let v2 = vec!["5", "-2", "4", "C", "D", "9", "+", "+"]
            .iter()
            .map(|&x| x.to_string())
            .collect();
        assert_eq!(cal_points(v1), 30);
        assert_eq!(cal_points(v2), 27);
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
        let v1 = vec![
            String::from("gin"),
            String::from("zen"),
            String::from("gig"),
            String::from("msg"),
        ]
            .into_iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>();
        assert_eq!(unique_morse_representations(v1), 2);
        assert_eq!(unique_morse_representations(vec![String::from("a")]), 1);
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
    fn test_subdomain_visits() {
        let cpdomains1 = vec!["9001 discuss.leetcode.com"]
            .iter()
            .map(|&x| x.to_string())
            .collect::<Vec<String>>();
        let cpdomains2 = vec!["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]
            .iter()
            .map(|&x| x.to_string())
            .collect::<Vec<String>>();
        let expected1 = vec!["9001 leetcode.com", "9001 discuss.leetcode.com", "9001 com"]
            .iter()
            .map(|&x| x.to_string())
            .collect::<Vec<String>>();
        let expected2 = vec!["901 mail.com", "50 yahoo.com", "900 google.mail.com", "5 wiki.org", "5 org", "1 intel.mail.com", "951 com"]
            .iter()
            .map(|&x| x.to_string())
            .collect::<Vec<String>>();
        assert_eq!(subdomain_visits(cpdomains1), expected1);
        assert_eq!(subdomain_visits(cpdomains2), expected2);
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
        assert_eq!(unique_letter_string("ABC".to_string()), 10);
        assert_eq!(unique_letter_string("ABA".to_string()), 8);
        assert_eq!(unique_letter_string("LEETCODE".to_string()), 92);
    }

    #[test]
    fn test_score_of_parentheses() {
        assert_eq!(1, score_of_parentheses("()".to_string()));
        assert_eq!(2, score_of_parentheses("(())".to_string()));
        assert_eq!(2, score_of_parentheses("()()".to_string()));
    }

    #[test]
    fn test_projection_area() {
        assert_eq!(projection_area(vec![vec![1, 2], vec![3, 4]]), 17);
        assert_eq!(projection_area(vec![vec![2]]), 5);
        assert_eq!(projection_area(vec![vec![1, 0], vec![0, 2]]), 8);
    }

    #[test]
    fn test_sort_array_by_parity() {
        assert_eq!(sort_array_by_parity(vec![3, 1, 2, 4]), vec![4, 2, 1, 3]);
        assert_eq!(sort_array_by_parity(vec![0]), vec![0]);
    }

    #[test]
    fn test_min_add_to_make_valid() {
        assert_eq!(min_add_to_make_valid("())".to_string()), 1);
        assert_eq!(min_add_to_make_valid("(((".to_string()), 3);
    }

    #[test]
    fn test_three_equal_parts() {
        assert_eq!(three_equal_parts(vec![1, 0, 1, 0, 1]), vec![0, 3]);
        assert_eq!(three_equal_parts(vec![1, 1, 0, 1, 1]), vec![-1, -1]);
        assert_eq!(three_equal_parts(vec![1, 1, 0, 0, 1]), vec![0, 2]);
    }

    #[test]
    fn test_di_string_match() {
        assert_eq!(di_string_match(String::from("IDID")), vec![0, 4, 1, 3, 2]);
        assert_eq!(di_string_match(String::from("III")), vec![0, 1, 2, 3]);
        assert_eq!(di_string_match(String::from("DDI")), vec![3, 2, 0, 1]);
    }

    #[test]
    fn test_min_deletion_size() {
        let v1 = vec!["cba", "daf", "ghi"]
            .iter()
            .map(|&x| x.to_string())
            .collect();
        let v2 = vec!["a", "b"].iter().map(|&x| x.to_string()).collect();
        let v3 = vec!["zyx", "wvu", "tsr"]
            .iter()
            .map(|&x| x.to_string())
            .collect();
        assert_eq!(min_deletion_size(v1), 1);
        assert_eq!(min_deletion_size(v2), 0);
        assert_eq!(min_deletion_size(v3), 3);
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
        assert_eq!(can_be_equal(vec![1, 2, 3, 4], vec![2, 1, 3, 4]), true);
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
        assert_eq!(
            (trim_mean(vec![
                1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3,
            ]) - 2.00000)
                <= 0.00001,
            true
        );
        assert_eq!(
            (trim_mean(vec![
                6, 2, 7, 5, 1, 2, 0, 3, 10, 2, 5, 0, 5, 5, 0, 8, 7, 6, 8, 0,
            ]) - 4.00000)
                <= 0.00001,
            true
        );
        assert_eq!(
            (trim_mean(vec![
                6, 0, 7, 0, 7, 5, 7, 8, 3, 4, 0, 7, 8, 1, 6, 8, 1, 1, 2, 4, 8, 1, 9, 5, 4, 3, 8, 5,
                10, 8, 6, 6, 1, 0, 6, 10, 8, 2, 3, 4,
            ]) - 4.77778)
                <= 0.00001,
            true
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
    fn test_maximum_wealth() {
        assert_eq!(maximum_wealth(vec![vec![1, 2, 3], vec![3, 2, 1]]), 6);
        assert_eq!(maximum_wealth(vec![vec![1, 5], vec![7, 3], vec![3, 5]]), 10);
        assert_eq!(
            maximum_wealth(vec![vec![2, 8, 7], vec![7, 1, 3], vec![1, 9, 5]]),
            17
        );
    }

    #[test]
    fn test_reformat_number() {
        assert_eq!(
            reformat_number("1-23-45 6".to_string()),
            "123-456".to_string()
        );
        assert_eq!(
            reformat_number("123 4-567".to_string()),
            "123-45-67".to_string()
        );
        assert_eq!(
            reformat_number("123 4-5678".to_string()),
            "123-456-78".to_string()
        );
    }

    #[test]
    fn test_check_ones_segment() {
        assert_eq!(check_ones_segment("1001".to_string()), false);
        assert_eq!(check_ones_segment("110".to_string()), true);
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
