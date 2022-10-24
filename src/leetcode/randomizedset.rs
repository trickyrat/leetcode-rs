use rand::{thread_rng, Rng};
use std::collections::HashMap;

pub struct RandomizedSet {
    nums: Vec<i32>,
    indices: HashMap<i32, usize>,
}

impl RandomizedSet {
    pub fn new() -> Self {
        RandomizedSet {
            nums: Vec::new(),
            indices: HashMap::new(),
        }
    }

    pub fn insert(&mut self, val: i32) -> bool {
        if self.indices.contains_key(&val) {
            return false;
        }
        self.indices.insert(val, self.nums.len());
        self.nums.push(val);
        true
    }

    pub fn remove(&mut self, val: i32) -> bool {
        let index = match self.indices.remove(&val) {
            None => {
                return false;
            }
            Some(val) => val,
        };
        let last = self.nums.pop().unwrap();
        if index != self.nums.len() {
            self.nums[index] = last;
            self.indices.insert(last, index);
        }
        true
    }

    pub fn get_random(&self) -> i32 {
        let mut rng = thread_rng();
        self.nums[rng.gen_range(0..self.nums.len())]
    }
}

#[cfg(test)]
mod tests {
    use crate::leetcode::RandomizedSet;

    #[test]
    fn test_randomize_set() {
        let mut randomized_set = RandomizedSet::new();
        assert_eq!(randomized_set.insert(1), true);
        assert_eq!(randomized_set.remove(2), false);
        assert_eq!(randomized_set.insert(2), true);
        let v = vec![1, 2];
        assert!(v.contains(&randomized_set.get_random()));
        assert_eq!(randomized_set.remove(1), true);
        assert_eq!(randomized_set.insert(2), false);
        assert_eq!(randomized_set.get_random(), 2);
    }
}
