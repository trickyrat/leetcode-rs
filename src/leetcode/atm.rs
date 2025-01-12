#![cfg_attr(debug_assertions, allow(dead_code))]

pub struct ATM {
    count: Vec<i64>,
    value: Vec<i64>,
}

impl ATM {
    fn new() -> Self {
        ATM {
            count: vec![0, 0, 0, 0, 0],
            value: vec![20, 50, 100, 200, 500],
        }
    }

    fn deposit(&mut self, banknotes_count: Vec<i32>) {
        for i in 0..5 {
            self.count[i] += banknotes_count[i] as i64;
        }
    }

    fn widthdraw(&mut self, mut amount: i32) -> Vec<i32> {
        let mut res: Vec<i32> = vec![0;5];
        for i in (0..5).rev() {
            res[i] = std::cmp::min(self.count[i], amount as i64 / self.value[i]) as i32;
            amount -= res[i] * self.value[i] as i32;
        }

        if amount > 0 {
            vec![-1]
        } else {
            for i in 0..5 {
                self.count[i] -= res[i] as i64;
            }
            res
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::leetcode::atm::ATM;

    #[test]
    fn test_atm() {
        let mut atm = ATM::new();
        atm.deposit(vec![0, 0, 1, 2, 1]);
        assert_eq!(vec![0, 0, 1, 0, 1], atm.widthdraw(600));
        atm.deposit(vec![0, 1, 0, 1, 1]);
        assert_eq!(vec![-1], atm.widthdraw(600));
        assert_eq!(vec![0, 1, 0, 0, 1], atm.widthdraw(550));
    }
}