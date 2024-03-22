#![cfg_attr(debug_assertions, allow(dead_code))]

pub struct StockSpanner {
    stack: Vec<(i32, i32)>,
    index: i32,
}

impl StockSpanner {
    pub fn new() -> Self {
        StockSpanner {
            stack: vec![(-1, i32::MAX)],
            index: -1,
        }
    }

    pub fn next(&mut self, price: i32) -> i32 {
        self.index += 1;
        while price >= self.stack.last().unwrap().1 {
            self.stack.pop();
        }
        let res = self.index - self.stack.last().unwrap().0;
        self.stack.push((self.index, price));
        res
    }
}

#[cfg(test)]
mod tests {
    use crate::leetcode::StockSpanner;

    #[test]
    fn test_stock_spanner() {
        let mut stock_spanner = StockSpanner::new();
        assert_eq!(1, stock_spanner.next(100));
        assert_eq!(1, stock_spanner.next(80));
        assert_eq!(1, stock_spanner.next(60));
        assert_eq!(2, stock_spanner.next(70));
        assert_eq!(1, stock_spanner.next(60));
        assert_eq!(4, stock_spanner.next(75));
        assert_eq!(6, stock_spanner.next(85));
    }
}
